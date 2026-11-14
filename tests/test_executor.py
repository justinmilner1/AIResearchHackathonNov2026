"""Spec -> config, run directory -> result row, and the technique registry. No model, no GPU."""

import json

import numpy as np
import pytest

from clharness.config import REPO_ROOT
from clharness.executor.rows import half_indices, measure_run
from clharness.executor.run import config_from_spec, run_name, run_spec
from clharness.spec import RunSpec, load_spec
from clharness.techniques import REGISTRY, build_hooks, register
from clharness.train.measure import default_hooks

SPECS = REPO_ROOT / "configs" / "bench" / "specs"
TASKS = ["arc_easy", "piqa"]


def test_committed_specs_resolve_against_the_fast_bench():
    for path in sorted(SPECS.glob("*.json")):
        spec = load_spec(path)
        cfg = config_from_spec(spec)
        assert cfg.name == run_name(spec) == spec.spec_hash()[:16]
        assert [(d.name, d.tokens) for d in cfg.data.domains] == [(d.name, d.tokens) for d in spec.stream]


def test_spec_overrides_reach_the_config():
    spec = RunSpec(name="x", seed=7, stream=[{"name": "code", "tokens": 4096}],
                   optim={"lr": 3e-5, "batch_size": 16})
    cfg = config_from_spec(spec, output_dir="/tmp/somewhere")
    assert (cfg.train.seed, cfg.optim.lr, cfg.optim.batch_size) == (7, 3e-5, 16)
    assert cfg.train.output_dir == "/tmp/somewhere"
    assert cfg.optim.weight_decay == 0.1  # untouched bench default


def test_fast_bench_switches_the_expensive_instruments_off():
    cfg = config_from_spec(load_spec(SPECS / "no-mitigation-s0.json"))
    assert [h.name for h in default_hooks(cfg)] == ["eval"]
    assert len(default_hooks()) == 4  # the pilot setup is unchanged
    assert {t.task for t in cfg.eval.lm_eval_tasks} == {"arc_easy", "hellaswag", "piqa", "pubmedqa_pqal"}


def test_unknown_bench_and_unknown_technique_become_failed_rows():
    row = run_spec(RunSpec(name="x", seed=0, bench="nope", stream=[{"name": "code", "tokens": 1}]))
    assert row.status == "failed" and "unknown bench" in row.reason
    row = run_spec(RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 1}],
                           techniques=[{"name": "no_such_technique"}]))
    assert row.status == "failed" and "unknown technique" in row.reason


def test_registry_builds_hooks_in_order_with_params():
    @register("_test_a")
    class A:
        def __init__(self, x: int = 1):
            self.x = x

    try:
        spec = RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 1}],
                       techniques=[{"name": "_test_a", "params": {"x": 5}}])
        (hook,) = build_hooks(spec.techniques)
        assert isinstance(hook, A) and hook.x == 5
    finally:
        del REGISTRY["_test_a"]


def test_halves_are_disjoint_stable_and_cover_everything():
    sel, rep = half_indices("arc_easy", 500)
    assert len(sel) == len(rep) == 250
    assert sorted([*sel, *rep]) == list(range(500))
    assert (half_indices("arc_easy", 500)[0] == sel).all()
    assert not (half_indices("piqa", 500)[0] == sel).all()  # each task gets its own shuffle


def _ckpt(run, index, domain, boundary, tokens, correct, ppl, share=0.4):
    d = run / "checkpoints" / f"ckpt_{index:04d}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"index": index, "domain": domain, "is_domain_boundary": boundary, "gradient_tokens": tokens}))
    metrics = {f"eval/ppl_doc/{k}": v for k, v in ppl.items()}
    for task in TASKS:
        np.save(d / f"items_{task}.npy", correct[task].astype(np.float32))
        metrics[f"eval/lm_eval/{task}/answer_share_max"] = share
    (d / "metrics.json").write_text(json.dumps(metrics))


def _run_dir(tmp_path, final_share=0.4):
    """Base: every item right. After code: the selection half of arc_easy all wrong.
    After biomedical: additionally the report half of piqa all wrong."""
    run = tmp_path / "run"
    ones = {t: np.ones(100) for t in TASKS}
    after_code = {t: v.copy() for t, v in ones.items()}
    after_code["arc_easy"][half_indices("arc_easy", 100)[0]] = 0
    after_bio = {t: v.copy() for t, v in after_code.items()}
    after_bio["piqa"][half_indices("piqa", 100)[1]] = 0
    _ckpt(run, 0, "base", True, 0, ones, {"code": 10.0, "biomedical": 20.0, "general": 30.0})
    _ckpt(run, 1, "code", True, 1000, after_code, {"code": 5.0, "biomedical": 22.0, "general": 33.0})
    _ckpt(run, 2, "biomedical", True, 2000, after_bio,
          {"code": 8.0, "biomedical": 10.0, "general": 36.0}, share=final_share)
    return run


def test_measure_run_reads_halves_boundaries_and_perplexity(tmp_path):
    m = measure_run(_run_dir(tmp_path), ["code", "biomedical"])
    sel, rep = m["retention"].selection, m["retention"].report
    assert sel.per_task["arc_easy"].delta == pytest.approx(-1.0)
    assert rep.per_task["arc_easy"].delta == pytest.approx(0.0)
    assert rep.per_task["piqa"].delta == pytest.approx(-1.0)
    assert (sel.mean_delta, rep.mean_delta) == (pytest.approx(-0.5), pytest.approx(-0.5))
    b = m["retention"].boundaries
    assert [(x.domain, x.gradient_tokens) for x in b] == [("code", 1000), ("biomedical", 2000)]
    assert (b[0].selection_mean_delta, b[0].report_mean_delta) == (pytest.approx(-0.5), pytest.approx(0.0))
    assert m["acquisition"].ppl_ratio == {"code": pytest.approx(0.8), "biomedical": pytest.approx(0.5)}
    assert m["acquisition"].mean_ppl_ratio == pytest.approx((0.8 * 0.5) ** 0.5)
    assert m["acquisition"].general_ppl_ratio == pytest.approx(1.2)
    assert m["gradient_tokens"] == 2000 and m["invalid_reason"] is None


def test_answer_collapse_marks_the_run_invalid(tmp_path):
    m = measure_run(_run_dir(tmp_path, final_share=0.97), ["code", "biomedical"])
    assert "answer collapse" in m["invalid_reason"]


def test_measure_run_agrees_with_a_september_baseline():
    """pilot-chosen-200M lost 7.4 points on the multiple-choice mean (docs/baselines.md).
    The other pilot runs are not used here: their boundary item files were written after the
    recoverability probe, so they differ slightly from the logged scores."""
    run = REPO_ROOT / "results" / "runs" / "pilot-chosen-200M"
    m = measure_run(run, ["code", "biomedical", "legal", "math"])
    r = m["retention"]
    mc = [t for t in r.report.per_task if t != "mbpp"]
    both = sum(r.selection.per_task[t].delta + r.report.per_task[t].delta for t in mc) / (2 * len(mc))
    assert both == pytest.approx(-0.074, abs=0.002)
    assert len(r.boundaries) == 4
    assert m["acquisition"].ppl_ratio["legal"] == pytest.approx(0.79, abs=0.01)
    assert m["acquisition"].general_ppl_ratio == pytest.approx(1.43, abs=0.01)
