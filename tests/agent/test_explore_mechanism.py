"""The fragile-core manifest, the per-item profile, and explore hypotheses resolved on it.
No network, no GPU: per-item files are written by the tests, and the fetch step is a fake."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from test_explore import SHA, FakeModal, _hyp, _row, _spec

from clharness import cli
from clharness import spec as a1
from clharness.agent import explore as ex
from clharness.agent.llm import FakeLLM
from clharness.analysis import item_profile as ip
from clharness.config import REPO_ROOT
from clharness.executor.rows import half_indices

CORE = json.loads((REPO_ROOT / "manifests" / "eval-protected-v1-core.json").read_text())
TASKS = list(CORE["tasks"])


def _write_items(root: Path, spec: a1.RunSpec, core_lost: float, other_lost: int = 0, gained: int = 0,
                 report_value: float | None = None) -> None:
    """A run's first and last checkpoint. The base model knows every selection-half question
    except the first 10 non-core ones per task; the run loses `core_lost` of each task's core,
    `other_lost` other known questions per task and gains `gained` unknown ones per task."""
    for task, info in CORE["tasks"].items():
        selection, report = half_indices(task, info["n_items"])
        core = info["core_positions"]
        rest = [p for p in selection if p not in set(core)]
        unknown, known = rest[:10], rest[10:]
        base = np.ones(info["n_items"])
        base[unknown] = 0.0
        final = base.copy()
        final[core[:round(core_lost * len(core))]] = 0.0
        final[known[:other_lost]] = 0.0
        final[unknown[:gained]] = 1.0
        if report_value is not None:  # whatever the report half holds must not matter
            base[report], final[report] = 1.0 - report_value, report_value
        for name, values in (("ckpt_0000", base), ("ckpt_0002", final)):
            folder = root / "results" / "items" / spec.spec_hash()[:16] / "checkpoints" / name
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"items_{task}.npy", values)


# ---- the frozen manifest -----------------------------------------------------------------


def test_core_manifest_has_the_counts_found_and_only_selection_half_positions():
    assert {t: v["n_core"] for t, v in CORE["tasks"].items()} == {
        "arc_easy": 23, "hellaswag": 18, "piqa": 5, "pubmedqa_pqal": 12}
    assert CORE["n_core"] == 58 and CORE["n_noncore_base_correct"] == 625
    assert CORE["rule"]["min_seeds"] == 4 and len(CORE["source_runs"]) == 5
    assert CORE["rows_file"] == "results/rows.phase0.remeasured.jsonl" and CORE["derived_on"]
    for task, info in CORE["tasks"].items():
        selection, report = half_indices(task, info["n_items"])
        positions = info["core_positions"]
        assert len(positions) == len(set(positions)) == info["n_core"]
        assert set(positions) <= set(selection.tolist())
        assert not set(positions) & set(report.tolist())
    assert ip.set_sizes() == {"core_forgotten": 58, "noncore_forgotten": 625}


def test_the_generator_script_reproduces_the_committed_manifest():
    script = REPO_ROOT / "scripts" / "make_core_items.py"
    module_spec = importlib.util.spec_from_file_location("make_core_items", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.build() == CORE


# ---- the per-item profile -----------------------------------------------------------------


def test_item_profile_counts_core_other_and_gained_on_the_selection_half(tmp_path):
    spec = _spec(0, lr=3e-5)
    row = _row(spec, -1.5)
    assert ip.item_profile(row, tmp_path) is None  # no files on this machine
    _write_items(tmp_path, spec, core_lost=1.0, other_lost=5, gained=3)
    got = ip.item_profile(row, tmp_path)
    assert got["n_core"] == 58 and got["n_base_wrong"] == 40
    assert got["core_forgotten"] == 1.0
    assert got["noncore_forgotten"] == pytest.approx(20 / got["n_noncore"])
    assert got["gained"] == pytest.approx(12 / 40)
    # the report half is never read: filling it with other answers changes nothing
    for value in (0.0, 1.0):
        _write_items(tmp_path, spec, core_lost=1.0, other_lost=5, gained=3, report_value=value)
        assert ip.item_profile(row, tmp_path) == got
    # a row that still points into the cloud volume falls back to the local mirror
    cloud = row.model_copy(update={"items_path": "/__modal/volumes/vo-x/runs/abc/checkpoints"})
    assert ip.item_profile(cloud, tmp_path) == got
    (tmp_path / "results" / "items" / spec.spec_hash()[:16] / "checkpoints" / "ckpt_0002" / "items_piqa.npy").unlink()
    assert ip.item_profile(row, tmp_path) is None


def test_item_profile_command_prints_one_line_per_arm_and_marks_the_defining_arm(tmp_path, monkeypatch, capsys):
    rows = []
    for seed in (0, 1):
        for spec, lost in ((_spec(seed), 1.0), (_spec(seed, lr=3e-5), 0.5)):
            _write_items(tmp_path, spec, core_lost=lost)
            rows.append(_row(spec, -2.0))
    rows.append(_row(_spec(0, lr=5e-5), -2.0))  # no item files
    path = tmp_path / "rows.jsonl"
    a1.write_rows(path, rows)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli.cmd_item_profile(Namespace(files=[str(path)])) == 0
    out = capsys.readouterr().out
    assert "no-mitigation* |" in out and "inflated" in out and "by construction" in out
    low = next(line for line in out.splitlines() if line.startswith("no-mitigation@lr3e-05"))
    assert "[0, 1]" in low and "0.0%" in low
    share = sum(round(0.5 * v["n_core"]) for v in CORE["tasks"].values()) / 58
    assert f"{share:.1%}" in low
    assert "no item files here" in next(x for x in out.splitlines() if x.startswith("no-mitigation@lr5e-05"))


# ---- explore: arms, gate, resolve ---------------------------------------------------------


def _arm(name, core, other, techniques=(), lr=3e-5, retention=(-1.5, -1.4)):
    seeds = range(len(retention))
    return ex.Arm(name=name, key=name, techniques=list(techniques), lr=lr,
                  retention_by_seed=dict(zip(seeds, retention, strict=True)),
                  legal_ratio_by_seed=dict.fromkeys(seeds, 0.76),
                  hashes_by_seed={s: f"{name}-{s}" for s in seeds},
                  core_forgotten_by_seed=dict(zip(seeds, core, strict=False)),
                  noncore_forgotten_by_seed=dict(zip(seeds, other, strict=False)))


def _stored(outcome, direction="less_forgetting"):
    return ex.Hypothesis(
        id="h001", created_at="t", author="model", origin="surprise", question="q?", claim="c",
        prediction=ex.Prediction(direction=direction, size_points=10), control="control", outcome=outcome,
        test_spec_hashes=["test-0", "test-1"], control_spec_hashes=["control-0", "control-1"],
        decision_rule=ex.rule_for(outcome), status="running")


def test_resolve_uses_the_chosen_outcome_with_binomial_item_noise():
    control = _arm("control", core=(50.0, 46.0), other=(4.0, 4.0))
    freeze = [a1.TechniqueSpec(name="freeze", params={})]
    # core: control minus test = +24 points; pooled share 0.36 over 58 questions and 2 seeds
    test = _arm("test", core=(24.0, 24.0), other=(4.0, 4.4), techniques=freeze)
    done = ex.resolve_one(_stored("core_forgotten"), [control, test], sd_points=None)
    expected = 2 * (0.36 * 0.64 / 58 * 2 / 2) ** 0.5 * 100
    assert done.status == "confirmed" and done.effect_points == pytest.approx(24.0)
    assert done.threshold_points == pytest.approx(expected, abs=0.001)
    assert "core_forgotten" in done.reason and "58" in done.reason
    assert "58 questions" in done.decision_rule and "binomial" in done.decision_rule
    # the same runs read on the other known questions: inside the noise of 625 questions
    other = ex.resolve_one(_stored("noncore_forgotten"), [control, test], sd_points=None)
    assert other.status == "inconclusive" and other.effect_points == pytest.approx(-0.2)
    assert other.threshold_points == pytest.approx(2 * (0.041 * 0.959 / 625) ** 0.5 * 100, abs=0.001)
    # inside the noise is a failed detection, not an equivalence test: inconclusive, not confirmed
    assert ex.resolve_one(_stored("noncore_forgotten", "no_difference"), [control, test], None).status == "inconclusive"
    # predicted less forgetting of the core, measured clearly more: refuted
    worse = _arm("test", core=(80.0, 76.0), other=(4.0, 4.0), techniques=freeze)
    assert ex.resolve_one(_stored("core_forgotten"), [control, worse], sd_points=None).status == "refuted"
    assert ex.resolve_one(_stored("core_forgotten", "more_forgetting"), [control, worse], None).status == "confirmed"


def test_resolve_is_inconclusive_when_item_files_are_missing_and_retention_is_unchanged():
    control = _arm("control", core=(50.0, 46.0), other=(4.0, 4.0))
    bare = _arm("test", core=(), other=(), techniques=[a1.TechniqueSpec(name="freeze", params={})],
                retention=(-0.2, -0.1))
    done = ex.resolve_one(_stored("core_forgotten"), [control, bare], sd_points=0.5)
    assert done.status == "inconclusive" and "per-item files are missing" in done.reason
    assert done.effect_points is None
    one = bare.model_copy(update={"core_forgotten_by_seed": {0: 20.0}})
    assert ex.resolve_one(_stored("core_forgotten"), [control, one], None).status == "inconclusive"
    # the retention outcome is the rule it always was: test minus control against the seed noise
    kept = ex.resolve_one(_stored("retention"), [control, bare], sd_points=0.5)
    assert kept.status == "confirmed" and kept.effect_points == pytest.approx(1.3)
    assert kept.threshold_points == pytest.approx(1.0) and kept.decision_rule == ex.RULE
    # test rows that have not landed leave it running, whatever the outcome
    assert ex.resolve_one(_stored("core_forgotten"), [control], None).status == "running"


@pytest.fixture
def bench(tmp_path):
    """A repo root with rows and per-item files: no mitigation at 1e-4 (the arm that defines the
    core), at 3e-5 (loses half the core) and at 5e-5 (rows only, no item files)."""
    rows = []
    for seed in range(5):
        _write_items(tmp_path, _spec(seed), core_lost=1.0, other_lost=14)
        rows.append(_row(_spec(seed), [-5.8, -5.0, -5.8, -5.4, -6.6][seed]))
    for seed in range(3):
        _write_items(tmp_path, _spec(seed, lr=3e-5), core_lost=0.5, other_lost=5)
        rows.append(_row(_spec(seed, lr=3e-5), -1.5, legal=0.754))
        rows.append(_row(_spec(seed, lr=5e-5), -2.5, legal=0.75))
    path = tmp_path / "have.jsonl"
    a1.write_rows(path, rows)
    return path


def test_arms_carry_core_and_other_forgetting_and_the_table_shows_them(tmp_path, bench):
    arms = {a.name: a for a in ex.read_arms([bench], tmp_path)}
    low, plain, bare = arms["no-mitigation@lr3e-05"], arms["no-mitigation"], arms["no-mitigation@lr5e-05"]
    half = sum(round(0.5 * v["n_core"]) for v in CORE["tasks"].values()) / 58 * 100
    assert low.core_forgotten_by_seed == pytest.approx(dict.fromkeys(range(3), half))
    assert low.noncore_forgotten_by_seed[0] == pytest.approx(20 / 902 * 100)  # 1000 - 58 core - 40 unknown
    assert plain.defines_core and not low.defines_core
    assert bare.core_forgotten_by_seed == {} and bare.noncore_forgotten_by_seed == {}
    table = ex.arm_table(list(arms.values()))
    assert "core forgotten (%)" in table and "other known forgotten (%)" in table
    assert f"| {half:.1f} | 2.2 |" in table and "| 100.0* |" in table and "no item files" in table
    assert "by construction" in table


def test_the_gate_refuses_a_forgetting_outcome_without_a_usable_control(tmp_path, bench):
    arms = ex.read_arms([bench], tmp_path)

    def gate(**kw):
        return ex.gate_one(ex.ProposedHypothesis.model_validate(_hyp(**kw)), "h001", arms, set(), None)

    specs, why = gate(outcome="core_forgotten", control="no-mitigation", lr=None)
    assert not specs and "define the fragile core" in why
    specs, why = gate(outcome="noncore_forgotten", control="no-mitigation@lr5e-05", lr=5e-5)
    assert not specs and "per-item files" in why
    assert gate(outcome="retention", control="no-mitigation", lr=None)[0]
    assert gate(outcome="core_forgotten")[0]
    with pytest.raises(ValueError):
        ex.ProposedHypothesis.model_validate(_hyp(outcome="report_half"))


def test_a_cycle_fetches_item_files_then_resolves_on_the_core(tmp_path, bench):
    llm = FakeLLM().queue("explorer", {"hypotheses": [
        _hyp(claim="directional interference: replay protects the core", outcome="core_forgotten",
             prediction={"direction": "less_forgetting", "size_points": 20}),
        _hyp(claim="displacement: freezing changes nothing on the core", outcome="core_forgotten", lr=3e-5,
             techniques=[{"name": "freeze", "params": {"bottom_layers": 13}}],
             prediction={"direction": "less_forgetting", "size_points": 20})]})
    backend = FakeModal(lambda spec: (-1.4, 0.76))
    fetched, said = [], []

    def fake_fetch(files):
        """Stands in for `modal volume get`: the replay runs' files arrive, the freeze runs' do not."""
        fetched.append([Path(f).name for f in files])
        for row in a1.read_rows(files[0]):
            if "replay" in row.spec.technique_names():
                _write_items(tmp_path, row.spec, core_lost=0.1, other_lost=5)
        return 2

    out = ex.cycle(tmp_path / "explore", [bench], llm=llm, backend=backend, git_sha=SHA, go=True,
                   sleep=lambda s: None, say=said.append, fetch_items=fake_fetch, repo_root=tmp_path)
    assert fetched == [["rows.jsonl"]]
    replay, freeze = out
    assert replay.outcome == "core_forgotten" and replay.status == "confirmed"
    assert replay.effect_points > 30 and 0 < replay.threshold_points < 20
    assert "58 questions" in replay.decision_rule
    assert freeze.status == "inconclusive" and "per-item files are missing" in freeze.reason
    assert any("2 runs could not be fetched" in line for line in said)
    prompt = "\n".join(llm.prompts("explorer"))
    assert "core forgotten (%)" in prompt and "WHAT BLOCKS" in prompt and "readout drift" in prompt
    assert '"outcome"' in prompt and "7777" not in prompt
    # the stored record and the decision log both say which outcome and threshold were used
    stored = ex.load_hypotheses(tmp_path / "explore" / "hypotheses.jsonl")
    assert [h.outcome for h in stored] == ["core_forgotten", "core_forgotten"]
    log = (tmp_path / "explore" / "decisions.jsonl").read_text()
    assert "\"outcome\":\"core_forgotten\"" in log and "\"threshold_points\"" in log


def test_without_a_fetch_step_nothing_is_downloaded_and_a_failing_one_does_not_stop_the_cycle(tmp_path, bench):
    def run(name, **kw):
        llm = FakeLLM().queue("explorer", {"hypotheses": [_hyp(outcome="core_forgotten")]})
        said = []
        out = ex.cycle(tmp_path / name, [bench], llm=llm, backend=FakeModal(lambda spec: (-1.4, 0.76)),
                       git_sha=SHA, go=True, sleep=lambda s: None, say=said.append, repo_root=tmp_path, **kw)
        return out[0], said

    plain, _ = run("a")  # the default: no fetch, so the item outcome cannot be read
    assert plain.status == "inconclusive" and "per-item files are missing" in plain.reason

    def broken(files):
        raise OSError("modal is not installed")

    failed, said = run("b", fetch_items=broken)
    assert failed.status == "inconclusive"
    assert any("fetch failed (OSError" in line for line in said)
    # a dry run never fetches
    ex.cycle(tmp_path / "c", [bench], llm=FakeLLM().queue("explorer", {"hypotheses": [_hyp()]}),
             say=lambda s: None, fetch_items=broken, repo_root=tmp_path)
