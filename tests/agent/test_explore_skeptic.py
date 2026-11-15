"""--skeptic: the stricter verdict, the replication rule and the skeptic step of the explore loop.
A scripted model and a fake Modal: no network, no GPU."""

import shutil
from pathlib import Path

import pytest
from test_explore import SHA, FakeModal, _cycle, _hyp, _row, _spec

from clharness import spec as a1
from clharness.agent import explore as ex
from clharness.agent.contracts import DecisionLog
from clharness.agent.llm import FakeLLM
from clharness.config import REPO_ROOT

A = _hyp(claim="A helps and learns as well", techniques=[{"name": "replay", "params": {"share": 0.25}}])
B = _hyp(claim="B helps but learns worse", techniques=[{"name": "wavg", "params": {"alpha": 0.5}}])


@pytest.fixture
def have(tmp_path):
    """No mitigation at 1e-4 (five seeds, about -6) and at 3e-5 (seeds 0, 1, 2, about -1.5, legal 0.754)."""
    rows = [_row(_spec(s), v) for s, v in enumerate([-5.8, -5.0, -5.8, -5.4, -6.6])]
    rows += [_row(_spec(s, lr=3e-5), v, legal=0.754) for s, v in enumerate([-1.6, -1.3, -1.5])]
    path = tmp_path / "have.jsonl"
    a1.write_rows(path, rows)
    return path


def score(spec):
    """(selection points, legal ratio). A helps at every seed; B helps at seeds 0 and 1 only and
    learns legal worse; plain training at 2e-5 (the skeptic's check) protects as much as A."""
    names = [t.name for t in spec.techniques]
    if names == ["replay"]:
        return (0.6, 0.76)
    if names == ["wavg"]:
        return (0.6, 0.95) if spec.seed < 2 else (-3.5, 0.95)
    if not names and spec.optim.get("lr") == 2e-5:
        return (0.6, 0.76)
    return (-1.5, 0.754)


def _first_cycle(tmp_path, have, skeptic=True):
    """A and B, both confirmed at two seeds by the fixed rule."""
    llm = FakeLLM().queue("explorer", {"hypotheses": [A, B]})
    out, said = _cycle(tmp_path, have, llm=llm, backend=FakeModal(score), go=True, skeptic=skeptic)
    return {h.claim: h for h in out}, said


def test_a_confirmation_that_learns_worse_is_confirmed_with_cost_with_or_without_skeptic(tmp_path, have):
    plain, _ = _first_cycle(tmp_path / "off", have, skeptic=False)
    assert plain[A["claim"]].status == "confirmed"
    assert plain[B["claim"]].status == "confirmed_with_cost" and plain[B["claim"]].acquisition_ok is False
    doubted, said = _first_cycle(tmp_path / "on", have)
    assert doubted[A["claim"]].status == "confirmed" and doubted[A["claim"]].acquisition_ok is True
    assert doubted[B["claim"]].status == "confirmed_with_cost"
    assert any(line.startswith("h002 CONFIRMED_WITH_COST") for line in said)
    # nothing has standing yet: both are tagged for the proposer, and neither is replicated
    assert not doubted[A["claim"]].replicated
    prompt = ex.build_prompt(ex.read_arms([have, tmp_path / "on" / "explore" / "rows.jsonl"]),
                             list(doubted.values()), doubt=True)
    assert "[confirmed] [unreplicated] A helps" in prompt and "[confirmed with cost] [unreplicated] B helps" in prompt


def test_replication_is_planned_in_order_within_the_budget_then_re_resolves(tmp_path, have):
    _first_cycle(tmp_path, have)
    def quiet():
        return FakeLLM().queue("explorer", {"hypotheses": []}).queue("skeptic", {"checks": []})

    # Dry run, four runs allowed: the plain confirmation comes first and needs three runs (its
    # seeds 2 and 3, and the control's missing seed 3); B's two do not fit and are left out whole.
    out, said = _cycle(tmp_path, have, llm=quiet(), skeptic=True, replication_runs=4)
    plans = [line for line in said if "REPLICATE" in line]
    assert len(plans) == 1 and plans[0].startswith("h001 REPLICATE 3 runs")
    assert "explore-h001-rep-ctl-s3" in plans[0] and "dry run" in said[-1]
    assert not ex.load_hypotheses(tmp_path / "explore" / "hypotheses.jsonl")[0].replication_spec_hashes

    backend = FakeModal(score)
    out, said = _cycle(tmp_path, have, llm=quiet(), backend=backend, go=True, skeptic=True, replication_runs=10)
    assert sorted(s.name for s in backend.spawned) == [
        "explore-h001-rep-ctl-s3", "explore-h001-rep-s2", "explore-h001-rep-s3",
        "explore-h002-rep-s2", "explore-h002-rep-s3"]  # the control's seed 3 is planned once
    assert all(s.notes.startswith(ex.NOTE_PREFIX + " exploratory: replication") and s.parent_spec_hash
               for s in backend.spawned)
    a, b = out[0], out[1]
    assert a.status == "confirmed" and a.replicated and len(a.replication_spec_hashes) == 3
    assert "replication over seeds [0, 1, 2, 3]" in a.reason and a.threshold_points < 1.0  # four seeds: tighter
    assert b.status == "not_replicated" and not b.replicated and "first verdict was confirmed with cost" in b.reason
    assert any(line.startswith("h001 CONFIRMED") and "[replicated]" in line for line in said)
    log = DecisionLog(tmp_path / "explore" / "decisions.jsonl")
    assert log.verify().ok
    doubt = [d for d in log.entries() if d.data.get("replication")][-1]
    assert [p["id"] for p in doubt.data["replication"]] == ["h001", "h002"]
    # a third cycle has nothing left to replicate
    _, said = _cycle(tmp_path, have, llm=quiet(), skeptic=True, replication_runs=10)
    assert not any("REPLICATE" in line for line in said)


def test_the_gate_refuses_a_control_that_is_an_unreplicated_result(tmp_path, have):
    _first_cycle(tmp_path, have)
    on_weak = _hyp(claim="built on A", techniques=[{"name": "replay", "params": {"share": 0.1}}],
                   control="replay@lr3e-05")
    on_plain = _hyp(claim="built on the plain arm", techniques=[{"name": "replay", "params": {"share": 0.1}}])

    def run(**kw):
        llm = FakeLLM().queue("explorer", {"hypotheses": [on_weak, on_plain]}).queue("skeptic", {"checks": []})
        out, _ = _cycle(tmp_path, have, llm=llm, replication_runs=0, **kw)
        return {h.claim: h for h in out}

    doubted = run(skeptic=True)
    assert doubted["built on A"].status == "rejected"
    assert "unreplicated result (h001" in doubted["built on A"].reason
    assert doubted["built on the plain arm"].status == "accepted"
    assert run(skeptic=False)["built on A"].status == "accepted"


def test_a_skeptic_check_runs_ahead_of_the_proposer_and_marks_its_target_challenged(tmp_path, have):
    _first_cycle(tmp_path, have)
    check = _hyp(claim="Skeptic check of h001: plain training at 2e-5 protects as much", techniques=[], lr=2e-5)
    other = _hyp(claim="the proposer's own idea", techniques=[{"name": "replay", "params": {"share": 0.1}}])
    llm = (FakeLLM()
           .queue("skeptic", {"checks": [{"target_id": "h001", "alternative": "it only learned less", "check": check},
                                         {"target_id": "h999", "alternative": "no such result", "check": other}]})
           .queue("explorer", {"hypotheses": [other, _hyp(claim="one too many", lr=5e-5)]}))
    out, _ = _cycle(tmp_path, have, llm=llm, backend=FakeModal(score), go=True, skeptic=True,
                    replication_runs=0, max_hypotheses=2)
    assert [h.claim for h in out[2:]] == [check["claim"], other["claim"]]  # the check took one of the two places
    assert out[2].origin == "skeptic" and out[2].author == "skeptic" and out[2].challenges == "h001"
    assert out[2].status == "confirmed"  # the cheap explanation's prediction held
    assert out[0].challenged and "skeptic check h003" in out[0].challenge_reason and out[0].skeptic_seen
    assert not out[1].challenged and not out[1].skeptic_seen  # B was shown but no check of it was taken
    skeptic_prompt = llm.prompts("skeptic")[0]
    assert "h001" in skeptic_prompt and "legal_ratio_test" in skeptic_prompt and "7777" not in skeptic_prompt
    assert "over-explored" in llm.prompts("explorer")[0]
    prompt = ex.build_prompt(ex.read_arms([have, tmp_path / "explore" / "rows.jsonl"]), out, doubt=True)
    assert "[confirmed] [unreplicated] [challenged] A helps" in prompt
    # a challenged result cannot be a control either, and the skeptic is not asked about it twice
    assert "challenged" in ex.weak_controls(out, ex.read_arms([have, tmp_path / "explore" / "rows.jsonl"]))["replay@lr3e-05"]
    assert {h.id for h in ex.skeptic_targets(out)} == {"h002", "h004"}  # not h001 again, not the check itself
    log = DecisionLog(tmp_path / "explore" / "decisions.jsonl")
    assert log.verify().ok
    doubt = [d for d in log.entries() if d.data.get("skeptic")][-1]
    assert doubt.data["skeptic"]["targets"] == ["h001", "h002"] and len(doubt.data["skeptic"]["input_sha256"]) == 64


def test_without_a_model_or_a_fresh_result_the_skeptic_is_silent(tmp_path, have):
    out, _ = _cycle(tmp_path, have, skeptic=True)  # no model: the fixed list, no skeptic call
    assert out and all(h.challenges is None for h in out)
    assert ex.skeptic_step(ex.read_arms([have]), [], FakeLLM(), 1) == ([], {})


def test_last_nights_records_load_and_resolve_unchanged_with_the_flag_off(tmp_path, have):
    # A frozen copy of results/agent/explore-justin/hypotheses.jsonl as it was before the first
    # --skeptic run (main at 1a8f279). The live file now also holds that run's replicated and
    # challenged records, which this test is not about.
    real = Path(__file__).parent / "fixtures" / "explore_justin_before_skeptic.hypotheses.jsonl"
    out = tmp_path / "explore"
    out.mkdir()
    shutil.copy(real, out / "hypotheses.jsonl")
    before = ex.load_hypotheses(out / "hypotheses.jsonl")
    assert before and all(not h.replicated and not h.challenged and h.challenges is None for h in before)
    after = ex.cycle(out, [have], resolve_only=True, git_sha=SHA, say=lambda s: None)
    # A record still "running" (its runs never landed) only gains a "rows not in yet" note.
    def key(hs):
        return [(h.id, h.status, h.effect_points, None if h.status == "running" else h.reason) for h in hs]
    assert key(after) == key(before)
    assert {h.status for h in after} <= {"accepted", "rejected", "running", "confirmed", "refuted", "inconclusive"}


def test_a_result_with_four_seeds_on_file_gets_its_standing_without_new_runs(tmp_path, have):
    """Rows at seeds 2 and 3 that someone else ran (passed in --have) count: no run is planned."""
    _first_cycle(tmp_path, have)
    extra = tmp_path / "extra.jsonl"
    replay = [{"name": "replay", "params": {"share": 0.25}}]
    a1.write_rows(extra, [_row(_spec(3, lr=3e-5), -1.5, legal=0.754),
                          *(_row(_spec(s, [a1.TechniqueSpec(**t) for t in replay], lr=3e-5), 0.6, legal=0.76)
                            for s in (2, 3))])
    said = []
    out = ex.cycle(tmp_path / "explore", [have, extra], skeptic=True, say=said.append,
                   llm=FakeLLM().queue("explorer", {"hypotheses": []}).queue("skeptic", {"checks": []}))
    assert out[0].replicated and out[0].status == "confirmed"
    assert not any("h001 REPLICATE" in line for line in said)
