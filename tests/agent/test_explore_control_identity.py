"""A stored hypothesis finds its control by the spec hashes it stored, not by the arm name.

read_arms makes a name up per call (technique base name plus primes in first-seen order), so the
same spec can be `no-mitigation@lr3e-05` in one table and `no-mitigation@lr3e-05'` in another,
depending only on which rows files were read and in what order. No network, no GPU."""

import pytest
from test_explore import STREAM, _row

from clharness import spec as a1
from clharness.agent import explore as ex
from clharness.agent import explore_summary as summary

SD = 0.5  # a fixed noise floor: the verdict depends only on which control arm is read
REPLAY = [a1.TechniqueSpec(name="replay", params={"share": 0.25})]


def _spec(seed, techniques=(), **optim):
    return a1.RunSpec(name=f"x-s{seed}", seed=seed, bench="fast", stream=STREAM, optim=optim,
                      techniques=list(techniques))


@pytest.fixture
def files(tmp_path):
    """Two different plain arms that share the base name `no-mitigation@lr3e-05`, one per file,
    and a replay test arm. The stored control (file a) is at -1.5; the other (file b) at -5.0."""
    paths = {k: tmp_path / f"{k}.jsonl" for k in ("a", "b", "test")}
    a1.write_rows(paths["a"], [_row(_spec(s, lr=3e-5), -1.5) for s in (0, 1)])
    a1.write_rows(paths["b"], [_row(_spec(s, lr=3e-5, weight_decay=0.0), -5.0) for s in (0, 1)])
    a1.write_rows(paths["test"], [_row(_spec(s, REPLAY, lr=3e-5), 0.6) for s in (0, 1)])
    return paths


def _stored(files):
    """A hypothesis as the gate writes it: control name and hashes from the table of that cycle."""
    arms = ex.read_arms([files["a"], files["b"], files["test"]])
    control = next(a for a in arms if a.hashes_by_seed[0] == _spec(0, lr=3e-5).spec_hash())
    test = next(a for a in arms if a.techniques)
    assert control.name == "no-mitigation@lr3e-05"  # file a came first, so it got the plain name
    h = ex.Hypothesis(id="h001", created_at="t", author="test", origin="surprise", question="q?",
                      claim="replay forgets less", prediction={"direction": "less_forgetting", "size_points": 0.5},
                      control=control.name, test_spec_hashes=[test.hashes_by_seed[s] for s in (0, 1)],
                      control_spec_hashes=[control.hashes_by_seed[s] for s in (0, 1)], status="running")
    return ex.resolve_one(h, arms, SD), control


def test_a_reordered_input_set_resolves_against_the_stored_control_not_its_old_name(files):
    first, control = _stored(files)
    assert first.status == "confirmed" and first.effect_points == pytest.approx(2.1)
    reordered = ex.read_arms([files["b"], files["a"], files["test"]])
    # the same spec now carries a prime, and its old name belongs to the other plain arm
    assert next(a.name for a in reordered if a.key == control.key) == "no-mitigation@lr3e-05'"
    again = ex.resolve_one(first.model_copy(update={"status": "running"}), reordered, SD)
    assert again.status == "confirmed" and again.effect_points == pytest.approx(2.1)
    assert ex._control_arm(first, reordered).key == control.key
    assert summary.record(first, reordered)["control"].key == control.key
    # the skeptic is shown the control's name in the table it reads, not the stale stored one
    assert '"control": "no-mitigation@lr3e-05\'"' in ex.skeptic_prompt(reordered, [first])


def test_a_stored_control_whose_rows_are_absent_is_missing_not_another_arm(files):
    first, _ = _stored(files)
    without = ex.read_arms([files["b"], files["test"]])  # the other plain arm now has the old name
    again = ex.resolve_one(first.model_copy(update={"status": "running"}), without, SD)
    assert again.reason == "control or noise floor missing" and again.effect_points == first.effect_points
    assert ex._control_arm(first, without) is None
    assert summary.record(first, without)["control"] is None


def test_a_record_without_control_hashes_still_resolves_by_name(files):
    """Records written before the gate stored control hashes (and rejected ones) keep the old lookup."""
    first, control = _stored(files)
    legacy = first.model_copy(update={"control_spec_hashes": [], "status": "running"})
    arms = ex.read_arms([files["a"], files["b"], files["test"]])
    assert ex._control_arm(legacy, arms).key == control.key
    assert ex.resolve_one(legacy, arms, SD).effect_points == pytest.approx(2.1)
