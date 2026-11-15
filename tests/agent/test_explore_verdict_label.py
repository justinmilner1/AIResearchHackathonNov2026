"""resolve_one owns the final verdict label, in every mode: a win that costs learning is
`confirmed_with_cost` without --skeptic, and a no_difference prediction that merely fails to
detect an effect is `inconclusive`, not `confirmed`. No network, no GPU."""

from clharness.agent import explore as ex


def _arm(name, retention, legal):
    seeds = range(len(retention))
    return ex.Arm(name=name, key=name, techniques=[], lr=3e-5,
                  retention_by_seed=dict(zip(seeds, retention, strict=True)),
                  legal_ratio_by_seed=dict.fromkeys(seeds, legal),
                  hashes_by_seed={s: f"{name}-{s}" for s in seeds})


def _stored(direction="less_forgetting"):
    return ex.Hypothesis(
        id="h001", created_at="t", author="model", origin="surprise", question="q?", claim="c",
        prediction=ex.Prediction(direction=direction, size_points=5), control="control", outcome="retention",
        test_spec_hashes=["test-0", "test-1"], control_spec_hashes=["control-0", "control-1"],
        decision_rule=ex.rule_for("retention"), status="running")


def test_a_win_that_learns_legal_worse_is_confirmed_with_cost_without_skeptic():
    control = _arm("control", (-6.0, -6.0), legal=0.76)
    # +5 points less forgetting (threshold 2 x 1.0 x 1 = 2), legal ratio 0.95 > 1.05 x 0.76
    costly = _arm("test", (-1.0, -1.0), legal=0.95)
    done = ex.resolve_one(_stored(), [control, costly], sd_points=1.0)
    assert done.acquisition_ok is False
    assert done.status == "confirmed_with_cost"
    assert "5% margin" in done.reason
    # the same win inside the margin stays plain confirmed
    clean = _arm("test", (-1.0, -1.0), legal=0.77)
    assert ex.resolve_one(_stored(), [control, clean], sd_points=1.0).status == "confirmed"
    # a refuted result has no win for the cost to undercut
    worse = _arm("test", (-11.0, -11.0), legal=0.95)
    assert ex.resolve_one(_stored(), [control, worse], sd_points=1.0).status == "refuted"


def test_no_difference_inside_the_detection_threshold_is_inconclusive_not_confirmed():
    control = _arm("control", (-6.0, -6.0), legal=0.76)
    same = _arm("test", (-5.6, -6.2), legal=0.76)  # effect +0.1, threshold 2.0
    done = ex.resolve_one(_stored("no_difference"), [control, same], sd_points=1.0)
    assert done.status == "inconclusive"
    assert "not an equivalence test" in done.reason
    moved = _arm("test", (-1.0, -1.0), legal=0.76)
    assert ex.resolve_one(_stored("no_difference"), [control, moved], sd_points=1.0).status == "refuted"
