from brainscanai.config import load_config
from brainscanai.scale_up.cost import (
    ScaleUpAssumptions,
    estimate,
    sensitivity,
    validation_sample_size,
)


def test_validation_sample_size():
    assert validation_sample_size(0.95, 0.05, 0.5)["n_positives"] == 73


def test_default_scenario_is_feasible_and_naive_is_not():
    a = ScaleUpAssumptions.from_config(load_config())
    r = estimate(a)
    assert r["naive_pilot_extrapolation_eur"] == 800_000
    assert r["total_eur"] <= a.budget_eur and r["feasible"]
    assert abs(r["budget_per_image"] - 0.00125) < 1e-9


def test_infeasible_when_budget_too_small():
    a = ScaleUpAssumptions(budget_eur=50)
    assert not estimate(a)["feasible"]


def test_sensitivity_monotonic():
    s = sensitivity(ScaleUpAssumptions())
    assert s.iloc[0, 0] >= s.iloc[0, -1]   # plus de débit -> moins cher
    assert s.iloc[0, 0] <= s.iloc[-1, 0]   # GPU plus cher -> plus cher
