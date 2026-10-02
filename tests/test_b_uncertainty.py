"""Frame-grouped uncertainty for the B-series disparity gain."""

from experiments.B.B_0_fused_baseline.uncertainty import bootstrap_delta


def test_constant_framewise_gain_has_exact_interval():
    groups = {i: {"baseline_abs_error": 20.0, "fused_abs_error": 10.0,
                  "valid_pixels": 10} for i in range(10)}
    result = bootstrap_delta(groups, draws=100, seed=42)
    assert result["delta_epe_px"] == -1.0
    assert result["ci95_low_px"] == -1.0
    assert result["ci95_high_px"] == -1.0
