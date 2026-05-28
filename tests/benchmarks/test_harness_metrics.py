import pytest
import torch

from benchmarks.harness.cases import BENCHMARK_CASES, MATMUL_CASE_SPEC
from benchmarks.harness.runner import derive_performance_metrics


def test_matmul_metric_model_matches_closed_form_formula():
    params = {"M": 128, "K": 1024, "N": 512}
    metrics = MATMUL_CASE_SPEC.metric_model.account(params, torch.float32)

    expected_flops = 2 * 128 * 1024 * 512
    expected_bytes = (128 * 1024 + 1024 * 512 + 128 * 512) * 4

    assert metrics["flops"] == expected_flops
    assert metrics["algo_bytes"] == expected_bytes

    derived = derive_performance_metrics(
        flops=metrics["flops"],
        algo_bytes=metrics["algo_bytes"],
        median_ms=2.0,
    )
    assert derived["arithmetic_intensity"] == pytest.approx(round(expected_flops / expected_bytes, 6))
    assert derived["achieved_tflops"] == pytest.approx(round(expected_flops / (2.0e-3) / 1e12, 6))


def test_case_registry_contains_only_matmul():
    assert set(BENCHMARK_CASES) == {"matmul"}


def test_derive_performance_metrics_rejects_non_positive_median():
    with pytest.raises(ValueError, match="median_ms must be positive"):
        derive_performance_metrics(flops=1.0, algo_bytes=1.0, median_ms=0.0)
