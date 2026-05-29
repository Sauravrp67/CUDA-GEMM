import pytest
import torch

from benchmark.harness.baselines import (
    KERNEL_REGISTRY,
    KERNEL_SPECS,
    default_baseline_kernel,
    default_kernel_list,
    resolve_kernel_name,
)
from benchmark.harness.cases import BENCHMARK_CASES, MATMUL_CASE_SPEC
from benchmark.harness.runner import derive_performance_metrics


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


def test_kernel_specs_are_single_source_for_matmul_defaults():
    specs = KERNEL_SPECS["matmul"]
    expected_defaults = tuple(
        spec.name for spec in specs if spec.default or spec.role == "baseline"
    )

    assert default_kernel_list("matmul") == expected_defaults
    assert MATMUL_CASE_SPEC.default_kernels == expected_defaults
    assert default_baseline_kernel("matmul") == "torch_matmul"
    assert MATMUL_CASE_SPEC.baseline_kernel == "torch_matmul"


def test_kernel_specs_expose_callables_and_valid_aliases():
    registry = KERNEL_REGISTRY["matmul"]

    for spec in KERNEL_SPECS["matmul"]:
        assert spec.name in registry
        assert callable(spec.callable)
        assert resolve_kernel_name("matmul", spec.name) == spec.name
        for alias in spec.aliases:
            assert resolve_kernel_name("matmul", alias) == spec.name


def test_derive_performance_metrics_rejects_non_positive_median():
    with pytest.raises(ValueError, match="median_ms must be positive"):
        derive_performance_metrics(flops=1.0, algo_bytes=1.0, median_ms=0.0)
