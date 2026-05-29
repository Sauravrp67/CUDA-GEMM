"""Case definitions and analytical metric models for GEMM benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from benchmark.harness.baselines import default_baseline_kernel, default_kernel_list


CaseParams = Mapping[str, Any]
InputBuilder = Callable[[CaseParams, torch.dtype, str], dict[str, torch.Tensor]]
OutputValidator = Callable[[torch.Tensor, dict[str, torch.Tensor], CaseParams], None]
ContextFormatter = Callable[[str, CaseParams, str], dict[str, str]]
MetricAccountFn = Callable[[CaseParams, torch.dtype], dict[str, float]]


@dataclass(frozen=True)
class SweepAxisSpec:
    name: str
    parameter: str
    x_label: str
    default_values: tuple[int, ...]
    log_x: bool = True
    tick_format: str = "plain"


@dataclass(frozen=True)
class MetricModel:
    name: str
    account_fn: MetricAccountFn

    def account(self, params: CaseParams, dtype: torch.dtype) -> dict[str, float]:
        metrics = dict(self.account_fn(params, dtype))
        if "flops" not in metrics or "algo_bytes" not in metrics:
            raise ValueError(f"Metric model {self.name!r} must provide flops and algo_bytes.")
        return metrics


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    operation: str
    operation_label: str
    suite_label: str
    legacy_report_prefix: str
    default_kernels: tuple[str, ...]
    baseline_kernel: str
    supported_sweep_axes: dict[str, SweepAxisSpec]
    default_params: dict[str, Any]
    input_builder: InputBuilder
    output_validator: OutputValidator
    metric_model: MetricModel
    context_formatter: ContextFormatter

    def get_axis(self, sweep_axis: str) -> SweepAxisSpec:
        if sweep_axis not in self.supported_sweep_axes:
            supported = ", ".join(sorted(self.supported_sweep_axes))
            raise ValueError(
                f"{self.operation} does not support sweep axis {sweep_axis!r}. Supported axes: {supported}"
            )
        return self.supported_sweep_axes[sweep_axis]

    def default_axis_values(self, sweep_axis: str) -> tuple[int, ...]:
        return self.get_axis(sweep_axis).default_values

    def format_context(self, sweep_axis: str, params: CaseParams, dtype_str: str) -> dict[str, str]:
        return self.context_formatter(sweep_axis, params, dtype_str)


def _dtype_itemsize(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    return torch.iinfo(dtype).bits // 8


def _matmul_metric_account(params: CaseParams, dtype: torch.dtype) -> dict[str, float]:
    M = int(params["M"])
    K = int(params["K"])
    N = int(params["N"])
    itemsize = _dtype_itemsize(dtype)
    flops = 2 * M * K * N
    algo_bytes = (M * K + K * N + M * N) * itemsize
    return {"flops": float(flops), "algo_bytes": float(algo_bytes)}


def _build_matmul_inputs(params: CaseParams, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    seed = int(params.get("seed", 42))
    torch.manual_seed(seed)
    M = int(params["M"])
    K = int(params["K"])
    N = int(params["N"])
    return {
        "a": torch.randn(M, K, dtype=dtype, device=device),
        "b": torch.randn(K, N, dtype=dtype, device=device),
    }


def _validate_matmul_output(output: torch.Tensor, _inputs: dict[str, torch.Tensor], params: CaseParams) -> None:
    expected_shape = (int(params["M"]), int(params["N"]))
    if tuple(output.shape) != expected_shape:
        raise AssertionError(
            f"Matmul output shape mismatch: got {tuple(output.shape)}, expected {expected_shape}"
        )


def _matmul_context(sweep_axis: str, params: CaseParams, dtype_str: str) -> dict[str, str]:
    M = int(params["M"])
    K = int(params["K"])
    N = int(params["N"])
    if sweep_axis != "M":
        raise ValueError(f"Unsupported matmul sweep axis: {sweep_axis!r}")
    return {
        "banner_desc": f"M sweep  K={K} N={N}",
        "fixed_desc": f"K={K}, N={N}",
        "compare_config": f"K={K}, N={N}",
        "footnote": f"K={K}  N={N}  dtype={dtype_str}",
        "dim_tag": f"K{K}_N{N}",
        "causal_tag": "none",
    }


MATMUL_CASE_SPEC = BenchmarkCaseSpec(
    operation="matmul",
    operation_label="Matmul",
    suite_label="CUDA GEMM Suite",
    legacy_report_prefix="matmul_benchmark",
    default_kernels=default_kernel_list("matmul"),
    baseline_kernel=default_baseline_kernel("matmul"),
    supported_sweep_axes={
        "M": SweepAxisSpec(
            name="M",
            parameter="M",
            x_label="Rows M",
            default_values=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
        ),
    },
    default_params={"M": 64, "K": 8192, "N": 8192, "seed": 42},
    input_builder=_build_matmul_inputs,
    output_validator=_validate_matmul_output,
    metric_model=MetricModel(name="matmul", account_fn=_matmul_metric_account),
    context_formatter=_matmul_context,
)


BENCHMARK_CASES: dict[str, BenchmarkCaseSpec] = {
    "matmul": MATMUL_CASE_SPEC,
}


def get_case_spec(operation: str) -> BenchmarkCaseSpec:
    if operation not in BENCHMARK_CASES:
        raise KeyError(f"Unknown benchmark case: {operation!r}")
    return BENCHMARK_CASES[operation]
