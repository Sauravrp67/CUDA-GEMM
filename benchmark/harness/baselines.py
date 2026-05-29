"""Callable kernel adapters and plotting metadata for GEMM benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from cuda_gemm.backends.cuda.loader import (
    coarsened_tiled_matmul,
    gemm,
    reg1d_tiled_gemm,
    reg2d_tiled_gemm,
    sgemm,
    vec_reg2d_tiled_gemm,
    warpTiled_gemm,
)


CaseInputs = Mapping[str, torch.Tensor]
CaseParams = Mapping[str, Any]
KernelFactory = Callable[[CaseInputs, CaseParams], Callable[[], torch.Tensor]]
MatmulKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class KernelSpec:
    """Single source of truth for one benchmarkable kernel."""

    name: str
    callable: MatmulKernel
    label: str
    line_color: str
    bar_color: str
    marker: str
    line_style: str = "-"
    line_width: float = 2.0
    marker_size: float = 7.0
    role: str = "candidate"
    default: bool = True
    aliases: tuple[str, ...] = ()


def _matmul_factory(kernel: MatmulKernel) -> KernelFactory:
    def _factory(inputs: CaseInputs, _params: CaseParams) -> Callable[[], torch.Tensor]:
        return lambda: kernel(inputs["a"], inputs["b"])

    return _factory


KERNEL_SPECS: dict[str, tuple[KernelSpec, ...]] = {
    "matmul": (
        KernelSpec(
            name="naive_matmul",
            callable=gemm,
            label="Naive GEMM (CUDA)",
            line_color="#ffa657",
            bar_color="#6ACC65",
            marker="s",
        ),
        KernelSpec(
            name="tiled_matmul",
            callable=sgemm,
            label="Tiled GEMM (CUDA)",
            line_color="#ff7b72",
            bar_color="#8C613C",
            marker="^",
        ),
        KernelSpec(
            name="coarsened_tiled_matmul",
            callable=coarsened_tiled_matmul,
            label="Rectangular Tiled GEMM (CUDA)",
            line_color="#d2a8ff",
            bar_color="#579D8C",
            marker="P",
            aliases=("threadcoarsedTiledgemm",),
        ),
        KernelSpec(
            name="reg2d_tiled_matmul",
            callable=reg2d_tiled_gemm,
            label="Register 2D Tiled GEMM (CUDA)",
            line_color="#79c0ff",
            bar_color="#E3B341",
            marker="X",
            aliases=("reg2DTiledsgemm",),
        ),
        KernelSpec(
            name="reg1d_tiled_matmul",
            callable=reg1d_tiled_gemm,
            label="Register 1D Tiled GEMM (CUDA)",
            line_color="#56d364",
            bar_color="#F47067",
            marker="v",
            aliases=("reg1DTiledsgemm",),
        ),
        KernelSpec(
            name="vec_reg2d_tiled_matmul",
            callable=vec_reg2d_tiled_gemm,
            label="Vectorized Register 2D Tiled GEMM (CUDA)",
            line_color="#a5d6ff",
            bar_color="#C678DD",
            marker="h",
            aliases=("vec_reg1DTiledsgemm",),
        ),
        KernelSpec(
            name="warp_tiled_matmul",
            callable=warpTiled_gemm,
            label="Warp-Tiled Register 2D GEMM (CUDA)",
            line_color="#f2cc60",
            bar_color="#4DB6AC",
            marker="8",
            aliases=("warpTiled_gemm",),
        ),
        KernelSpec(
            name="torch_matmul",
            callable=torch.matmul,
            label="torch.matmul",
            line_color="#7ee787",
            bar_color="#B47CC7",
            marker="D",
            line_style="--",
            role="baseline",
        ),
    ),
}


KERNEL_REGISTRY: dict[str, dict[str, KernelSpec]] = {
    operation: {spec.name: spec for spec in specs}
    for operation, specs in KERNEL_SPECS.items()
}


MATMUL_BASELINE_LABELS: dict[str, str] = {
    name: spec.label for name, spec in KERNEL_REGISTRY["matmul"].items()
}


def resolve_kernel_name(operation: str, kernel_name: str) -> str:
    registry = KERNEL_REGISTRY.get(operation, {})
    if kernel_name in registry:
        return kernel_name
    for spec in registry.values():
        if kernel_name in spec.aliases:
            return spec.name
    raise ValueError(f"Unknown {operation} kernel: {kernel_name!r}")


def normalize_kernel_list(
    operation: str,
    kernels: list[str] | tuple[str, ...],
    include_baseline: bool = True,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for kernel_name in kernels:
        canonical = resolve_kernel_name(operation, kernel_name)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)

    if include_baseline:
        baseline = default_baseline_kernel(operation)
        if baseline and baseline not in seen:
            normalized.append(baseline)
    return normalized


def default_kernel_list(operation: str) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in KERNEL_SPECS.get(operation, ())
        if spec.default or spec.role == "baseline"
    )


def default_baseline_kernel(operation: str) -> str:
    registry = KERNEL_REGISTRY.get(operation, {})
    for kernel, spec in registry.items():
        if spec.role == "baseline":
            return kernel
    raise KeyError(f"No baseline kernel configured for operation: {operation!r}")


def make_case_callable(
    operation: str,
    kernel_name: str,
    inputs: CaseInputs,
    params: CaseParams,
) -> Callable[[], torch.Tensor]:
    canonical = resolve_kernel_name(operation, kernel_name)
    spec = KERNEL_REGISTRY[operation][canonical]
    return _matmul_factory(spec.callable)(inputs, params)


def kernel_label(operation: str, kernel_name: str) -> str:
    canonical = resolve_kernel_name(operation, kernel_name)
    return KERNEL_REGISTRY[operation][canonical].label


def kernel_line_style(operation: str, kernel_name: str) -> dict[str, object]:
    canonical = resolve_kernel_name(operation, kernel_name)
    spec = KERNEL_REGISTRY[operation][canonical]
    return {
        "color": spec.line_color,
        "marker": spec.marker,
        "lw": spec.line_width,
        "ms": spec.marker_size,
        "ls": spec.line_style,
    }


def kernel_bar_color(operation: str, kernel_name: str) -> str:
    canonical = resolve_kernel_name(operation, kernel_name)
    return KERNEL_REGISTRY[operation][canonical].bar_color
