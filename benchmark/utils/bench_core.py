"""Lightweight analytical helpers retained for GEMM roofline utilities."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.utils.hardware_constants import RTX_4050_LAPTOP

HW = RTX_4050_LAPTOP


@dataclass(frozen=True)
class KernelMeta:
    label: str


KERNEL_REGISTRY: dict[str, KernelMeta] = {
    "naive_matmul": KernelMeta(label="Naive GEMM (CUDA)"),
    "tiled_matmul": KernelMeta(label="Tiled GEMM (CUDA)"),
    "torch_matmul": KernelMeta(label="torch.matmul"),
}


def matmul_flops(M: int, K: int, N: int) -> int:
    return 2 * M * K * N


def matmul_bytes(M: int, K: int, N: int, itemsize: int) -> int:
    return (M * K + K * N + M * N) * itemsize
