"""Shared helpers for GEMM profiler launch targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmarks.harness.baselines import make_case_callable, resolve_kernel_name
from benchmarks.harness.cases import BenchmarkCaseSpec, get_case_spec


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

PROFILER_KERNEL_REGEX: dict[tuple[str, str], str] = {
    ("matmul", "naive_matmul"): "naive_gemm",
    ("matmul", "tiled_matmul"): "squareTiledgemm",
    ("matmul", "coarsened_tiled_matmul"): "ThreadCoarsedTiledgemm",
    ("matmul", "reg2d_tiled_matmul"): "register2DTiledSgemm",
    ("matmul", "reg1d_tiled_matmul"): "register1DTiledSgemm",
    ("matmul", "vec_reg2d_tiled_matmul"): "vectorized_register2DTiledSgemm",
    ("matmul", "warp_tiled_matmul"): "warptile_register2DTiledSgemm",
}


def add_case_target_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--operation", choices=["matmul"], default="matmul")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPE_MAP))
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    return parser


def build_case_context(args: argparse.Namespace) -> dict[str, object]:
    case_spec = get_case_spec(args.operation)
    dtype = DTYPE_MAP[args.dtype]
    params = dict(case_spec.default_params)

    for key in ("M", "K", "N"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value

    kernel = resolve_kernel_name(case_spec.operation, args.kernel)
    inputs = case_spec.input_builder(params, dtype, "cuda")
    fn = make_case_callable(case_spec.operation, kernel, inputs, params)
    return {
        "case_spec": case_spec,
        "kernel": kernel,
        "dtype": dtype,
        "dtype_name": args.dtype,
        "params": params,
        "inputs": inputs,
        "fn": fn,
    }


def validate_case_output(case_spec: BenchmarkCaseSpec, output: torch.Tensor, context: dict[str, object]) -> None:
    case_spec.output_validator(output, context["inputs"], context["params"])


def profiler_kernel_regex(operation: str, kernel: str) -> str | None:
    return PROFILER_KERNEL_REGEX.get((operation, kernel))
