import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

ATOL_FP32 = 1e-4
RTOL_FP32 = 1e-3
ATOL_FP16 = 5e-3
RTOL_FP16 = 5e-3
ATOL_BF16 = 5e-2
RTOL_BF16 = 5e-2


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: skip if no CUDA device is available")


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


MATMUL_SHAPES = [
    pytest.param((16, 256, 512), id="M16_K256_N512"),
    pytest.param((64, 512, 512), id="M64_K512_N512"),
    pytest.param((128, 1024, 512), id="M128_K1024_N512"),
    pytest.param((256, 1024, 1024), id="M256_K1024_N1024"),
]


def make_matmul_inputs(
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn(M, K, generator=generator).to(dtype=dtype, device=device)
    b = torch.randn(K, N, generator=generator).to(dtype=dtype, device=device)
    return a, b


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def transposed_matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    lhs = a.transpose(-2, -1) if trans_a else a
    rhs = b.transpose(-2, -1) if trans_b else b
    return torch.matmul(lhs, rhs)
