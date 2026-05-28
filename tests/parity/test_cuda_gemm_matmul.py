import pytest
import torch

from tests.conftest import (
    ATOL_BF16,
    ATOL_FP32,
    MATMUL_SHAPES,
    RTOL_BF16,
    RTOL_FP16,
    RTOL_FP32,
    make_matmul_inputs,
    matmul_reference,
    transposed_matmul_reference,
)

MATMUL_ATOL_FP16 = 4e-2
TILED_MATMUL_SHAPES = MATMUL_SHAPES
TILED_MATMUL_TILE_TAIL_SHAPES = [
    pytest.param((17, 31, 19), id="M17_K31_N19_tile_tail"),
]
TRANSPOSED_MATMUL_CASES = [
    pytest.param(16, 256, 512, False, True, id="M16_K256_N512_ABt"),
    pytest.param(64, 512, 512, True, False, id="M64_K512_N512_AtB"),
    pytest.param(128, 1024, 512, True, True, id="M128_K1024_N512_AtBt"),
]


def _import_structured_gemm():
    from cuda_gemm.backends.cuda.loader import gemm

    return gemm


def _import_tiled_gemm():
    from cuda_gemm.backends.cuda.loader import sgemm

    return sgemm


def _import_coarsed_tiled_gemm():
    from cuda_gemm.backends.cuda.loader import threadcoarsedTiledgemm

    return threadcoarsedTiledgemm


def _import_reg2d_tiled_gemm():
    from cuda_gemm.backends.cuda.loader import regtiled2DSgemm

    return regtiled2DSgemm


def _import_reg1d_tiled_gemm():
    from cuda_gemm.backends.cuda.loader import regtiled1DSgemm

    return regtiled1DSgemm


def _import_vec_reg2d_tiled_gemm():
    from cuda_gemm.backends.cuda.loader import vec_regtiled2DSgemm

    return vec_regtiled2DSgemm


def _import_warp_tiled_gemm():
    from cuda_gemm.backends.cuda.loader import warpTiled_gemm

    return warpTiled_gemm


def _make_transposed_operands(M: int, K: int, N: int, *, dtype: torch.dtype, trans_a: bool, trans_b: bool):
    a, b = make_matmul_inputs(M, K, N, dtype=dtype)
    a_operand = a.t().contiguous() if trans_a else a
    b_operand = b.t().contiguous() if trans_b else b
    return a_operand, b_operand


class _MatmulKernelParitySuite:
    importer = None
    enabled = True
    supports_fp16 = True
    supports_bf16 = True
    supports_tile_tail = True
    supports_transposed = True

    def _ensure_enabled(self):
        if not self.enabled:
            pytest.skip("kernel is not yet stable across the canonical parity suite")

    def _run(self, a: torch.Tensor, b: torch.Tensor, trans_a: bool = False, trans_b: bool = False) -> torch.Tensor:
        self._ensure_enabled()
        kernel = self.importer()
        return kernel(a, b, trans_a, trans_b)

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_output_shape(self, shape):
        self._ensure_enabled()
        a, b = make_matmul_inputs(*shape)
        out = self._run(a, b)
        assert out.shape == (shape[0], shape[2])

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_output_is_finite(self, shape):
        self._ensure_enabled()
        a, b = make_matmul_inputs(*shape)
        out = self._run(a, b)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_parity_torch_fp32(self, shape):
        self._ensure_enabled()
        a, b = make_matmul_inputs(*shape, dtype=torch.float32)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_parity_torch_fp16(self, shape):
        self._ensure_enabled()
        if not self.supports_fp16:
            pytest.skip("kernel currently exposed as float32-only")
        a, b = make_matmul_inputs(*shape, dtype=torch.float16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES[:-1])
    def test_parity_torch_bf16(self, shape):
        self._ensure_enabled()
        if not self.supports_bf16:
            pytest.skip("kernel currently exposed as float32-only")
        a, b = make_matmul_inputs(*shape, dtype=torch.bfloat16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    @pytest.mark.parametrize("shape", TILED_MATMUL_TILE_TAIL_SHAPES)
    def test_parity_torch_fp32_tile_tail(self, shape):
        self._ensure_enabled()
        if not self.supports_tile_tail:
            pytest.skip("kernel currently assumes aligned/vector-friendly tail shapes")
        a, b = make_matmul_inputs(*shape, dtype=torch.float32)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", TILED_MATMUL_TILE_TAIL_SHAPES)
    def test_parity_torch_fp16_tile_tail(self, shape):
        self._ensure_enabled()
        if not self.supports_tile_tail:
            pytest.skip("kernel currently assumes aligned/vector-friendly tail shapes")
        if not self.supports_fp16:
            pytest.skip("kernel currently exposed as float32-only")
        a, b = make_matmul_inputs(*shape, dtype=torch.float16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES)
    def test_parity_transposed_fp32(self, M, K, N, trans_a, trans_b):
        self._ensure_enabled()
        if not self.supports_transposed:
            pytest.skip("kernel currently wired for non-transposed operands only")
        a, b = _make_transposed_operands(M, K, N, dtype=torch.float32, trans_a=trans_a, trans_b=trans_b)
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES)
    def test_parity_transposed_fp16(self, M, K, N, trans_a, trans_b):
        self._ensure_enabled()
        if not self.supports_transposed:
            pytest.skip("kernel currently wired for non-transposed operands only")
        if not self.supports_fp16:
            pytest.skip("kernel currently exposed as float32-only")
        a, b = _make_transposed_operands(M, K, N, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b)
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES[:-1])
    def test_parity_transposed_bf16(self, M, K, N, trans_a, trans_b):
        self._ensure_enabled()
        if not self.supports_transposed:
            pytest.skip("kernel currently wired for non-transposed operands only")
        if not self.supports_bf16:
            pytest.skip("kernel currently exposed as float32-only")
        a, b = _make_transposed_operands(M, K, N, dtype=torch.bfloat16, trans_a=trans_a, trans_b=trans_b)
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    def test_shape_mismatch_raises(self):
        self._ensure_enabled()
        a = torch.randn(16, 32, device="cuda")
        b = torch.randn(31, 64, device="cuda")
        with pytest.raises(RuntimeError, match="Shape mismatch for GEMM"):
            self.importer()(a, b)


@pytest.mark.requires_cuda
class TestStructuredNaiveMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_structured_gemm)


@pytest.mark.requires_cuda
class TestStructuredTiledMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_tiled_gemm)


@pytest.mark.requires_cuda
class TestStructuredRectTiledMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_coarsed_tiled_gemm)


@pytest.mark.requires_cuda
class TestStructuredReg2DTiledMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_reg2d_tiled_gemm)


@pytest.mark.requires_cuda
class TestStructuredReg1DTiledMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_reg1d_tiled_gemm)


@pytest.mark.requires_cuda
class TestStructuredVecReg2DTiledMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_vec_reg2d_tiled_gemm)
    supports_fp16 = False
    supports_bf16 = False
    supports_tile_tail = False
    supports_transposed = False


@pytest.mark.requires_cuda
class TestStructuredWarpTiledMatmul(_MatmulKernelParitySuite):
    importer = staticmethod(_import_warp_tiled_gemm)
    enabled = False
    supports_fp16 = False
    supports_bf16 = False
