from benchmark.profiling.common import profiler_kernel_regex


def test_matmul_profiler_kernel_regex_covers_all_registered_custom_kernels():
    assert profiler_kernel_regex("matmul", "naive_matmul") == "naive_gemm"
    assert profiler_kernel_regex("matmul", "tiled_matmul") == "squareTiledgemm"
    assert profiler_kernel_regex("matmul", "coarsened_tiled_matmul") == "ThreadCoarsedTiledgemm"
    assert profiler_kernel_regex("matmul", "reg2d_tiled_matmul") == "register2DTiledSgemm"
    assert profiler_kernel_regex("matmul", "reg1d_tiled_matmul") == "register1DTiledSgemm"
    assert (
        profiler_kernel_regex("matmul", "vec_reg2d_tiled_matmul")
        == "vectorized_register2DTiledSgemm"
    )
    assert (
        profiler_kernel_regex("matmul", "warp_tiled_matmul")
        == "warptile_register2DTiledSgemm"
    )
