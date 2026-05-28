#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

#include <primitives/matmul/matmul.h>

#include "primitives/matmul/kernels/naive_gemm.cuh"
#include "primitives/matmul/kernels/block_tiling.cuh"
#include "primitives/matmul/kernels/thread_coarsened.cuh"
#include "primitives/matmul/kernels/register_tiling.cuh"
#include "primitives/matmul/kernels/vectorized.cuh"
#include "primitives/matmul/kernels/warp_tiling.cuh"
#include "primitives/matmul/matmul_dispatch.cuh"
