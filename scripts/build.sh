#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGET_DIR="${ROOT_DIR}/src/cuda_gemm/backends/cuda"

echo "Compiling CUDA GEMM extension..."

cmake --fresh -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DCUDA_GEMM_OUTPUT_DIR="${TARGET_DIR}" \
    -DCUDA_GEMM_OUTPUT_NAME="cuda_gemm_cuda" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo "CUDA GEMM compilation completed."
echo "Extension placement: ${TARGET_DIR}/cuda_gemm_cuda.so"
