#pragma once

template <typename scalar_t>
__global__ void naive_gemm(
    const scalar_t* __restrict__ A, const int M, const int K, const int lda, bool transA,
    const scalar_t* __restrict__ B, const int N, const int ldb, bool transB,
    scalar_t* __restrict__ C, int ldc,
    const float alpha, float beta
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    float p_value = 0.0f;
    if (row < M && column < N) {
        for (int k = 0; k < K; k++) {
            float a_value = transA ? (float)A[k * lda + row] : (float)A[row * lda + k];
            float b_value = transB ? (float)B[column * ldb + k] : (float)B[k * ldb + column];
            p_value += a_value * b_value;
        }
        const float c_value = beta == 0.0f ? 0.0f : static_cast<float>(C[row * ldc + column]);
        C[row * ldc + column] = scalar_t(alpha * p_value + beta * c_value);
    }
}
