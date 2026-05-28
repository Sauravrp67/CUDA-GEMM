#pragma once

template <typename scalar_t, const int TILE_WIDTH>
__global__ void squareTiledgemm(
    const scalar_t* A, const int M, const int K, const int lda, bool transA,
    const scalar_t* B, const int N, const int ldb, bool transB,
    scalar_t* C, const int ldc,
    const float alpha, const float beta
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ scalar_t Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t Nds[TILE_WIDTH][TILE_WIDTH];

    int row = by * TILE_WIDTH + ty;
    int column = bx * TILE_WIDTH + tx;
    float pValue = 0.0f;

    for (size_t ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (row < M && (ph * TILE_WIDTH + tx) < K) {
            Mds[ty][tx] = transA
                ? A[(ph * TILE_WIDTH + tx) * lda + row]
                : A[row * lda + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        if (column < N && (ph * TILE_WIDTH + ty) < K) {
            Nds[ty][tx] = transB
                ? B[column * ldb + ph * TILE_WIDTH + ty]
                : B[(ph * TILE_WIDTH + ty) * ldb + column];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (size_t k = 0; k < TILE_WIDTH; k++) {
            float Mds_value = static_cast<float>(Mds[ty][k]);
            float Nds_value = static_cast<float>(Nds[k][tx]);
            pValue += Mds_value * Nds_value;
        }
        __syncthreads();
    }

    if (row < M && column < N) {
        const float cValue = beta == 0.0f ? 0.0f : static_cast<float>(C[row * ldc + column]);
        C[row * ldc + column] = (scalar_t)(alpha * pValue + beta * cValue);
    }
}
