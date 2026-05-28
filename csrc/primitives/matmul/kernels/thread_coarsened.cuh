#pragma once

template <typename scalar_t, const int TILE_WIDTH_M, const int TILE_WIDTH_K, const int TILE_WIDTH_N, const int TM, const int TN>
__global__ void ThreadCoarsedTiledgemm(
    const scalar_t* A, const int M, const int K, const int lda, bool transA,
    const scalar_t* B, const int N, const int ldb, bool transB,
    scalar_t* C, const int ldc,
    const float alpha, const float beta
) {
    const uint bx{blockIdx.x};
    const uint by{blockIdx.y};
    const uint tx{threadIdx.x};
    const uint ty{threadIdx.y};

    const uint linear_tid{ty * blockDim.x + tx};
    const uint num_threads{blockDim.x * blockDim.y};

    __shared__ scalar_t Mds[TILE_WIDTH_M][TILE_WIDTH_K];
    __shared__ scalar_t Nds[TILE_WIDTH_K][TILE_WIDTH_N];

    const uint row{by * TILE_WIDTH_M + ty * TM};
    const uint column{bx * TILE_WIDTH_N + tx};

    float PVALUES[TM] = {0.0f};
    size_t global_row, global_col, global_k, smem_row, smem_col;

    for (size_t ph = 0; ph < (K + TILE_WIDTH_K - 1) / TILE_WIDTH_K; ph++) {
        for (size_t tid = linear_tid; tid < TILE_WIDTH_M * TILE_WIDTH_K; tid += num_threads) {
            smem_row = tid / TILE_WIDTH_K;
            smem_col = tid % TILE_WIDTH_K;
            global_row = by * TILE_WIDTH_M + smem_row;
            global_k = ph * TILE_WIDTH_K + smem_col;

            if (global_row < M && global_k < K) {
                Mds[smem_row][smem_col] = transA
                    ? A[global_k * lda + global_row]
                    : A[global_row * lda + global_k];
            } else {
                Mds[smem_row][smem_col] = 0.0f;
            }
        }

        for (size_t tid = linear_tid; tid < TILE_WIDTH_K * TILE_WIDTH_N; tid += num_threads) {
            smem_row = tid / TILE_WIDTH_N;
            smem_col = tid % TILE_WIDTH_N;
            global_col = bx * TILE_WIDTH_N + smem_col;
            global_k = ph * TILE_WIDTH_K + smem_row;

            if (global_col < N && global_k < K) {
                Nds[smem_row][smem_col] = transB
                    ? B[global_col * ldb + global_k]
                    : B[global_k * ldb + global_col];
            } else {
                Nds[smem_row][smem_col] = 0.0f;
            }
        }

        __syncthreads();

        for (size_t k = 0; k < TILE_WIDTH_K; k++) {
            float Nds_temp = static_cast<float>(Nds[k][tx]);
            for (size_t i = 0; i < TM; i++) {
                PVALUES[i] += static_cast<float>(Mds[ty * TM + i][k]) * Nds_temp;
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < TM; i++) {
        if ((row + i) < M && column < N) {
            const float cValue = beta == 0.0f ? 0.0f : static_cast<float>(C[(row + i) * ldc + column]);
            C[(row + i) * ldc + column] = (scalar_t)(alpha * PVALUES[i] + beta * cValue);
        }
    }
}
