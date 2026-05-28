#pragma once

template <typename scalar_t, const int TILE_WIDTH_M, const int TILE_WIDTH_K, const int TILE_WIDTH_N, const int THREAD_M, const int THREAD_N>
__global__ void register1DTiledSgemm(
    const scalar_t* A, const int M, const int K, const int lda, bool transA,
    const scalar_t* B, const int N, const int ldb, bool transB,
    scalar_t* C, const int ldc,
    const float alpha, const float beta
) {
    const size_t bx{blockIdx.x};
    const size_t by{blockIdx.y};
    const size_t tx{threadIdx.x};
    const size_t ty{threadIdx.y};

    const size_t linear_tid{ty * blockDim.x + tx};
    const size_t num_threads = blockDim.x * blockDim.y;

    __shared__ scalar_t Mds[TILE_WIDTH_M][TILE_WIDTH_K];
    __shared__ scalar_t Nds[TILE_WIDTH_K][TILE_WIDTH_N];

    const size_t row{by * TILE_WIDTH_M + ty * THREAD_M};
    const size_t column{bx * TILE_WIDTH_N + tx * THREAD_N};

    float PVALUES[THREAD_M][THREAD_N] = {0.0f};
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
            for (size_t i = 0; i < THREAD_N; i++) {
                float Nds_temp = static_cast<float>(Nds[k][tx * THREAD_N + i]);
                for (size_t j = 0; j < THREAD_M; j++) {
                    PVALUES[j][i] += static_cast<float>(Mds[ty * THREAD_M + j][k]) * Nds_temp;
                }
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < THREAD_M; i++) {
        for (size_t j = 0; j < THREAD_N; j++) {
            if ((row + i) < M && (column + j) < N) {
                const float c_value = beta == 0.0f ? 0.0f : static_cast<float>(C[(row + i) * ldc + (column + j)]);
                C[(row + i) * ldc + (column + j)] = (scalar_t)(alpha * PVALUES[i][j] + beta * c_value);
            }
        }
    }
}

template <typename scalar_t, const int TILE_WIDTH_M, const int TILE_WIDTH_K, const int TILE_WIDTH_N, const int THREAD_M, const int THREAD_N>
__global__ void register2DTiledSgemm(
    const scalar_t* A, const int M, const int K, const int lda, bool transA,
    const scalar_t* B, const int N, const int ldb, bool transB,
    scalar_t* C, const int ldc,
    const float alpha, const float beta
) {
    const size_t bx{blockIdx.x};
    const size_t by{blockIdx.y};
    const size_t tx{threadIdx.x};
    const size_t ty{threadIdx.y};

    const size_t linear_tid{ty * blockDim.x + tx};
    const size_t num_threads = blockDim.x * blockDim.y;

    __shared__ scalar_t Mds[TILE_WIDTH_M][TILE_WIDTH_K];
    __shared__ scalar_t Nds[TILE_WIDTH_K][TILE_WIDTH_N];

    const size_t row{by * TILE_WIDTH_M + ty * THREAD_M};
    const size_t column{bx * TILE_WIDTH_N + tx * THREAD_N};

    float PVALUES[THREAD_M][THREAD_N] = {0.0f};
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
            scalar_t a_reg[THREAD_M];
            scalar_t b_reg[THREAD_N];

            for (size_t i = 0; i < THREAD_M; i++) {
                a_reg[i] = Mds[ty * THREAD_M + i][k];
            }
            for (size_t j = 0; j < THREAD_N; j++) {
                b_reg[j] = Nds[k][tx * THREAD_N + j];
            }
            for (size_t i = 0; i < THREAD_M; i++) {
                for (size_t j = 0; j < THREAD_N; j++) {
                    PVALUES[i][j] += static_cast<float>(a_reg[i]) * static_cast<float>(b_reg[j]);
                }
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < THREAD_M; i++) {
        for (size_t j = 0; j < THREAD_N; j++) {
            if ((row + i) < M && (column + j) < N) {
                const float c_value = beta == 0.0f ? 0.0f : static_cast<float>(C[(row + i) * ldc + (column + j)]);
                C[(row + i) * ldc + (column + j)] = (scalar_t)(alpha * PVALUES[i][j] + beta * c_value);
            }
        }
    }
}
