#pragma once

template <typename scalar_t, const uint TILE_WIDTH_M, const uint TILE_WIDTH_K, const uint TILE_WIDTH_N, const uint THREAD_M, const uint THREAD_N>
__global__ void vectorized_register2DTiledSgemm(
    const float* A, const int M, const int K, const int lda, bool transA,
    const float* B, const int N, const int ldb, bool transB,
    float* C, const int ldc,
    const float alpha, const float beta
) {
    const uint bx{blockIdx.x};
    const uint by{blockIdx.y};
    const uint tx{threadIdx.x};
    const uint ty{threadIdx.y};

    const uint linear_id{ty * blockDim.x + tx};
    const uint num_threads{blockDim.x * blockDim.y};

    __shared__ float Mds[TILE_WIDTH_K][TILE_WIDTH_M + 4];
    __shared__ float Nds[TILE_WIDTH_K][TILE_WIDTH_N];

    const uint row_block_offset{by * TILE_WIDTH_M};
    const uint column_block_offset{bx * TILE_WIDTH_N};

    float PVALUE[THREAD_M][THREAD_N] = {0.0f};
    float a_reg[THREAD_M];
    float b_reg[THREAD_N];

    uint global_k;

    for (size_t ph = 0; ph < (K + TILE_WIDTH_K - 1) / TILE_WIDTH_K; ph++) {
        for (size_t tid = linear_id; tid < TILE_WIDTH_M * TILE_WIDTH_K / 4; tid += num_threads) {
            uint smem_row_A = tid / (TILE_WIDTH_K / 4);
            uint smem_col_A = tid % (TILE_WIDTH_K / 4);
            const uint global_row_A = row_block_offset + smem_row_A;
            global_k = ph * TILE_WIDTH_K + smem_col_A * 4;

            if (global_row_A < M && (global_k + 3) < K) {
                float4 vecA = reinterpret_cast<const float4*>(&A[global_row_A * lda + global_k])[0];
                Mds[(smem_col_A * 4 + 0)][smem_row_A] = vecA.x;
                Mds[(smem_col_A * 4 + 1)][smem_row_A] = vecA.y;
                Mds[(smem_col_A * 4 + 2)][smem_row_A] = vecA.z;
                Mds[(smem_col_A * 4 + 3)][smem_row_A] = vecA.w;
            } else {
                for (size_t lane = 0; lane < 4; ++lane) {
                    const uint k_lane = global_k + lane;
                    Mds[(smem_col_A * 4 + lane)][smem_row_A] =
                        (global_row_A < M && k_lane < K) ? A[global_row_A * lda + k_lane] : 0.0f;
                }
            }
        }

        for (size_t tid = linear_id; tid < TILE_WIDTH_K * TILE_WIDTH_N / 4; tid += num_threads) {
            const uint smem_row_B = tid / (TILE_WIDTH_N / 4);
            const uint smem_col_B = tid % (TILE_WIDTH_N / 4);
            const uint global_col_B = column_block_offset + smem_col_B * 4;
            global_k = ph * TILE_WIDTH_K + smem_row_B;

            if ((global_col_B + 3) < N && global_k < K) {
                float4 vecB = reinterpret_cast<const float4*>(&B[global_k * ldb + global_col_B])[0];
                reinterpret_cast<float4*>(&Nds[smem_row_B][(smem_col_B * 4)])[0] = vecB;
            } else {
                for (size_t lane = 0; lane < 4; ++lane) {
                    const uint col_lane = global_col_B + lane;
                    Nds[smem_row_B][(smem_col_B * 4) + lane] =
                        (global_k < K && col_lane < N) ? B[global_k * ldb + col_lane] : 0.0f;
                }
            }
        }
        __syncthreads();

        for (size_t k = 0; k < TILE_WIDTH_K; k++) {
            for (size_t i = 0; i < THREAD_M; i += 4) {
                uint smem_row_index = ty * THREAD_M + i;
                float4 temp_Mds = reinterpret_cast<const float4*>(&Mds[k][smem_row_index])[0];
                a_reg[i + 0] = temp_Mds.x;
                a_reg[i + 1] = temp_Mds.y;
                a_reg[i + 2] = temp_Mds.z;
                a_reg[i + 3] = temp_Mds.w;
            }

            for (size_t j = 0; j < THREAD_N; j += 4) {
                uint smem_column_index = tx * THREAD_N + j;
                float4 temp_Nds = reinterpret_cast<const float4*>(&Nds[k][smem_column_index])[0];
                b_reg[j + 0] = temp_Nds.x;
                b_reg[j + 1] = temp_Nds.y;
                b_reg[j + 2] = temp_Nds.z;
                b_reg[j + 3] = temp_Nds.w;
            }

            for (size_t i = 0; i < THREAD_M; i++) {
                for (size_t j = 0; j < THREAD_N; j++) {
                    PVALUE[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < THREAD_M; i++) {
        for (size_t j = 0; j < THREAD_N; j += 4) {
            uint global_row_index = row_block_offset + ty * THREAD_M + i;
            uint global_column_index = column_block_offset + tx * THREAD_N + j;
            if (global_row_index >= M) {
                continue;
            }

            if ((global_column_index + 3) < N) {
                float4 tempC = beta == 0.0
                    ? make_float4(0.0, 0.0, 0.0, 0.0)
                    : reinterpret_cast<float4*>(&C[global_row_index * ldc + global_column_index])[0];

                tempC.x = (alpha * PVALUE[i][j] + beta * tempC.x);
                tempC.y = (alpha * PVALUE[i][j + 1] + beta * tempC.y);
                tempC.z = (alpha * PVALUE[i][j + 2] + beta * tempC.z);
                tempC.w = (alpha * PVALUE[i][j + 3] + beta * tempC.w);

                reinterpret_cast<float4*>(&C[global_row_index * ldc + global_column_index])[0] = tempC;
            } 
            else {
                for (size_t lane = 0; lane < 4; ++lane) {
                    const uint col_lane = global_column_index + lane;
                    if (col_lane < N) {
                        const float c_value = beta == 0.0f ? 0.0f : C[global_row_index * ldc + col_lane];
                        C[global_row_index * ldc + col_lane] = alpha * PVALUE[i][j + lane] + beta * c_value;
                    }
                }
            }
        }
    }
}
