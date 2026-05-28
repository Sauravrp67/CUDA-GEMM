#pragma once

template <typename scalar_t, const uint TILE_WIDTH_M, const uint TILE_WIDTH_K, const uint TILE_WIDTH_N, const uint W_TILE_WIDTH_M, const uint W_TILE_WIDTH_N, const uint THREAD_M, const uint THREAD_N>
__global__ void warptile_register2DTiledSgemm(
    const float* A, const int M, const int K, const int lda, bool transA,
    const float* B, const int N, const int ldb, bool transB,
    float* C, const int ldc,
    const float alpha, const float beta
) {
    const uint tx{threadIdx.x};
    const uint ty{threadIdx.y};
    const uint bx{blockIdx.x};
    const uint by{blockIdx.y};

    const uint linear_id{ty * blockDim.x + tx};
    const uint num_threads{blockDim.x * blockDim.y};

    const uint warp_idx{linear_id / 32};
    const uint warp_row{warp_idx / (TILE_WIDTH_N / W_TILE_WIDTH_N)};
    const uint warp_col{warp_idx % (TILE_WIDTH_N / W_TILE_WIDTH_N)};
    const uint threadIdxInWarp = (linear_id % 32);

    const uint global_row_block_offset{by * TILE_WIDTH_M};
    const uint global_col_block_offset{bx * TILE_WIDTH_N};
    const uint global_row_warp_offset{warp_row * W_TILE_WIDTH_M};
    const uint global_col_warp_offset{warp_col * W_TILE_WIDTH_N};

    constexpr uint warp_N_iter = 2;
    constexpr uint warp_M_iter = (W_TILE_WIDTH_M * W_TILE_WIDTH_N) / (32 * THREAD_N * THREAD_M * warp_N_iter);
    constexpr uint warp_SUB_M = W_TILE_WIDTH_M / warp_M_iter;
    constexpr uint warp_SUB_N = W_TILE_WIDTH_N / warp_N_iter;

    const uint threadIdx_col_InWarp = (threadIdxInWarp % (warp_SUB_N / THREAD_N));
    const uint threadIdx_row_InWarp = (threadIdxInWarp / (warp_SUB_N / THREAD_N));

    float PVALUES[warp_M_iter * THREAD_M][warp_N_iter * THREAD_N] = {0.0f};
    float a_reg[warp_M_iter * THREAD_M];
    float b_reg[warp_N_iter * THREAD_N];

    __shared__ float Mds[TILE_WIDTH_K][TILE_WIDTH_M + 4];
    __shared__ float Nds[TILE_WIDTH_K][TILE_WIDTH_N];

    uint global_k;

    for (size_t ph = 0; ph < (K + TILE_WIDTH_K - 1) / TILE_WIDTH_K; ph++) {
        for (size_t tid = linear_id; tid < TILE_WIDTH_K * TILE_WIDTH_M / 4; tid += num_threads) {
            uint smem_row_A = tid / (TILE_WIDTH_K / 4);
            uint smem_col_A = tid % (TILE_WIDTH_K / 4);
            uint global_row = global_row_block_offset + smem_row_A;
            global_k = ph * TILE_WIDTH_K + smem_col_A * 4;

            if (global_row < M && (global_k + 3) < K) {
                float4 tempA = reinterpret_cast<const float4*>(&A[global_row * lda + global_k])[0];
                Mds[(smem_col_A * 4) + 0][smem_row_A] = tempA.x;
                Mds[(smem_col_A * 4) + 1][smem_row_A] = tempA.y;
                Mds[(smem_col_A * 4) + 2][smem_row_A] = tempA.z;
                Mds[(smem_col_A * 4) + 3][smem_row_A] = tempA.w;
            } else {
                for (size_t lane = 0; lane < 4; ++lane) {
                    uint k_lane = global_k + lane;
                    Mds[smem_col_A * 4 + lane][smem_row_A] =
                        (global_row < M && k_lane < K) ? A[global_row * lda + k_lane] : 0.0f;
                }
            }
        }

        for (size_t tid = linear_id; tid < TILE_WIDTH_K * TILE_WIDTH_N / 4; tid += num_threads) {
            uint smem_row_B = tid / (TILE_WIDTH_N / 4);
            uint smem_col_B = tid % (TILE_WIDTH_N / 4);
            uint global_col = global_col_block_offset + smem_col_B * 4;
            global_k = ph * TILE_WIDTH_K + smem_row_B;

            if ((global_col + 3) < N && global_k < K) {
                float4 tempB = reinterpret_cast<const float4*>(&B[global_k * ldb + global_col])[0];
                reinterpret_cast<float4*>(&Nds[smem_row_B][smem_col_B * 4])[0] = tempB;
            } else {
                for (size_t lane = 0; lane < 4; ++lane) {
                    uint col_lane = global_col + lane;
                    Nds[smem_row_B][smem_col_B * 4 + lane] =
                        (col_lane < N && global_k < K) ? B[global_k * ldb + col_lane] : 0.0f;
                }
            }
        }
        __syncthreads();

        for (size_t k = 0; k < TILE_WIDTH_K; k++) {
            for (size_t subTile_row = 0; subTile_row < warp_M_iter; ++subTile_row) {
                float4 tempAs = reinterpret_cast<const float4*>(
                    &Mds[k][global_row_warp_offset + subTile_row * warp_SUB_M + threadIdx_row_InWarp * 4]
                )[0];
                a_reg[subTile_row * THREAD_M + 0] = tempAs.x;
                a_reg[subTile_row * THREAD_M + 1] = tempAs.y;
                a_reg[subTile_row * THREAD_M + 2] = tempAs.z;
                a_reg[subTile_row * THREAD_M + 3] = tempAs.w;
            }

            for (size_t subTile_col = 0; subTile_col < warp_N_iter; ++subTile_col) {
                float4 tempBs = reinterpret_cast<const float4*>(
                    &Nds[k][global_col_warp_offset + subTile_col * warp_SUB_N + threadIdx_col_InWarp * 4]
                )[0];
                b_reg[subTile_col * THREAD_N + 0] = tempBs.x;
                b_reg[subTile_col * THREAD_N + 1] = tempBs.y;
                b_reg[subTile_col * THREAD_N + 2] = tempBs.z;
                b_reg[subTile_col * THREAD_N + 3] = tempBs.w;
            }

            for (size_t subTile_row = 0; subTile_row < warp_M_iter; ++subTile_row) {
                for (size_t subTile_col = 0; subTile_col < warp_N_iter; ++subTile_col) {
                    for (size_t i = 0; i < THREAD_M; ++i) {
                        for (size_t j = 0; j < THREAD_N; ++j) {
                            PVALUES[subTile_row * THREAD_M + i][subTile_col * THREAD_N + j] +=
                                a_reg[subTile_row * THREAD_M + i] * b_reg[subTile_col * THREAD_N + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (size_t subTile_row = 0; subTile_row < warp_M_iter; ++subTile_row) {
        for (size_t subTile_col = 0; subTile_col < warp_N_iter; ++subTile_col) {
            for (size_t i = 0; i < THREAD_M; ++i) {
                for (size_t j = 0; j < THREAD_N; j += 4) {
                    uint global_cRow =
                        global_row_block_offset + global_row_warp_offset + subTile_row * warp_SUB_M
                        + threadIdx_row_InWarp * THREAD_M + i;
                    uint global_cCol =
                        global_col_block_offset + global_col_warp_offset + subTile_col * warp_SUB_N
                        + threadIdx_col_InWarp * THREAD_N + j;

                    if (global_cRow >= M) {
                        continue;
                    }

                    if ((global_cCol + 3) < N) {
                        float4 tempC = beta == 0.0
                            ? make_float4(0.0, 0.0, 0.0, 0.0)
                            : reinterpret_cast<float4*>(&C[global_cRow * ldc + global_cCol])[0];

                        tempC.x = (alpha * PVALUES[subTile_row * THREAD_M + i][subTile_col * THREAD_N + j] + beta * tempC.x);
                        tempC.y = (alpha * PVALUES[subTile_row * THREAD_M + i][subTile_col * THREAD_N + j + 1] + beta * tempC.y);
                        tempC.z = (alpha * PVALUES[subTile_row * THREAD_M + i][subTile_col * THREAD_N + j + 2] + beta * tempC.z);
                        tempC.w = (alpha * PVALUES[subTile_row * THREAD_M + i][subTile_col * THREAD_N + j + 3] + beta * tempC.w);

                        reinterpret_cast<float4*>(&C[global_cRow * ldc + global_cCol])[0] = tempC;
                    } else {
                        for (size_t lane = 0; lane < 4; lane++) {
                            uint col_lane = global_cCol + lane;
                            if (col_lane < N) {
                                float c_value = beta == 0.0 ? 0.0f : C[global_cRow * ldc + col_lane];
                                C[global_cRow * ldc + col_lane] =
                                    alpha * PVALUES[subTile_row * THREAD_M + i][subTile_col * THREAD_N + j + lane]
                                    + beta * c_value;
                            }
                        }
                    }
                }
            }
        }
    }
}
