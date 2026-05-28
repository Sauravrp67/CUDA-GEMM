#pragma once

#include <ATen/ATen.h> 

struct GemmProblem {
    const at::Tensor& input;
    const at::Tensor& weight;
    at::Tensor output;
    int M;
    int K;
    int N;
    int lda;
    int ldb;
    int ldc;
    bool transA;
    bool transB;
    float alpha;
    float beta;
};

static GemmProblem make_gemm_problem(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    TORCH_CHECK(input.is_cuda(), "GEMM input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "GEMM weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "GEMM input must be 2D, got ", input.dim(), "D");
    TORCH_CHECK(weight.dim() == 2, "GEMM weight must be 2D, got ", weight.dim(), "D");
    TORCH_CHECK(
        input.scalar_type() == weight.scalar_type(),
        "GEMM input and weight must have the same dtype"
    );
    TORCH_CHECK(input.device() == weight.device(), "GEMM tensors must be on the same device");
    TORCH_CHECK(input.is_contiguous(), "GEMM input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "GEMM weight must be contiguous");

    const int input_rows = input.size(0);
    const int input_cols = input.size(1);
    const int weight_rows = weight.size(0);
    const int weight_cols = weight.size(1);

    const int M = transA ? input_cols : input_rows;
    const int K = transA ? input_rows : input_cols;

    const int weight_K = transB ? weight_cols : weight_rows;
    const int N = transB ? weight_rows : weight_cols;

    const int lda = input_cols;
    const int ldb = weight_cols;

    TORCH_CHECK(
        K == weight_K,
        "Shape mismatch for GEMM: input K (", K,
        ") must match weight K (", weight_K, ")"
    );

    return GemmProblem{
        input,
        weight,
        at::empty({M, N}, input.options()),
        M,
        K,
        N,
        lda,
        ldb,
        N,
        transA,
        transB,
        1.0f,
        0.0f,
    };
}

struct NaiveGemmLauncher {
    static constexpr int BM = 16;
    static constexpr int BN = 16;
    static constexpr int TM = 1;
    static constexpr int TN = 1;
    static constexpr const char* kDispatchName = "naive_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        naive_gemm<scalar_t><<<grid, block>>>(
            problem.input.data_ptr<scalar_t>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<scalar_t>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<scalar_t>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct TiledGemmLauncher {
    static constexpr int BM = 32;
    static constexpr int BN = 32;
    static constexpr int TM = 1;
    static constexpr int TN = 1;
    static constexpr const char* kDispatchName = "tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        squareTiledgemm<scalar_t, BM><<<grid, block>>>(
            problem.input.data_ptr<scalar_t>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<scalar_t>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<scalar_t>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct VarTiledGemmLauncher {
    static constexpr int BM = 128;
    static constexpr int BN = 32;
    static constexpr int BK = 8;
    static constexpr int TM = 8;
    static constexpr int TN = 1;
    static constexpr const char* kDispatchName = "var_tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        ThreadCoarsedTiledgemm<scalar_t, BM, BK, BN, TM, TN><<<grid, block>>>(
            problem.input.data_ptr<scalar_t>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<scalar_t>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<scalar_t>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct reg2DTiledsgemmLauncher {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int TM = 8;
    static constexpr int TN = 8;
    static constexpr const char* kDispatchName = "Reg_Tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        register2DTiledSgemm<scalar_t, BM, BK, BN, TM, TN><<<grid, block>>>(
            problem.input.data_ptr<scalar_t>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<scalar_t>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<scalar_t>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct reg1DTiledsgemmLauncher {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int TM = 8;
    static constexpr int TN = 8;
    static constexpr const char* kDispatchName = "Reg_Tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        register1DTiledSgemm<scalar_t, BM, BK, BN, TM, TN><<<grid, block>>>(
            problem.input.data_ptr<scalar_t>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<scalar_t>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<scalar_t>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct vectorized_reg2DTiledsgemmLauncher {
    static constexpr uint BM = 128;
    static constexpr uint BN = 128;
    static constexpr uint BK = 16;

    static constexpr uint TM = 8;
    static constexpr uint TN = 8;
    
    static constexpr const char* kDispatchName = "Vectorized_Reg_Tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        vectorized_register2DTiledSgemm<scalar_t, BM, BK, BN, TM, TN><<<grid, block>>>(
            problem.input.data_ptr<float>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<float>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<float>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
};

struct warpTiled_matmulLauncher{
    static constexpr uint BM = 128;
    static constexpr uint BN = 128;
    static constexpr uint BK = 16;

    static constexpr uint sub_TM = 4;
    static constexpr uint sub_TN = 4;

    static constexpr uint TM = 2 * sub_TM;
    static constexpr uint TN = 2 * sub_TN;

    static constexpr uint WM = 32;
    static constexpr uint WN = 64;

    static constexpr const char* kDispatchName = "Vectorized_Reg_Tiled_gemm_structured";

    template <typename scalar_t>
    static void launch(const GemmProblem& problem, dim3 grid, dim3 block) {
        warptile_register2DTiledSgemm<scalar_t, BM, BK, BN, WM, WN, sub_TM, sub_TN><<<grid, block>>>(
            problem.input.data_ptr<float>(),
            problem.M,
            problem.K,
            problem.lda,
            problem.transA,
            problem.weight.data_ptr<float>(),
            problem.N,
            problem.ldb,
            problem.transB,
            problem.output.data_ptr<float>(),
            problem.ldc,
            problem.alpha,
            problem.beta
        );
    }
    
};

template <typename Launcher>
static at::Tensor dispatch_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    GemmProblem problem = make_gemm_problem(input, weight, transA, transB);

    dim3 block(Launcher::BN / Launcher::TN, Launcher::BM / Launcher::TM, 1);
    dim3 grid(
        (problem.N + Launcher::BN - 1) / Launcher::BN,
        (problem.M + Launcher::BM - 1) / Launcher::BM,
        1
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        Launcher::kDispatchName,
        [&] {
            Launcher::template launch<scalar_t>(problem, grid, block);
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return problem.output;
}

at::Tensor gemm(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB) {
    return dispatch_gemm<NaiveGemmLauncher>(input, weight, transA, transB);
}

at::Tensor sgemm(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB) {
    return dispatch_gemm<TiledGemmLauncher>(input, weight, transA, transB);
}

at::Tensor threadcoarsedTiledgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    return dispatch_gemm<VarTiledGemmLauncher>(input, weight, transA, transB);
}

at::Tensor reg2DTiledsgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    return dispatch_gemm<reg2DTiledsgemmLauncher>(input, weight, transA, transB);
}

at::Tensor reg1DTiledsgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    return dispatch_gemm<reg1DTiledsgemmLauncher>(input, weight, transA, transB);
}

at::Tensor vec_reg2DTiledsgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    TORCH_CHECK(
        input.scalar_type() == at::kFloat,
        "vec_reg2DTiledsgemm only supports float32 tensors"
    );
    TORCH_CHECK(
        !transA && !transB,
        "vec_reg2DTiledsgemm currently supports only non-transposed inputs"
    );
    return dispatch_gemm<vectorized_reg2DTiledsgemmLauncher>(input, weight, transA, transB);
}

at::Tensor warpTiled_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
) {
    TORCH_CHECK(
        input.scalar_type() == at::kFloat,
        "warpTiled_gemm only supports float32 tensors"
    );
    TORCH_CHECK(
        !transA && !transB,
        "warpTiled_gemm currently supports only non-transposed inputs"
    );
    return dispatch_gemm<warpTiled_matmulLauncher>(input, weight, transA, transB);
}
