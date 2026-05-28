#pragma once

#include <torch/extension.h>

at::Tensor gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor sgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor threadcoarsedTiledgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor reg2DTiledsgemm(
    const at::Tensor& input, 
    const at::Tensor& weight, 
    bool transA, 
    bool transB
);

at::Tensor reg1DTiledsgemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);

at::Tensor vec_reg2DTiledsgemm(
    const at::Tensor& input, 
    const at::Tensor& weight, 
    bool transA, 
    bool transB
);

at::Tensor warpTiled_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transA,
    bool transB
);