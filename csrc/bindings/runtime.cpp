#include <torch/extension.h>

#include "bindings/register.h"

PYBIND11_MODULE(cuda_gemm_cuda, m) {
    register_primitives(m);
}
