#include <torch/extension.h>

#include "bindings/register.h"
#include "primitives/matmul/matmul.h"

namespace py = pybind11;

void register_primitives(py::module_& m) {
    m.def(
        "gemm",
        &gemm,
        "Naive GEMM forward pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );

    m.def(
        "sgemm",
        &sgemm,
        "Tiled GEMM forward pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );

    m.def(
        "threadcoarsedTiledgemm",
        &threadcoarsedTiledgemm,
        "Non-Square Tiled GEMM Forward Pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );

    m.def(
        "reg2DTiledsgemm",
        &reg2DTiledsgemm,
        "2D register Tiled GEMM Forward Pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );

    m.def(
        "reg1DTiledsgemm",
        &reg1DTiledsgemm,
        "1D register Tiled GEMM Forward Pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );

    m.def(
        "vec_reg2DTiledsgemm",
        &vec_reg2DTiledsgemm,
        "Vectorized 2D register Tiled GEMM Forward Pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );

    m.def(
        "warpTiled_gemm",
        &warpTiled_gemm,
        "warpTiled GEMM Forward Pass (structured backend)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("transA"),
        py::arg("transB")
    );
}
