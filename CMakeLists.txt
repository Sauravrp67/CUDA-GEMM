cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CUDA_GEMM LANGUAGES CXX CUDA)

execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" GPU_ARCH_LIST "${GPU_ARCH}")
list(GET GPU_ARCH_LIST 0 GPU_ARCH)
string(REPLACE "." "" GPU_ARCH_CLEAN ${GPU_ARCH})
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

message(STATUS "Targeting CUDA Architecture: sm_${GPU_ARCH_CLEAN}")
set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
message(STATUS "Locked to Architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")

find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; import os; print(os.path.join(sys.prefix, 'lib'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Syncing with PyTorch ABI: ${TORCH_ABI}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
    CUDA_GEMM_OUTPUT_DIR
    "${CMAKE_SOURCE_DIR}/src/cuda_gemm/backends/cuda"
    CACHE PATH
    "Output directory for the CUDA GEMM extension."
)
set(
    CUDA_GEMM_OUTPUT_NAME
    "cuda_gemm_cuda"
    CACHE STRING
    "Basename of the CUDA GEMM extension."
)

function(add_torch_cuda_extension target_name output_name output_dir)
    add_library(${target_name} SHARED ${ARGN})

    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math -Xptxas=-v -O3 -Xcompiler -fPIC>
    )

    target_include_directories(${target_name} PRIVATE
        csrc
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(${target_name} PRIVATE
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
        "${TORCH_LIB_DIR}/lib/libtorch_python.so"
    )

    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "${output_name}"
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_LIB_DIR}/lib/;${PYTHON_LIB_DIR}"
    )
endfunction()

set(CUDA_GEMM_SOURCES
    csrc/bindings/runtime.cpp
    csrc/bindings/primitives.cpp
    csrc/primitives/matmul/matmul.cu
)

add_torch_cuda_extension(
    cuda_gemm_cuda
    "${CUDA_GEMM_OUTPUT_NAME}"
    "${CUDA_GEMM_OUTPUT_DIR}"
    ${CUDA_GEMM_SOURCES}
)

message(STATUS "CUDA GEMM extension output directory: ${CUDA_GEMM_OUTPUT_DIR}")
