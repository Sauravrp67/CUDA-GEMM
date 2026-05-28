"""Loader for the CUDA GEMM extension module."""

from __future__ import annotations


def _backend():
    from . import cuda_gemm_cuda

    return cuda_gemm_cuda


def gemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().gemm(A, B, transA, transB)


def sgemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().sgemm(A, B, transA, transB)


def tiled_gemm(A, B, transA: bool = False, transB: bool = False):
    return sgemm(A, B, transA, transB)


def threadcoarsedTiledgemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().threadcoarsedTiledgemm(A, B, transA, transB)


def coarsened_tiled_matmul(A, B, transA: bool = False, transB: bool = False):
    return threadcoarsedTiledgemm(A, B, transA, transB)


def regtiled2DSgemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().reg2DTiledsgemm(A, B, transA, transB)


def reg2d_tiled_gemm(A, B, transA: bool = False, transB: bool = False):
    return regtiled2DSgemm(A, B, transA, transB)


def regtiled1DSgemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().reg1DTiledsgemm(A, B, transA, transB)


def reg1d_tiled_gemm(A, B, transA: bool = False, transB: bool = False):
    return regtiled1DSgemm(A, B, transA, transB)


def vec_regtiled2DSgemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().vec_reg2DTiledsgemm(A, B, transA, transB)


def vec_reg2d_tiled_gemm(A, B, transA: bool = False, transB: bool = False):
    return vec_regtiled2DSgemm(A, B, transA, transB)


def warpTiled_vec_gemm(A, B, transA: bool = False, transB: bool = False):
    return _backend().warpTiled_gemm(A, B, transA, transB)


def warpTiled_gemm(A, B, transA: bool = False, transB: bool = False):
    return warpTiled_vec_gemm(A, B, transA, transB)
