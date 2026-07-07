#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace WindowsApp { namespace Compute { namespace Kernels
{
    // WMMA tile dimensions (Volta/Ampere tensor cores: 16x16x16)
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // =====================================================================
    // Tensor Core: Batch Matrix Multiply (FP16 accumulate to FP32)
    // =====================================================================
    // Multiplies a batch of matrices: C[i] = A[i] * B[i]
    // A: batchA_M x batchA_K, B: batchA_K x batchB_N, C: batchA_M x batchB_N
    // All matrices row-major, stored consecutively for each batch element.
    // Uses WMMA for tensor core acceleration when available.
    __global__ void TensorBatchMatMul(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float* __restrict__ C,
        int batchA_M, int batchA_K, int batchB_N,
        int batchSize);

    // =====================================================================
    // Tensor Core: Homography Estimation (8-DOF DLT)
    // =====================================================================
    // Builds the coefficient matrix A for the Direct Linear Transform:
    //   For each correspondence (x,y) -> (x',y'):
    //   [ -x  -y  -1   0   0   0  x*x'  y*x'  x' ]
    //   [  0   0   0  -x  -y  -1  x*y'  y*y'  y' ]
    // Then computes A^T * A (9x9) using tensor cores.
    // pointPairs: [x0,y0,x0',y0', x1,y1,x1',y1', ...] (float)
    // AtA: output 9x9 matrix (float, row-major)
    // numPairs: number of point correspondences
    __global__ void TensorBuildAtA(
        const float* __restrict__ pointPairs,
        float* __restrict__ AtA,
        int numPairs);

    // =====================================================================
    // Tensor Core: Solve 9x9 system via Cholesky (FP16 compute)
    // =====================================================================
    // Solves AtA * h = Atb for the homography vector h.
    // AtA: 9x9 symmetric positive-definite (float, row-major)
    // Atb: 9x1 (float)
    // h: output 9x1 (float)
    // Uses FP16 tensor cores for the forward/backward substitution.
    __global__ void TensorSolveHomography(
        const float* __restrict__ AtA,
        const float* __restrict__ Atb,
        float* __restrict__ h);

    // =====================================================================
    // Utility: FP32 -> FP16 conversion (batch)
    // =====================================================================
    __global__ void FloatToHalf(
        const float* __restrict__ input,
        __half* __restrict__ output,
        int count);

    // =====================================================================
    // Utility: FP16 -> FP32 conversion (batch)
    // =====================================================================
    __global__ void HalfToFloat(
        const __half* __restrict__ input,
        float* __restrict__ output,
        int count);

    // =====================================================================
    // Tensor Core: Normal Equations for Bundle Adjustment
    // =====================================================================
    // Computes J^T * J and J^T * r for the Levenberg-Marquardt step.
    // J: Jacobian matrix (numResiduals x numParams), FP16
    // r: residual vector (numResiduals), FP32
    // JtJ: output numParams x numParams (FP32, symmetric)
    // Jtr: output numParams x 1 (FP32)
    __global__ void TensorNormalEquations(
        const __half* __restrict__ J,
        const float* __restrict__ r,
        float* __restrict__ JtJ,
        float* __restrict__ Jtr,
        int numResiduals, int numParams);
    }
    }
}
