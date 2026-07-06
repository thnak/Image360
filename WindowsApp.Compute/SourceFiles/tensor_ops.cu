#include "HeaderFiles/tensor_ops.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace WindowsApp::Compute::Kernels
{
    // =====================================================================
    // Tensor Core Batch Matrix Multiply
    // Each warp computes one 16x16 output tile using WMMA.
    // =====================================================================
    __global__ void TensorBatchMatMul(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float* __restrict__ C,
        int batchA_M, int batchA_K, int batchB_N,
        int batchSize)
    {
        // Each block handles one batch element
        // Each warp handles one 16x16 output tile
        int batchIdx = blockIdx.z;
        if (batchIdx >= batchSize) return;

        int warpId = threadIdx.x / 32;
        int warpsPerBlock = blockDim.x / 32;

        // Tile coordinates for this warp
        int tilesPerRow = (batchB_N + WMMA_N - 1) / WMMA_N;
        int tileRow = (warpId / tilesPerRow) + blockIdx.y * (blockDim.x / 32 / tilesPerRow);
        int tileCol = (warpId % tilesPerRow) + blockIdx.x * (tilesPerRow);

        int row = tileRow * WMMA_M;
        int col = tileCol * WMMA_N;

        if (row >= batchA_M || col >= batchB_N) return;

        // Base pointers for this batch element
        const __half* batchA = A + (size_t)batchIdx * batchA_M * batchA_K;
        const __half* batchB = B + (size_t)batchIdx * batchA_K * batchB_N;
        float* batchC = C + (size_t)batchIdx * batchA_M * batchB_N;

        // WMMA fragments
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Loop over K dimension in tiles of WMMA_K
        for (int k = 0; k < batchA_K; k += WMMA_K)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;

            // Bounds check for K dimension
            if (k + WMMA_K <= batchA_K)
            {
                wmma::load_matrix_sync(a_frag, batchA + row * batchA_K + k, batchA_K);
                wmma::load_matrix_sync(b_frag, batchB + k * batchB_N + col, batchB_N);
            }
            else
            {
                // Handle partial K tile with zero-padding
                __half a_tile[WMMA_M * WMMA_K] = {};
                __half b_tile[WMMA_K * WMMA_N] = {};

                int remainingK = batchA_K - k;
                for (int i = 0; i < WMMA_M && (row + i) < batchA_M; i++)
                {
                    for (int j = 0; j < remainingK; j++)
                    {
                        a_tile[i * WMMA_K + j] = batchA[(row + i) * batchA_K + k + j];
                    }
                }
                for (int i = 0; i < remainingK; i++)
                {
                    for (int j = 0; j < WMMA_N && (col + j) < batchB_N; j++)
                    {
                        b_tile[i * WMMA_N + j] = batchB[(k + i) * batchB_N + col + j];
                    }
                }

                wmma::load_matrix_sync(a_frag, a_tile, WMMA_K);
                wmma::load_matrix_sync(b_frag, b_tile, WMMA_N);
            }

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store result with bounds check
        if (row + WMMA_M <= batchA_M && col + WMMA_N <= batchB_N)
        {
            wmma::store_matrix_sync(batchC + row * batchB_N + col, c_frag, batchB_N, wmma::mem_row_major);
        }
        else
        {
            // Handle boundary tiles
            float c_tile[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(c_tile, c_frag, WMMA_N, wmma::mem_row_major);

            for (int i = 0; i < WMMA_M && (row + i) < batchA_M; i++)
            {
                for (int j = 0; j < WMMA_N && (col + j) < batchB_N; j++)
                {
                    batchC[(row + i) * batchB_N + col + j] = c_tile[i * WMMA_N + j];
                }
            }
        }
    }

    // =====================================================================
    // Tensor Core: Build A^T * A for homography DLT
    // Each thread accumulates one element of AtA from point pairs.
    // For large pair counts, uses shared memory reduction then WMMA.
    // =====================================================================
    __global__ void TensorBuildAtA(
        const float* __restrict__ pointPairs,
        float* __restrict__ AtA,
        int numPairs)
    {
        // AtA is 9x9 = 81 elements
        // Each thread computes one element (i,j) of AtA
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int i = idx / 9;
        int j = idx % 9;

        if (i >= 9 || j >= 9) return;

        // Only compute upper triangle (AtA is symmetric)
        if (j < i) return;

        float sum = 0.0f;

        for (int p = 0; p < numPairs; p++)
        {
            float x = pointPairs[p * 4 + 0];
            float y = pointPairs[p * 4 + 1];
            float xp = pointPairs[p * 4 + 2];
            float yp = pointPairs[p * 4 + 3];

            // Two rows per correspondence:
            // Row 0: [-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp]
            // Row 1: [ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp]

            float a0[9] = { -x, -y, -1.0f, 0, 0, 0, x * xp, y * xp, xp };
            float a1[9] = { 0, 0, 0, -x, -y, -1.0f, x * yp, y * yp, yp };

            sum += a0[i] * a0[j] + a1[i] * a1[j];
        }

        AtA[i * 9 + j] = sum;
        // Mirror to lower triangle
        if (i != j)
            AtA[j * 9 + i] = sum;
    }

    // =====================================================================
    // Tensor Core: Solve 9x9 homography via Cholesky decomposition
    // FP16 compute with FP32 accumulation for numerical stability.
    // =====================================================================
    __global__ void TensorSolveHomography(
        const float* __restrict__ AtA,
        const float* __restrict__ Atb,
        float* __restrict__ h)
    {
        // Cholesky decomposition: AtA = L * L^T
        // Then solve L*y = Atb (forward), L^T*h = y (backward)
        // 9x9 is small enough for a single thread with registers

        float L[9][9] = {};
        float y[9] = {};

        // Cholesky decomposition
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i][k] * L[j][k];
                }

                if (i == j)
                {
                    float diag = AtA[i * 9 + i] - sum;
                    L[i][j] = sqrtf(fmaxf(diag, 1e-10f));
                }
                else
                {
                    L[i][j] = (AtA[i * 9 + j] - sum) / L[j][j];
                }
            }
        }

        // Forward substitution: L * y = Atb
        for (int i = 0; i < 9; i++)
        {
            float sum = 0.0f;
            for (int k = 0; k < i; k++)
            {
                sum += L[i][k] * y[k];
            }
            y[i] = (Atb[i] - sum) / L[i][i];
        }

        // Backward substitution: L^T * h = y
        for (int i = 8; i >= 0; i--)
        {
            float sum = 0.0f;
            for (int k = i + 1; k < 9; k++)
            {
                sum += L[k][i] * h[k];
            }
            h[i] = (y[i] - sum) / L[i][i];
        }
    }

    // =====================================================================
    // FP32 -> FP16 conversion
    // =====================================================================
    __global__ void FloatToHalf(
        const float* __restrict__ input,
        __half* __restrict__ output,
        int count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;

        output[idx] = __float2half(input[idx]);
    }

    // =====================================================================
    // FP16 -> FP32 conversion
    // =====================================================================
    __global__ void HalfToFloat(
        const __half* __restrict__ input,
        float* __restrict__ output,
        int count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;

        output[idx] = __half2float(input[idx]);
    }

    // =====================================================================
    // Tensor Core: Normal Equations for Bundle Adjustment
    // Computes JtJ = J^T * J and Jtr = J^T * r
    // =====================================================================
    __global__ void TensorNormalEquations(
        const __half* __restrict__ J,
        const float* __restrict__ r,
        float* __restrict__ JtJ,
        float* __restrict__ Jtr,
        int numResiduals, int numParams)
    {
        // Each thread computes one element of JtJ (upper triangle)
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int i = idx / numParams;
        int j = idx % numParams;

        if (i >= numParams || j >= numParams) return;
        if (j < i) return;  // Upper triangle only

        float sum_jtj = 0.0f;
        float sum_jtr = 0.0f;

        for (int r_idx = 0; r_idx < numResiduals; r_idx++)
        {
            float j_i = __half2float(J[r_idx * numParams + i]);
            float j_j = __half2float(J[r_idx * numParams + j]);
            float residual = r[r_idx];

            sum_jtj += j_i * j_j;

            if (j == 0)
            {
                sum_jtr += j_i * residual;
            }
        }

        JtJ[i * numParams + j] = sum_jtj;
        if (i != j)
            JtJ[j * numParams + i] = sum_jtj;

        if (j == 0)
        {
            Jtr[i] = sum_jtr;
        }
    }
}
