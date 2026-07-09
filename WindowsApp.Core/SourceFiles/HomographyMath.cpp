#include "pch.h"
#include "HeaderFiles/HomographyMath.h"

#include <algorithm>
#include <cmath>

namespace WindowsApp::Core
{
    bool SolveHomographyDlt(const float* pointPairs, int numPairs, float outHomography[9])
    {
        if (!pointPairs || !outHomography || numPairs < 4) return false;

        // AtA (9x9) and Atb (9), built exactly as
        // WindowsApp.Compute::Kernels::TensorBuildAtA / the CPU Atb loop in
        // CudaPipeline::TensorEstimateHomography do: two DLT rows per
        // correspondence, Atb formed by dotting each row against its own
        // target coordinate (not a null-space SVD solve).
        float AtA[9][9] = {};
        float Atb[9] = {};

        for (int p = 0; p < numPairs; ++p)
        {
            float x = pointPairs[p * 4 + 0];
            float y = pointPairs[p * 4 + 1];
            float xp = pointPairs[p * 4 + 2];
            float yp = pointPairs[p * 4 + 3];

            float a0[9] = { -x, -y, -1.0f, 0, 0, 0, x * xp, y * xp, xp };
            float a1[9] = { 0, 0, 0, -x, -y, -1.0f, x * yp, y * yp, yp };

            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    AtA[i][j] += a0[i] * a0[j] + a1[i] * a1[j];
                }
                Atb[i] += a0[i] * xp + a1[i] * yp;
            }
        }

        // Cholesky decomposition: AtA = L * L^T
        float L[9][9] = {};
        for (int i = 0; i < 9; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < j; ++k)
                    sum += L[i][k] * L[j][k];

                if (i == j)
                {
                    float diag = AtA[i][i] - sum;
                    L[i][j] = std::sqrt((std::max)(diag, 1e-10f));
                }
                else
                {
                    L[i][j] = (AtA[i][j] - sum) / L[j][j];
                }
            }
        }

        // Forward substitution: L*y = Atb
        float y[9] = {};
        for (int i = 0; i < 9; ++i)
        {
            float sum = 0.0f;
            for (int k = 0; k < i; ++k)
                sum += L[i][k] * y[k];
            y[i] = (Atb[i] - sum) / L[i][i];
        }

        // Backward substitution: L^T*h = y
        float h[9] = {};
        for (int i = 8; i >= 0; --i)
        {
            float sum = 0.0f;
            for (int k = i + 1; k < 9; ++k)
                sum += L[k][i] * h[k];
            h[i] = (y[i] - sum) / L[i][i];
        }

        for (int i = 0; i < 9; ++i) outHomography[i] = h[i];

        if (std::fabs(outHomography[8]) > 1e-10f)
        {
            float inv = 1.0f / outHomography[8];
            for (int i = 0; i < 9; ++i) outHomography[i] *= inv;
        }

        return true;
    }

    bool InvertHomography3x3(const float h[9], float outInverse[9])
    {
        if (!h || !outInverse) return false;

        float a = h[0], b = h[1], c = h[2];
        float d = h[3], e = h[4], f = h[5];
        float g = h[6], i2 = h[7], j = h[8]; // avoid shadowing std::i/h names

        float det = a * (e * j - f * i2) - b * (d * j - f * g) + c * (d * i2 - e * g);
        if (std::fabs(det) < 1e-12f) return false;

        float invDet = 1.0f / det;
        outInverse[0] = (e * j - f * i2) * invDet;
        outInverse[1] = (c * i2 - b * j) * invDet;
        outInverse[2] = (b * f - c * e) * invDet;
        outInverse[3] = (f * g - d * j) * invDet;
        outInverse[4] = (a * j - c * g) * invDet;
        outInverse[5] = (c * d - a * f) * invDet;
        outInverse[6] = (d * i2 - e * g) * invDet;
        outInverse[7] = (b * g - a * i2) * invDet;
        outInverse[8] = (a * e - b * d) * invDet;
        return true;
    }
}
