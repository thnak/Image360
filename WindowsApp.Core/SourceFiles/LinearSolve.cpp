#include "pch.h"
#include "HeaderFiles/LinearSolve.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace WindowsApp::Core
{
    bool SolveNormalEquationsLm(const float* J, const float* r, float* delta,
                                 int numResiduals, int numParams, float lambda)
    {
        if (!J || !r || !delta || numResiduals <= 0 || numParams <= 0) return false;

        // JtJ = J^T*J, Jtr = J^T*r
        std::vector<float> JtJ(static_cast<size_t>(numParams) * numParams, 0.0f);
        std::vector<float> Jtr(static_cast<size_t>(numParams), 0.0f);

        for (int k = 0; k < numResiduals; ++k)
        {
            const float* row = J + static_cast<size_t>(k) * numParams;
            for (int i = 0; i < numParams; ++i)
            {
                Jtr[i] += row[i] * r[k];
                for (int j = 0; j < numParams; ++j)
                {
                    JtJ[static_cast<size_t>(i) * numParams + j] += row[i] * row[j];
                }
            }
        }

        // LM damping: JtJ[i][i] *= (1 + lambda)
        for (int i = 0; i < numParams; ++i)
        {
            JtJ[static_cast<size_t>(i) * numParams + i] *= (1.0f + lambda);
        }

        // Cholesky decomposition: JtJ = L * L^T
        std::vector<float> L(static_cast<size_t>(numParams) * numParams, 0.0f);
        for (int i = 0; i < numParams; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < j; ++k)
                    sum += L[static_cast<size_t>(i) * numParams + k] * L[static_cast<size_t>(j) * numParams + k];

                if (i == j)
                {
                    float diag = JtJ[static_cast<size_t>(i) * numParams + i] - sum;
                    L[static_cast<size_t>(i) * numParams + j] = std::sqrt((std::max)(diag, 1e-10f));
                }
                else
                {
                    L[static_cast<size_t>(i) * numParams + j] =
                        (JtJ[static_cast<size_t>(i) * numParams + j] - sum) / L[static_cast<size_t>(j) * numParams + j];
                }
            }
        }

        // Forward substitution: L*y = Jtr
        std::vector<float> y(static_cast<size_t>(numParams), 0.0f);
        for (int i = 0; i < numParams; ++i)
        {
            float sum = 0.0f;
            for (int k = 0; k < i; ++k)
                sum += L[static_cast<size_t>(i) * numParams + k] * y[k];
            y[i] = (Jtr[i] - sum) / L[static_cast<size_t>(i) * numParams + i];
        }

        // Backward substitution: L^T*delta = y
        for (int i = numParams - 1; i >= 0; --i)
        {
            float sum = 0.0f;
            for (int k = i + 1; k < numParams; ++k)
                sum += L[static_cast<size_t>(k) * numParams + i] * delta[k];
            delta[i] = (y[i] - sum) / L[static_cast<size_t>(i) * numParams + i];
        }

        return true;
    }
}
