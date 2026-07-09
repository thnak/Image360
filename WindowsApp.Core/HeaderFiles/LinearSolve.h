#pragma once

namespace WindowsApp::Core
{
    // Assembles JtJ = J^T*J and Jtr = J^T*r from a dense row-major Jacobian
    // J (numResiduals x numParams) and residual vector r, applies
    // Levenberg-Marquardt diagonal damping (JtJ[i][i] *= (1+lambda)), and
    // solves JtJ*delta = Jtr via Cholesky - matches
    // WindowsApp.Compute::CudaPipeline::TensorSolveNormalEquations's shipped
    // formulation exactly, including its sign convention: delta is
    // (JtJ)^-1 * Jtr, not its negation (callers subtract delta from
    // params, per Gauss-Newton/LM). Portable host C++, no backend
    // dependency - bundle-adjustment's expensive step is rebuilding the
    // Jacobian via central differences (already host-side); the solve
    // itself is a small dense system, no benefit to a GPU/SIMD backend.
    //
    // Returns false if numResiduals/numParams <= 0 or the system is
    // singular.
    bool SolveNormalEquationsLm(const float* J, const float* r, float* delta,
                                 int numResiduals, int numParams, float lambda);
}
