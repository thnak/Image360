#pragma once

namespace WindowsApp::Core
{
    // Normalized-DLT homography solve via AtA*h = Atb (Cholesky) - matches
    // WindowsApp.Compute::CudaPipeline::TensorEstimateHomography's shipped
    // (non-SVD) formulation exactly, so RANSAC produces the same result
    // regardless of which compute backend is selected. Portable host C++,
    // no backend dependency - the per-solve system is tiny (9x9 for the
    // usual 4-point RANSAC sample), so there's no benefit to routing this
    // through a GPU/SIMD backend.
    //
    // pointPairs: [x0,y0,x0',y0', x1,y1,x1',y1', ...] (4 floats per pair,
    // numPairs >= 4). outHomography: row-major 3x3 (9 floats), normalized
    // so outHomography[8] == 1.
    // Returns false if numPairs < 4 or the system is singular.
    bool SolveHomographyDlt(const float* pointPairs, int numPairs, float outHomography[9]);

    // Closed-form 3x3 matrix inverse (adjugate/cofactor, row-major).
    // Align/BundleAdjustment produce a homography mapping an image's own
    // LOCAL pixel coordinates into the shared world/canvas frame
    // (confirmed via WindowsApp.Tests/BundleAdjustmentTests.cpp), but
    // Compute::IComputeBackend::WarpPerspective's own `homography`
    // parameter is documented (WarpPerspectiveKernels.h) to be the
    // OPPOSITE direction - DESTINATION -> SOURCE, for backward-mapping
    // sampling - so callers warping an image into a world-space chunk
    // must invert the stored homography first. Returns false (leaves
    // outInverse untouched) if the matrix is singular/near-singular.
    bool InvertHomography3x3(const float h[9], float outInverse[9]);
}
