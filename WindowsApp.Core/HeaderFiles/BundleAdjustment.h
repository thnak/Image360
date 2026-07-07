#pragma once
#include "Types.h"
#include "CudaPipeline.h"
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    struct BaCheckpoint
    {
        int iteration = 0;
        float lambda = 1e-3f;
        std::vector<float> parameters; // 8 DOF per non-reference image (h00..h21, h22 fixed to 1), flattened, image-id order
    };

    // Narrow, purpose-built for exactly {"iteration":N,"lambda":F,"parameters":[f0,f1,...]}
    // - not a general JSON parser (docs/superpowers/plans/2026-07-07-optimize-stage.md's
    // header note on why this doesn't pull in a JSON library for one
    // fixed-shape payload). DeserializeCheckpoint rejects (returns false)
    // anything that doesn't match this exact shape rather than being lenient.
    std::string SerializeCheckpoint(const BaCheckpoint& cp);
    bool DeserializeCheckpoint(const std::string& json, BaCheckpoint& out);

    // Converts one image's 8 flattened BA parameters (h00..h21) back into
    // a Homography (h22 fixed to 1). Exposed publicly (not file-local to
    // BundleAdjustment.cpp) because OptimizeExecutor needs it too, to
    // commit a converged checkpoint's parameters back via
    // ProjectManager::UpdateHomography.
    Homography HomographyFromBaParams(const float* params);

    // One correspondence between two images' feature points, re-derived
    // from Align's persisted feature blobs (not stored in the Task table).
    struct BaCorrespondence
    {
        int imageA = 0;
        int imageB = 0;
        Compute::FeaturePoint pointA;
        Compute::FeaturePoint pointB;
    };

    struct LmStepResult
    {
        BaCheckpoint checkpoint;
        bool converged = false;
    };

    // One LM iteration: builds a numerical (central-difference) Jacobian
    // and residual vector from every correspondence against the current
    // parameter estimate, calls
    // CudaPipeline::TensorSolveNormalEquations, applies the resulting
    // delta if it reduces total reprojection error (classic LM
    // accept/reject + lambda up/down). `nonReferenceImageIds` fixes the
    // parameter-block order (the reference image contributes no
    // parameters - its homography is fixed identity).
    LmStepResult RunOneLmIteration(
        Compute::CudaPipeline& cudaPipeline,
        const std::vector<int>& nonReferenceImageIds,
        const std::vector<BaCorrespondence>& correspondences,
        const BaCheckpoint& current);
}
