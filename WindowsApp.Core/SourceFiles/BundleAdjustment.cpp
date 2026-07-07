#include "pch.h"
#include "HeaderFiles/BundleAdjustment.h"
#include <sstream>
#include <cmath>
#include <cstring>

namespace WindowsApp::Core
{
    Homography HomographyFromBaParams(const float* params)
    {
        Homography h;
        h.h = { params[0], params[1], params[2],
                params[3], params[4], params[5],
                params[6], params[7], 1.0f };
        return h;
    }

    namespace
    {
        Compute::FeaturePoint ApplyHomography(const Homography& h, const Compute::FeaturePoint& p)
        {
            const auto& m = h.h;
            float denom = m[6] * p.x + m[7] * p.y + m[8];
            if (std::fabs(denom) < 1e-10f) denom = (denom >= 0.0f) ? 1e-10f : -1e-10f;

            Compute::FeaturePoint out;
            out.x = (m[0] * p.x + m[1] * p.y + m[2]) / denom;
            out.y = (m[3] * p.x + m[4] * p.y + m[5]) / denom;
            return out;
        }

        int ParamBlockIndex(const std::vector<int>& nonReferenceImageIds, int imageId)
        {
            for (size_t k = 0; k < nonReferenceImageIds.size(); ++k)
            {
                if (nonReferenceImageIds[k] == imageId) return static_cast<int>(k);
            }
            return -1; // reference image - fixed identity, no parameters
        }

        Homography HomographyFor(const std::vector<int>& nonReferenceImageIds, int imageId, const std::vector<float>& params)
        {
            int idx = ParamBlockIndex(nonReferenceImageIds, imageId);
            if (idx < 0) return Homography{}; // identity
            return HomographyFromBaParams(&params[static_cast<size_t>(idx) * 8]);
        }

        void ComputeResiduals(
            const std::vector<int>& nonReferenceImageIds,
            const std::vector<BaCorrespondence>& correspondences,
            const std::vector<float>& params,
            std::vector<float>& outResiduals)
        {
            outResiduals.resize(correspondences.size() * 2);
            for (size_t i = 0; i < correspondences.size(); ++i)
            {
                const auto& corr = correspondences[i];
                Homography hA = HomographyFor(nonReferenceImageIds, corr.imageA, params);
                Homography hB = HomographyFor(nonReferenceImageIds, corr.imageB, params);

                Compute::FeaturePoint worldA = ApplyHomography(hA, corr.pointA);
                Compute::FeaturePoint worldB = ApplyHomography(hB, corr.pointB);

                outResiduals[i * 2] = worldA.x - worldB.x;
                outResiduals[i * 2 + 1] = worldA.y - worldB.y;
            }
        }

        double SumSquares(const std::vector<float>& values)
        {
            double sum = 0.0;
            for (float v : values) sum += static_cast<double>(v) * v;
            return sum;
        }
    }

    std::string SerializeCheckpoint(const BaCheckpoint& cp)
    {
        std::ostringstream oss;
        oss << "{\"iteration\":" << cp.iteration << ",\"lambda\":" << cp.lambda << ",\"parameters\":[";
        for (size_t i = 0; i < cp.parameters.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << cp.parameters[i];
        }
        oss << "]}";
        return oss.str();
    }

    bool DeserializeCheckpoint(const std::string& json, BaCheckpoint& out)
    {
        const std::string iterKey = "\"iteration\":";
        const std::string lambdaKey = "\"lambda\":";
        const std::string paramsKey = "\"parameters\":[";

        size_t iterPos = json.find(iterKey);
        size_t lambdaPos = json.find(lambdaKey);
        size_t paramsPos = json.find(paramsKey);
        if (iterPos == std::string::npos || lambdaPos == std::string::npos || paramsPos == std::string::npos)
        {
            return false;
        }

        try
        {
            out.iteration = std::stoi(json.substr(iterPos + iterKey.size()));
            out.lambda = std::stof(json.substr(lambdaPos + lambdaKey.size()));

            size_t arrayStart = paramsPos + paramsKey.size();
            size_t arrayEnd = json.find(']', arrayStart);
            if (arrayEnd == std::string::npos) return false;

            out.parameters.clear();
            std::string arrayContent = json.substr(arrayStart, arrayEnd - arrayStart);

            size_t pos = 0;
            while (pos < arrayContent.size())
            {
                size_t comma = arrayContent.find(',', pos);
                std::string token = (comma == std::string::npos)
                    ? arrayContent.substr(pos)
                    : arrayContent.substr(pos, comma - pos);

                if (!token.empty())
                {
                    out.parameters.push_back(std::stof(token));
                }

                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        catch (...)
        {
            return false;
        }

        return true;
    }

    LmStepResult RunOneLmIteration(
        Compute::CudaPipeline& cudaPipeline,
        const std::vector<int>& nonReferenceImageIds,
        const std::vector<BaCorrespondence>& correspondences,
        const BaCheckpoint& current)
    {
        LmStepResult result;
        result.checkpoint = current;
        result.converged = false;

        int numParams = static_cast<int>(nonReferenceImageIds.size()) * 8;
        int numResiduals = static_cast<int>(correspondences.size()) * 2;

        if (numParams == 0 || numResiduals == 0)
        {
            result.converged = true;
            return result;
        }

        std::vector<float> baseResiduals;
        ComputeResiduals(nonReferenceImageIds, correspondences, current.parameters, baseResiduals);
        double baseError = SumSquares(baseResiduals);

        // Numerical (central-difference) Jacobian, full dense matrix -
        // each residual only truly depends on its two participating
        // images' 8 parameters each, but building the full dense
        // numParams-wide row keeps this correct and simple; a sparse
        // formulation is follow-up work, not required for a first
        // working implementation.
        constexpr float kEpsilon = 1e-3f;
        std::vector<float> jacobian(static_cast<size_t>(numResiduals) * static_cast<size_t>(numParams), 0.0f);
        std::vector<float> perturbed = current.parameters;

        for (int p = 0; p < numParams; ++p)
        {
            float original = perturbed[p];

            perturbed[p] = original + kEpsilon;
            std::vector<float> residualsPlus;
            ComputeResiduals(nonReferenceImageIds, correspondences, perturbed, residualsPlus);

            perturbed[p] = original - kEpsilon;
            std::vector<float> residualsMinus;
            ComputeResiduals(nonReferenceImageIds, correspondences, perturbed, residualsMinus);

            perturbed[p] = original;

            for (int r = 0; r < numResiduals; ++r)
            {
                jacobian[static_cast<size_t>(r) * numParams + p] =
                    (residualsPlus[r] - residualsMinus[r]) / (2.0f * kEpsilon);
            }
        }

        std::vector<float> delta(numParams, 0.0f);
        Compute::ComputeResult solveResult = cudaPipeline.TensorSolveNormalEquations(
            jacobian.data(), baseResiduals.data(), delta.data(), numResiduals, numParams, current.lambda);

        if (solveResult != Compute::ComputeResult::SUCCESS)
        {
            // Couldn't solve this step - not converged, the caller
            // dispatches another iteration against the same checkpoint.
            return result;
        }

        // NOTE on sign: CudaPipeline::TensorSolveNormalEquations's own
        // doc comment says it solves "JtJ * delta = -Jtr", but its
        // actual (pre-existing, shipped) implementation computes
        // Jtr = J^T*r directly and solves JtJ*delta = Jtr with no
        // negation applied anywhere - i.e. delta is (JtJ)^-1 * Jtr, not
        // its negation. Standard Gauss-Newton/LM minimizes ||r||^2 via
        // params -= (JtJ)^-1 * Jtr, so this code subtracts delta to
        // match the method's actual behavior, not its doc comment.
        std::vector<float> candidateParams = current.parameters;
        for (int p = 0; p < numParams; ++p)
        {
            candidateParams[p] -= delta[p];
        }

        std::vector<float> candidateResiduals;
        ComputeResiduals(nonReferenceImageIds, correspondences, candidateParams, candidateResiduals);
        double candidateError = SumSquares(candidateResiduals);

        BaCheckpoint next = current;
        next.iteration = current.iteration + 1;

        if (candidateError < baseError)
        {
            next.parameters = candidateParams;
            next.lambda = current.lambda / 10.0f;
            double improvement = baseError - candidateError;
            result.converged = improvement < 1e-4;
        }
        else
        {
            next.lambda = current.lambda * 10.0f;
            result.converged = false;
        }

        result.checkpoint = next;
        return result;
    }
}
