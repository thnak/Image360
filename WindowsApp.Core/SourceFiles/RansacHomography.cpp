#include "pch.h"
#include "HeaderFiles/RansacHomography.h"
#include <random>
#include <cmath>
#include <limits>

namespace WindowsApp::Core
{
    float ReprojectionError(const Homography& h, const Compute::FeaturePoint& src, const Compute::FeaturePoint& dst)
    {
        const auto& m = h.h;
        float denom = m[6] * src.x + m[7] * src.y + m[8];
        if (std::fabs(denom) < 1e-10f)
        {
            return (std::numeric_limits<float>::max)();
        }

        float px = (m[0] * src.x + m[1] * src.y + m[2]) / denom;
        float py = (m[3] * src.x + m[4] * src.y + m[5]) / denom;

        float dx = px - dst.x;
        float dy = py - dst.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    RansacResult RunRansacHomography(
        Compute::CudaPipeline& cudaPipeline,
        const std::vector<std::pair<Compute::FeaturePoint, Compute::FeaturePoint>>& correspondences,
        int iterations, float inlierThresholdPx)
    {
        RansacResult result;

        size_t n = correspondences.size();
        if (n < 4) return result;

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, n - 1);

        for (int iter = 0; iter < iterations; ++iter)
        {
            size_t idx[4];
            int attempts = 0;
            do
            {
                for (int k = 0; k < 4; ++k)
                {
                    idx[k] = dist(rng);
                }
                ++attempts;
            } while (attempts < 16 &&
                     (idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3] ||
                      idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3]));

            float pointPairs[16];
            for (int k = 0; k < 4; ++k)
            {
                const auto& pair = correspondences[idx[k]];
                pointPairs[k * 4 + 0] = pair.first.x;
                pointPairs[k * 4 + 1] = pair.first.y;
                pointPairs[k * 4 + 2] = pair.second.x;
                pointPairs[k * 4 + 3] = pair.second.y;
            }

            Homography candidate;
            Compute::ComputeResult computeResult =
                cudaPipeline.TensorEstimateHomography(pointPairs, candidate.h.data(), 4);
            if (computeResult != Compute::ComputeResult::SUCCESS) continue;

            int inlierCount = 0;
            for (const auto& corr : correspondences)
            {
                if (ReprojectionError(candidate, corr.first, corr.second) < inlierThresholdPx)
                {
                    ++inlierCount;
                }
            }

            if (inlierCount > result.inlierCount)
            {
                result.homography = candidate;
                result.inlierCount = inlierCount;
                result.success = true;
            }
        }

        return result;
    }
}
