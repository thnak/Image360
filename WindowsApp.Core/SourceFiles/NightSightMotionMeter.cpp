#include "pch.h"
#include "HeaderFiles/NightSightMotionMeter.h"
#include "HeaderFiles/BurstCommon.h"

#include <algorithm>
#include <cmath>

using namespace WindowsApp::Compute;

namespace WindowsApp::Core::Kernels
{
    namespace
    {
        float MeanMagnitude(const std::vector<TileOffsetF>& offsets)
        {
            if (offsets.empty()) return 0.0f;
            double sum = 0.0;
            for (const auto& o : offsets)
                sum += std::sqrt(static_cast<double>(o.dx) * o.dx + static_cast<double>(o.dy) * o.dy);
            return static_cast<float>(sum / offsets.size());
        }
    }

    MotionMeteringResult MeterMotion(
        const std::vector<std::vector<TileOffsetF>>& perFrameOffsets, float baseNoiseVariance)
    {
        MotionMeteringResult result;
        size_t numFrames = perFrameOffsets.size();
        result.noiseVariance = baseNoiseVariance;
        if (numFrames == 0) return result;

        std::vector<float> frameMotion(numFrames);
        for (size_t i = 0; i < numFrames; ++i) frameMotion[i] = MeanMagnitude(perFrameOffsets[i]);

        // Median of per-frame motion - robust to one/two severe outliers
        // skewing a mean the way this exclusion decision needs to detect
        // them.
        std::vector<float> sorted = frameMotion;
        std::sort(sorted.begin(), sorted.end());
        float median = sorted[sorted.size() / 2];

        std::vector<bool> keep(numFrames, true);
        for (size_t i = 0; i < numFrames; ++i)
        {
            bool relativeOutlier = median > 0.0f
                && frameMotion[i] > median * kNightSightMotionExclusionRelativeFactor;
            bool absoluteOutlier = frameMotion[i] > kNightSightMotionExclusionAbsolutePx;
            keep[i] = !(relativeOutlier || absoluteOutlier);
        }

        int keptCount = 0;
        for (bool k : keep) if (k) ++keptCount;

        if (keptCount < kNightSightMinKeptNonReferenceFrames
            && static_cast<int>(numFrames) >= kNightSightMinKeptNonReferenceFrames)
        {
            // Excluding this many would leave too few frames to merge
            // meaningfully - fall back to keeping the least-motion frames
            // up to the floor rather than an empty/near-empty merge (see
            // this phase's plan doc's self-review).
            std::vector<size_t> byMotion(numFrames);
            for (size_t i = 0; i < numFrames; ++i) byMotion[i] = i;
            std::sort(byMotion.begin(), byMotion.end(),
                      [&](size_t a, size_t b) { return frameMotion[a] < frameMotion[b]; });

            keep.assign(numFrames, false);
            for (int i = 0; i < kNightSightMinKeptNonReferenceFrames; ++i) keep[byMotion[i]] = true;
        }

        // Aggregate motion over the frames actually kept - what the merge
        // will really see.
        double keptSum = 0.0;
        int keptN = 0;
        for (size_t i = 0; i < numFrames; ++i)
        {
            if (!keep[i]) continue;
            keptSum += frameMotion[i];
            ++keptN;
        }
        float meanKeptMotion = keptN > 0 ? static_cast<float>(keptSum / keptN) : 0.0f;

        // Below kNightSightMotionReferencePx (tripod-like, confident
        // alignment) -> larger noiseVariance (more permissive, more
        // denoising); above it (handheld-like, less confident alignment)
        // -> smaller noiseVariance (tighter agreement, more ghost-safe) -
        // see this phase's plan doc's Architecture section 1, clamped so
        // a pathological input can't zero out or blow up the parameter.
        float scale = 1.0f + kNightSightNoiseVarianceGain * (kNightSightMotionReferencePx - meanKeptMotion);
        scale = (std::max)(kNightSightNoiseVarianceMinScale, (std::min)(kNightSightNoiseVarianceMaxScale, scale));

        result.keepFrame = std::move(keep);
        result.noiseVariance = baseNoiseVariance * scale;
        return result;
    }
}
