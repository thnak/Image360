#include "pch.h"
#include "HeaderFiles/SeamBlendKernels.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace WindowsApp::Core::Kernels::SeamBlend
{
    namespace
    {
        inline size_t Idx(int x, int y, int width) { return static_cast<size_t>(y) * width + x; }

        // Separable 5-tap binomial blur ([1,4,6,4,1]/16), replicate border.
        void Blur5Tap(const std::vector<float>& src, int w, int h, int channels, std::vector<float>& dst)
        {
            std::vector<float> tmp(src.size(), 0.0f);
            dst.assign(src.size(), 0.0f);
            const float k0 = 1.0f / 16.0f, k1 = 4.0f / 16.0f, k2 = 6.0f / 16.0f;

            auto clampX = [w](int x) { return (std::max)(0, (std::min)(w - 1, x)); };
            auto clampY = [h](int y) { return (std::max)(0, (std::min)(h - 1, y)); };

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    size_t base = (static_cast<size_t>(y) * w + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        tmp[base + c] =
                            k0 * src[(static_cast<size_t>(y) * w + clampX(x - 2)) * channels + c] +
                            k1 * src[(static_cast<size_t>(y) * w + clampX(x - 1)) * channels + c] +
                            k2 * src[(static_cast<size_t>(y) * w + clampX(x)) * channels + c] +
                            k1 * src[(static_cast<size_t>(y) * w + clampX(x + 1)) * channels + c] +
                            k0 * src[(static_cast<size_t>(y) * w + clampX(x + 2)) * channels + c];
                    }
                }
            }
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    size_t base = (static_cast<size_t>(y) * w + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        dst[base + c] =
                            k0 * tmp[(static_cast<size_t>(clampY(y - 2)) * w + x) * channels + c] +
                            k1 * tmp[(static_cast<size_t>(clampY(y - 1)) * w + x) * channels + c] +
                            k2 * tmp[(static_cast<size_t>(clampY(y)) * w + x) * channels + c] +
                            k1 * tmp[(static_cast<size_t>(clampY(y + 1)) * w + x) * channels + c] +
                            k0 * tmp[(static_cast<size_t>(clampY(y + 2)) * w + x) * channels + c];
                    }
                }
            }
        }

        // Blur then subsample by 2 (ceiling division, matching the
        // (w+1)/2 convention CudaPipeline::MultiBandBlend already uses).
        void Downsample2x(const std::vector<float>& src, int w, int h, int channels,
                           std::vector<float>& dst, int& outW, int& outH)
        {
            std::vector<float> blurred;
            Blur5Tap(src, w, h, channels, blurred);
            outW = (w + 1) / 2;
            outH = (h + 1) / 2;
            dst.assign(static_cast<size_t>(outW) * outH * channels, 0.0f);
            for (int y = 0; y < outH; ++y)
            {
                int sy = (std::min)(y * 2, h - 1);
                for (int x = 0; x < outW; ++x)
                {
                    int sx = (std::min)(x * 2, w - 1);
                    size_t dstBase = (static_cast<size_t>(y) * outW + x) * channels;
                    size_t srcBase = (static_cast<size_t>(sy) * w + sx) * channels;
                    for (int c = 0; c < channels; ++c) dst[dstBase + c] = blurred[srcBase + c];
                }
            }
        }

        // Bilinear resize from (w,h) up to (outW,outH). Simpler and more
        // robust than zero-insertion+blur for chunk dims that aren't
        // powers of 2, and visually equivalent for a smooth low-frequency
        // upsample.
        void Upsample(const std::vector<float>& src, int w, int h, int channels,
                      int outW, int outH, std::vector<float>& dst)
        {
            dst.assign(static_cast<size_t>(outW) * outH * channels, 0.0f);
            float sx = (w > 1) ? static_cast<float>(w - 1) / static_cast<float>((std::max)(1, outW - 1)) : 0.0f;
            float sy = (h > 1) ? static_cast<float>(h - 1) / static_cast<float>((std::max)(1, outH - 1)) : 0.0f;
            for (int y = 0; y < outH; ++y)
            {
                float fy = sy * y;
                int y0 = (std::min)(static_cast<int>(fy), h - 1);
                int y1 = (std::min)(y0 + 1, h - 1);
                float wy = fy - y0;
                for (int x = 0; x < outW; ++x)
                {
                    float fx = sx * x;
                    int x0 = (std::min)(static_cast<int>(fx), w - 1);
                    int x1 = (std::min)(x0 + 1, w - 1);
                    float wx = fx - x0;
                    size_t dstBase = (static_cast<size_t>(y) * outW + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        float v00 = src[(static_cast<size_t>(y0) * w + x0) * channels + c];
                        float v10 = src[(static_cast<size_t>(y0) * w + x1) * channels + c];
                        float v01 = src[(static_cast<size_t>(y1) * w + x0) * channels + c];
                        float v11 = src[(static_cast<size_t>(y1) * w + x1) * channels + c];
                        dst[dstBase + c] = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v10) +
                                            wy * ((1.0f - wx) * v01 + wx * v11);
                    }
                }
            }
        }

        struct PyramidLevel
        {
            std::vector<float> data;
            int w = 0, h = 0;
        };

        // Plain Gaussian pyramid of a hole-free float field - used for the
        // ownership/weight masks (well-defined 0/1 everywhere, no holes).
        void BuildGaussianPyramid(const std::vector<float>& base, int w, int h, int channels,
                                   int numBands, std::vector<PyramidLevel>& outPyramid)
        {
            outPyramid.resize(numBands);
            outPyramid[0].data = base;
            outPyramid[0].w = w;
            outPyramid[0].h = h;
            for (int level = 1; level < numBands; ++level)
            {
                int outW = 0, outH = 0;
                Downsample2x(outPyramid[level - 1].data, outPyramid[level - 1].w, outPyramid[level - 1].h,
                             channels, outPyramid[level].data, outW, outH);
                outPyramid[level].w = outW;
                outPyramid[level].h = outH;
            }
        }

        // Hole-filled Gaussian pyramid of a contributor's warped RGB image
        // via normalized convolution against its own raw coverage mask:
        // numerator = Blur+Downsample(image*coverage), denominator =
        // Blur+Downsample(coverage), gaussian = numerator/denominator.
        // Without this, blurring right up to a contributor's own true
        // (0,0,0) edge would bleed black into the pyramid and show up as
        // dark fringing wherever that contributor wins ownership near its
        // own boundary.
        void BuildHoleFilledGaussianPyramid(const unsigned short* image, const std::vector<float>& coverage,
                                             int w, int h, int numBands, std::vector<PyramidLevel>& outPyramid)
        {
            size_t count = static_cast<size_t>(w) * h;
            std::vector<float> numerator(count * 3), denominator(count);
            for (size_t p = 0; p < count; ++p)
            {
                float m = coverage[p];
                denominator[p] = m;
                for (int c = 0; c < 3; ++c) numerator[p * 3 + c] = static_cast<float>(image[p * 3 + c]) * m;
            }

            std::vector<PyramidLevel> numPyr, denPyr;
            BuildGaussianPyramid(numerator, w, h, 3, numBands, numPyr);
            BuildGaussianPyramid(denominator, w, h, 1, numBands, denPyr);

            constexpr float kEps = 1e-6f;
            outPyramid.resize(numBands);
            for (int level = 0; level < numBands; ++level)
            {
                int lw = numPyr[level].w, lh = numPyr[level].h;
                outPyramid[level].w = lw;
                outPyramid[level].h = lh;
                size_t levelCount = static_cast<size_t>(lw) * lh;
                outPyramid[level].data.assign(levelCount * 3, 0.0f);
                for (size_t p = 0; p < levelCount; ++p)
                {
                    float den = denPyr[level].data[p];
                    float inv = (den > kEps) ? (1.0f / den) : 0.0f;
                    for (int c = 0; c < 3; ++c) outPyramid[level].data[p * 3 + c] = numPyr[level].data[p * 3 + c] * inv;
                }
            }
        }

        void BuildLaplacianFromGaussian(const std::vector<PyramidLevel>& gaussian, int channels,
                                         std::vector<PyramidLevel>& outLaplacian)
        {
            int numBands = static_cast<int>(gaussian.size());
            outLaplacian.resize(numBands);
            for (int level = 0; level < numBands - 1; ++level)
            {
                std::vector<float> upsampled;
                Upsample(gaussian[level + 1].data, gaussian[level + 1].w, gaussian[level + 1].h, channels,
                         gaussian[level].w, gaussian[level].h, upsampled);
                size_t n = static_cast<size_t>(gaussian[level].w) * gaussian[level].h * channels;
                outLaplacian[level].data.assign(n, 0.0f);
                outLaplacian[level].w = gaussian[level].w;
                outLaplacian[level].h = gaussian[level].h;
                for (size_t p = 0; p < n; ++p) outLaplacian[level].data[p] = gaussian[level].data[p] - upsampled[p];
            }
            outLaplacian[numBands - 1] = gaussian[numBands - 1]; // top band: pure low-freq, no subtraction
        }
    }

    namespace Detail
    {
        void ComputeCoverageDistance(const std::vector<unsigned char>& covered,
                                      int width, int height, std::vector<float>& outDistance)
        {
            constexpr float kDiag = 1.41421356f;
            size_t count = static_cast<size_t>(width) * height;
            const float kInf = static_cast<float>(width + height);

            outDistance.assign(count, 0.0f);
            for (size_t p = 0; p < count; ++p) outDistance[p] = covered[p] ? kInf : 0.0f;

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    size_t p = Idx(x, y, width);
                    float best = outDistance[p];
                    if (x > 0) best = (std::min)(best, outDistance[Idx(x - 1, y, width)] + 1.0f);
                    if (y > 0) best = (std::min)(best, outDistance[Idx(x, y - 1, width)] + 1.0f);
                    if (x > 0 && y > 0) best = (std::min)(best, outDistance[Idx(x - 1, y - 1, width)] + kDiag);
                    if (x < width - 1 && y > 0) best = (std::min)(best, outDistance[Idx(x + 1, y - 1, width)] + kDiag);
                    outDistance[p] = best;
                }
            }
            for (int y = height - 1; y >= 0; --y)
            {
                for (int x = width - 1; x >= 0; --x)
                {
                    size_t p = Idx(x, y, width);
                    float best = outDistance[p];
                    if (x < width - 1) best = (std::min)(best, outDistance[Idx(x + 1, y, width)] + 1.0f);
                    if (y < height - 1) best = (std::min)(best, outDistance[Idx(x, y + 1, width)] + 1.0f);
                    if (x < width - 1 && y < height - 1) best = (std::min)(best, outDistance[Idx(x + 1, y + 1, width)] + kDiag);
                    if (x > 0 && y < height - 1) best = (std::min)(best, outDistance[Idx(x - 1, y + 1, width)] + kDiag);
                    outDistance[p] = best;
                }
            }
        }

        void ComputeOwnership(const std::vector<std::vector<float>>& distances,
                               int width, int height, std::vector<int>& outOwner)
        {
            size_t count = static_cast<size_t>(width) * height;
            int numContributors = static_cast<int>(distances.size());
            outOwner.assign(count, -1);
            for (size_t p = 0; p < count; ++p)
            {
                int bestOwner = -1;
                float bestDist = 0.0f; // must be > 0 ("covered") to count
                for (int i = 0; i < numContributors; ++i)
                {
                    float d = distances[i][p];
                    if (d > bestDist) { bestDist = d; bestOwner = i; }
                }
                outOwner[p] = bestOwner;
            }
        }

        BBox ComputeCoverageBBox(const std::vector<unsigned char>& covered, int width, int height)
        {
            BBox box;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    if (!covered[Idx(x, y, width)]) continue;
                    if (!box.valid) { box.x0 = box.x1 = x; box.y0 = box.y1 = y; box.valid = true; }
                    else
                    {
                        box.x0 = (std::min)(box.x0, x); box.x1 = (std::max)(box.x1, x);
                        box.y0 = (std::min)(box.y0, y); box.y1 = (std::max)(box.y1, y);
                    }
                }
            }
            return box;
        }

        bool IntersectBBox(const BBox& a, const BBox& b, BBox& outOverlap)
        {
            if (!a.valid || !b.valid) { outOverlap = BBox{}; return false; }
            outOverlap.x0 = (std::max)(a.x0, b.x0);
            outOverlap.y0 = (std::max)(a.y0, b.y0);
            outOverlap.x1 = (std::min)(a.x1, b.x1);
            outOverlap.y1 = (std::min)(a.y1, b.y1);
            outOverlap.valid = (outOverlap.x0 <= outOverlap.x1) && (outOverlap.y0 <= outOverlap.y1);
            return outOverlap.valid;
        }

        void RefineSeamDP(const unsigned short* imgI, const unsigned short* imgJ,
                           const std::vector<float>& distanceI, const std::vector<float>& distanceJ,
                           int width, int /*height*/, int contributorI, int contributorJ,
                           const BBox& iBBox, const BBox& jBBox, const BBox& overlap,
                           std::vector<int>& owner)
        {
            if (!overlap.valid) return;

            int bboxW = overlap.x1 - overlap.x0 + 1;
            int bboxH = overlap.y1 - overlap.y0 + 1;
            bool rowScan = bboxW >= bboxH; // wider-than-tall -> boundary more likely vertical-ish

            float centerI = rowScan ? (iBBox.x0 + iBBox.x1) * 0.5f : (iBBox.y0 + iBBox.y1) * 0.5f;
            float centerJ = rowScan ? (jBBox.x0 + jBBox.x1) * 0.5f : (jBBox.y0 + jBBox.y1) * 0.5f;
            bool iIsLowSide = centerI <= centerJ;

            constexpr float kSmoothnessWeight = 2.0f;
            // How far the seam can be nudged from its pre-existing
            // (pure distance-transform) crossover point. Deliberately
            // local - the candidate range is NOT the whole pairwise
            // overlap: a strong disagreement far from the true boundary
            // (e.g. two unrelated objects, each deep inside its own
            // contributor's dominant territory) must not cause the DP to
            // reroute across unrelated ground just because it's "cheaper"
            // there - it should only ever compete for pixels genuinely
            // near the existing boundary.
            constexpr int kSeamSearchRadius = 24;

            int numLines = rowScan ? bboxH : bboxW;
            int lineLength = rowScan ? bboxW : bboxH;

            std::vector<int> lo(numLines, 1), hi(numLines, 0); // hi < lo => no contest on this line
            for (int line = 0; line < numLines; ++line)
            {
                int fixedCoord = rowScan ? (overlap.y0 + line) : (overlap.x0 + line);

                // Find where the pre-existing per-pixel ownership
                // (possibly already touched by an earlier pair's
                // refinement) transitions between contributorI and
                // contributorJ along this line, and center the DP search
                // band there.
                int crossoverT = -1;
                int firstOwnerIJ = -1;
                for (int t = 0; t < lineLength; ++t)
                {
                    int x = rowScan ? (overlap.x0 + t) : fixedCoord;
                    int y = rowScan ? fixedCoord : (overlap.y0 + t);
                    int o = owner[Idx(x, y, width)];
                    if (o != contributorI && o != contributorJ) continue;
                    if (firstOwnerIJ < 0) { firstOwnerIJ = o; continue; }
                    if (o != firstOwnerIJ) { crossoverT = t; break; }
                }
                if (crossoverT < 0) continue; // no I/J boundary on this line - nothing to refine

                int bandLo = (std::max)(0, crossoverT - kSeamSearchRadius);
                int bandHi = (std::min)(lineLength - 1, crossoverT + kSeamSearchRadius);
                for (int t = bandLo; t <= bandHi; ++t)
                {
                    int x = rowScan ? (overlap.x0 + t) : fixedCoord;
                    int y = rowScan ? fixedCoord : (overlap.y0 + t);
                    size_t p = Idx(x, y, width);
                    if (distanceI[p] > 0.0f && distanceJ[p] > 0.0f)
                    {
                        if (hi[line] < lo[line]) { lo[line] = t; hi[line] = t; }
                        else { lo[line] = (std::min)(lo[line], t); hi[line] = (std::max)(hi[line], t); }
                    }
                }
            }

            std::vector<std::vector<float>> dp(numLines);
            std::vector<std::vector<int>> back(numLines);
            std::vector<int> prevContestedLine(numLines, -1);
            int lastContested = -1;

            for (int line = 0; line < numLines; ++line)
            {
                if (hi[line] < lo[line]) continue;
                int lineWidth = hi[line] - lo[line] + 1;
                dp[line].assign(lineWidth, 0.0f);
                back[line].assign(lineWidth, -1);

                int fixedCoord = rowScan ? (overlap.y0 + line) : (overlap.x0 + line);
                for (int k = 0; k < lineWidth; ++k)
                {
                    int t = lo[line] + k;
                    int x = rowScan ? (overlap.x0 + t) : fixedCoord;
                    int y = rowScan ? fixedCoord : (overlap.y0 + t);
                    size_t p = Idx(x, y, width);
                    float localCost = 0.0f;
                    for (int c = 0; c < 3; ++c)
                        localCost += std::fabs(static_cast<float>(imgI[p * 3 + c]) - static_cast<float>(imgJ[p * 3 + c]));

                    if (lastContested < 0)
                    {
                        dp[line][k] = localCost;
                    }
                    else
                    {
                        float best = (std::numeric_limits<float>::max)();
                        int bestPrev = -1;
                        int prevLo = lo[lastContested];
                        int prevWidth = static_cast<int>(dp[lastContested].size());
                        for (int pk = 0; pk < prevWidth; ++pk)
                        {
                            int prevT = prevLo + pk;
                            float cost = dp[lastContested][pk] + kSmoothnessWeight * std::fabs(static_cast<float>(t - prevT));
                            if (cost < best) { best = cost; bestPrev = pk; }
                        }
                        dp[line][k] = localCost + best;
                        back[line][k] = bestPrev;
                    }
                }
                prevContestedLine[line] = lastContested;
                lastContested = line;
            }

            if (lastContested < 0) return; // no contested pixels anywhere in this overlap

            std::vector<int> chosenT(numLines, -1);
            int line = lastContested;
            int bestK = 0;
            {
                float best = (std::numeric_limits<float>::max)();
                for (size_t k = 0; k < dp[line].size(); ++k)
                {
                    if (dp[line][k] < best) { best = dp[line][k]; bestK = static_cast<int>(k); }
                }
            }
            while (line >= 0)
            {
                chosenT[line] = lo[line] + bestK;
                int prevLine = prevContestedLine[line];
                int prevK = back[line][bestK];
                line = prevLine;
                bestK = prevK;
            }

            for (int ln = 0; ln < numLines; ++ln)
            {
                if (chosenT[ln] < 0) continue;
                int fixedCoord = rowScan ? (overlap.y0 + ln) : (overlap.x0 + ln);
                for (int t = lo[ln]; t <= hi[ln]; ++t)
                {
                    int x = rowScan ? (overlap.x0 + t) : fixedCoord;
                    int y = rowScan ? fixedCoord : (overlap.y0 + t);
                    size_t p = Idx(x, y, width);
                    if (owner[p] != contributorI && owner[p] != contributorJ) continue;

                    bool onLowSide = (t < chosenT[ln]);
                    owner[p] = (onLowSide == iIsLowSide) ? contributorI : contributorJ;
                }
            }
        }

        void MultiBandCompose(const std::vector<const unsigned short*>& warpedBuffers,
                               const std::vector<std::vector<float>>& distances,
                               const std::vector<int>& owner,
                               int width, int height, int numBands,
                               std::vector<unsigned short>& outResult)
        {
            size_t count = static_cast<size_t>(width) * height;
            outResult.assign(count * 3, 0);
            int numContributors = static_cast<int>(warpedBuffers.size());

            int maxBands = 1;
            { int w = width, h = height; while (w > 1 && h > 1 && maxBands < numBands) { w = (w + 1) / 2; h = (h + 1) / 2; ++maxBands; } }
            numBands = (std::min)(numBands, (std::max)(1, maxBands));

            std::vector<std::vector<float>> weightMask(numContributors);
            std::vector<bool> hasAnyOwnedPixel(numContributors, false);
            for (int i = 0; i < numContributors; ++i) weightMask[i].assign(count, 0.0f);
            for (size_t p = 0; p < count; ++p)
            {
                int o = owner[p];
                if (o >= 0) { weightMask[o][p] = 1.0f; hasAnyOwnedPixel[o] = true; }
            }

            bool blendedInit = false;
            std::vector<int> bandW(numBands), bandH(numBands);
            std::vector<std::vector<float>> blendedBand(numBands);

            for (int i = 0; i < numContributors; ++i)
            {
                if (!hasAnyOwnedPixel[i]) continue;

                std::vector<PyramidLevel> weightPyr;
                BuildGaussianPyramid(weightMask[i], width, height, 1, numBands, weightPyr);

                std::vector<float> coverageMask(count);
                for (size_t p = 0; p < count; ++p) coverageMask[p] = (distances[i][p] > 0.0f) ? 1.0f : 0.0f;

                std::vector<PyramidLevel> imageGaussian;
                BuildHoleFilledGaussianPyramid(warpedBuffers[i], coverageMask, width, height, numBands, imageGaussian);

                std::vector<PyramidLevel> imageLaplacian;
                BuildLaplacianFromGaussian(imageGaussian, 3, imageLaplacian);

                if (!blendedInit)
                {
                    for (int level = 0; level < numBands; ++level)
                    {
                        bandW[level] = imageLaplacian[level].w;
                        bandH[level] = imageLaplacian[level].h;
                        blendedBand[level].assign(static_cast<size_t>(bandW[level]) * bandH[level] * 3, 0.0f);
                    }
                    blendedInit = true;
                }

                for (int level = 0; level < numBands; ++level)
                {
                    size_t levelCount = static_cast<size_t>(bandW[level]) * bandH[level];
                    for (size_t p = 0; p < levelCount; ++p)
                    {
                        float w = weightPyr[level].data[p];
                        for (int c = 0; c < 3; ++c) blendedBand[level][p * 3 + c] += w * imageLaplacian[level].data[p * 3 + c];
                    }
                }
            }

            if (!blendedInit) return; // no contributor owns anything - shouldn't normally happen

            std::vector<float> recon = blendedBand[numBands - 1];
            int reconW = bandW[numBands - 1], reconH = bandH[numBands - 1];
            for (int level = numBands - 2; level >= 0; --level)
            {
                std::vector<float> upsampled;
                Upsample(recon, reconW, reconH, 3, bandW[level], bandH[level], upsampled);
                size_t n = static_cast<size_t>(bandW[level]) * bandH[level] * 3;
                recon.assign(n, 0.0f);
                for (size_t p = 0; p < n; ++p) recon[p] = blendedBand[level][p] + upsampled[p];
                reconW = bandW[level];
                reconH = bandH[level];
            }

            for (size_t p = 0; p < count; ++p)
            {
                for (int c = 0; c < 3; ++c)
                {
                    float v = recon[p * 3 + c];
                    v = (std::max)(0.0f, (std::min)(65535.0f, v));
                    outResult[p * 3 + c] = static_cast<unsigned short>(v + 0.5f);
                }
            }

            // Pyramid blur can bleed a sliver of weight/extrapolated color
            // into pixels no contributor actually covers (owner == -1) -
            // force those back to (0,0,0), matching the existing "no data"
            // convention exactly regardless of any such bleed.
            for (size_t p = 0; p < count; ++p)
            {
                if (owner[p] >= 0) continue;
                outResult[p * 3] = outResult[p * 3 + 1] = outResult[p * 3 + 2] = 0;
            }
        }
    }

    void BlendChunkContributors(const std::vector<const unsigned short*>& warpedBuffers,
                                 int width, int height, std::vector<unsigned short>& outResult)
    {
        size_t count = static_cast<size_t>(width) * height;
        outResult.assign(count * 3, 0);
        int numContributors = static_cast<int>(warpedBuffers.size());
        if (numContributors == 0) return;

        std::vector<std::vector<unsigned char>> covered(numContributors);
        std::vector<std::vector<float>> distances(numContributors);
        std::vector<Detail::BBox> bboxes(numContributors);
        for (int i = 0; i < numContributors; ++i)
        {
            covered[i].assign(count, 0);
            const unsigned short* buf = warpedBuffers[i];
            for (size_t p = 0; p < count; ++p)
                covered[i][p] = (buf[p * 3] != 0 || buf[p * 3 + 1] != 0 || buf[p * 3 + 2] != 0) ? 1 : 0;
            Detail::ComputeCoverageDistance(covered[i], width, height, distances[i]);
            bboxes[i] = Detail::ComputeCoverageBBox(covered[i], width, height);
        }

        std::vector<int> owner;
        Detail::ComputeOwnership(distances, width, height, owner);

        for (int i = 0; i < numContributors; ++i)
        {
            if (!bboxes[i].valid) continue;
            for (int j = i + 1; j < numContributors; ++j)
            {
                if (!bboxes[j].valid) continue;
                Detail::BBox overlap;
                if (!Detail::IntersectBBox(bboxes[i], bboxes[j], overlap)) continue;
                Detail::RefineSeamDP(warpedBuffers[i], warpedBuffers[j], distances[i], distances[j],
                                     width, height, i, j, bboxes[i], bboxes[j], overlap, owner);
            }
        }

        Detail::MultiBandCompose(warpedBuffers, distances, owner, width, height, kDefaultNumBands, outResult);
    }
}
