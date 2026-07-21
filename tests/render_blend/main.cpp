// Direct unit tests for Kernels::SeamBlend - the fix for the reported
// "ghosting" bug (RenderExecutor's old per-pixel median/average blended
// unrelated content wherever contributors disagreed, e.g. a nearby object
// at a slightly different position in each warp due to real camera
// parallax, producing a translucent double-image). Synthetic in-memory
// warped buffers, no image fixtures needed - see
// WindowsApp.Core/HeaderFiles/SeamBlendKernels.h for the algorithm.

#include "HeaderFiles/SeamBlendKernels.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace WindowsApp::Core::Kernels;

namespace
{
    int g_failures = 0;

    void Check(bool condition, const char* what)
    {
        if (!condition)
        {
            std::cerr << "FAIL: " << what << std::endl;
            ++g_failures;
        }
        else
        {
            std::cout << "OK: " << what << std::endl;
        }
    }

    void SetPixel(std::vector<unsigned short>& buf, int width, int x, int y, unsigned short r, unsigned short g, unsigned short b)
    {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b;
    }

    void GetPixel(const std::vector<unsigned short>& buf, int width, int x, int y, unsigned short out[3])
    {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        out[0] = buf[idx]; out[1] = buf[idx + 1]; out[2] = buf[idx + 2];
    }

    // Manhattan-ish distance between an actual output pixel and a
    // synthetic source pixel, summed over channels.
    int PixelDiff(const unsigned short a[3], const unsigned short b[3])
    {
        int d = 0;
        for (int c = 0; c < 3; ++c) d += std::abs(static_cast<int>(a[c]) - static_cast<int>(b[c]));
        return d;
    }

    // Background: a smooth per-column gradient shared by every synthetic
    // buffer below (so "the two images agree on the background", matching
    // a correctly-aligned real stitch away from any parallax object).
    void FillBackground(std::vector<unsigned short>& buf, int width, int height, int x0, int x1)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = x0; x < x1; ++x)
            {
                unsigned short v = static_cast<unsigned short>(10000 + x * 100);
                SetPixel(buf, width, x, y, v, 20000, 30000);
            }
        }
    }

    void FillRect(std::vector<unsigned short>& buf, int width, int x0, int x1, int y0, int y1,
                  unsigned short r, unsigned short g, unsigned short b)
    {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x)
                SetPixel(buf, width, x, y, r, g, b);
    }

    void RunNoGhostingTest()
    {
        // A covers [0,250), B covers [150,400) - overlap [150,250), naive
        // Voronoi midpoint ~x=200. Background agrees everywhere. Each
        // buffer additionally draws the SAME conceptual foreground object
        // at a slightly different x position - simulating parallax after
        // otherwise-correct alignment (exactly the reported bug). Objects
        // sit well inside each contributor's own dominant sub-region of
        // the overlap (close to the OTHER contributor's true edge, far
        // from the crossover point) so the multi-band feather has fully
        // saturated to a single source by the time it reaches them.
        constexpr int width = 400, height = 40;
        std::vector<unsigned short> a(static_cast<size_t>(width) * height * 3, 0);
        std::vector<unsigned short> b(static_cast<size_t>(width) * height * 3, 0);

        FillBackground(a, width, height, 0, 250);
        FillBackground(b, width, height, 150, 400);

        const unsigned short objR = 60000, objG = 2000, objB = 2000;
        FillRect(a, width, 160, 170, 0, height, objR, objG, objB); // A's copy of the object
        FillRect(b, width, 230, 240, 0, height, objR, objG, objB); // B's copy, shifted (parallax)

        std::vector<const unsigned short*> buffers{ a.data(), b.data() };
        std::vector<unsigned short> result;
        SeamBlend::BlendChunkContributors(buffers, width, height, result);

        unsigned short expectedObj[3] = { objR, objG, objB };
        unsigned short outAtA[3], outAtB[3];
        GetPixel(result, width, 165, height / 2, outAtA);
        GetPixel(result, width, 235, height / 2, outAtB);

        // A plain average of object-vs-background there would differ from
        // the pure object color by roughly half the object/background gap
        // (tens of thousands) - a tight tolerance here is only satisfiable
        // if a single source won outright, not a blend.
        Check(PixelDiff(outAtA, expectedObj) < 3000, "A's own object position renders as a clean single source, not a ghost blend");
        Check(PixelDiff(outAtB, expectedObj) < 3000, "B's own object position renders as a clean single source, not a ghost blend");
    }

    void RunGapHandlingTest()
    {
        // No overlap at all: A covers only the left half, B only the
        // right half. Must reconstruct the full width with no erased
        // (spuriously zeroed) pixels - the guarantee the old
        // CombineIgnoringGaps provided.
        constexpr int width = 200, height = 40;
        std::vector<unsigned short> a(static_cast<size_t>(width) * height * 3, 0);
        std::vector<unsigned short> b(static_cast<size_t>(width) * height * 3, 0);

        FillBackground(a, width, height, 0, 100);
        FillBackground(b, width, height, 100, 200);

        std::vector<const unsigned short*> buffers{ a.data(), b.data() };
        std::vector<unsigned short> result;
        SeamBlend::BlendChunkContributors(buffers, width, height, result);

        bool anyZero = false;
        for (int y = 0; y < height && !anyZero; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                unsigned short px[3];
                GetPixel(result, width, x, y, px);
                if (px[0] == 0 && px[1] == 0 && px[2] == 0) { anyZero = true; break; }
            }
        }
        Check(!anyZero, "no pixel is spuriously erased to (0,0,0) when contributors partition the chunk with no overlap");

        unsigned short expectedLeft[3], expectedRight[3], outLeft[3], outRight[3];
        GetPixel(a, width, 10, 5, expectedLeft);
        GetPixel(b, width, 190, 5, expectedRight);
        GetPixel(result, width, 10, 5, outLeft);
        GetPixel(result, width, 190, 5, outRight);
        Check(PixelDiff(outLeft, expectedLeft) < 500, "far-left pixel matches A's own value exactly");
        Check(PixelDiff(outRight, expectedRight) < 500, "far-right pixel matches B's own value exactly");
    }

    void RunSeamRoutingTest()
    {
        // Same overlap geometry as the no-ghosting test, but this time the
        // two images otherwise agree on the background and disagree
        // strongly ONLY in a narrow band straddling the naive geometric
        // midpoint (x=100) - the position a plain distance-based Voronoi
        // seam would put the boundary with no cost-aware refinement.
        constexpr int width = 200, height = 40;
        std::vector<unsigned short> a(static_cast<size_t>(width) * height * 3, 0);
        std::vector<unsigned short> b(static_cast<size_t>(width) * height * 3, 0);

        FillBackground(a, width, height, 0, 120);
        FillBackground(b, width, height, 80, 200);

        // Strong, narrow disagreement patch centered on the naive midline.
        FillRect(a, width, 98, 102, 0, height, 60000, 2000, 2000);
        FillRect(b, width, 98, 102, 0, height, 2000, 2000, 60000);

        std::vector<const unsigned short*> buffers{ a.data(), b.data() };
        std::vector<unsigned short> result;
        SeamBlend::BlendChunkContributors(buffers, width, height, result);

        unsigned short valA[3], valB[3], out[3];
        GetPixel(a, width, 100, height / 2, valA);
        GetPixel(b, width, 100, height / 2, valB);
        GetPixel(result, width, 100, height / 2, out);

        int diffA = PixelDiff(out, valA);
        int diffB = PixelDiff(out, valB);
        int total = diffA + diffB;

        // A naive/un-refined boundary sitting exactly on this disagreement
        // patch would blend close to 50/50 (diffA ~= diffB ~= total/2).
        // Seam routing should pull the effective blend weight toward one
        // side instead of splitting the disagreement evenly.
        Check(total > 0, "sanity: A and B actually disagree at the naive midline in this scenario");
        int closer = (std::min)(diffA, diffB);
        Check(closer * 10 < total * 3, "seam routing favors one source at the disagreement patch instead of a near-even 50/50 blend");
    }
}

int main()
{
    RunNoGhostingTest();
    RunGapHandlingTest();
    RunSeamRoutingTest();

    if (g_failures == 0)
    {
        std::cout << "\nAll render_blend checks passed." << std::endl;
        return 0;
    }
    std::cerr << "\n" << g_failures << " render_blend check(s) failed." << std::endl;
    return 1;
}
