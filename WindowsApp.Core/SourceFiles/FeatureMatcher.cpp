#include "pch.h"
#include "HeaderFiles/FeatureMatcher.h"

#include <climits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace WindowsApp::Core
{
    namespace
    {
        // POPCNT has been standard on x86-64 since ~2008 (Nehalem/
        // Barcelona) - unlike AVX-512/AVX2, no runtime CPUID dispatch here,
        // matching every other x86-64-v2-baseline assumption in this file.
        inline int PopCount64(uint64_t v)
        {
#if defined(_MSC_VER)
            return static_cast<int>(__popcnt64(v));
#else
            return __builtin_popcountll(v);
#endif
        }

        int HammingDistance256(const Compute::BriefDescriptor& a, const Compute::BriefDescriptor& b)
        {
            return PopCount64(a[0] ^ b[0]) + PopCount64(a[1] ^ b[1])
                 + PopCount64(a[2] ^ b[2]) + PopCount64(a[3] ^ b[3]);
        }
    }

    void MatchFeaturesBruteForce(
        const Compute::BriefDescriptor* descA, int countA,
        const Compute::BriefDescriptor* descB, int countB,
        Compute::MatchResult* outMatches, int* outMatchCount, int maxMatches,
        float ratioThreshold)
    {
        *outMatchCount = 0;
        if (!descA || !descB || !outMatches || !outMatchCount) return;

        int count = 0;
        for (int i = 0; i < countA; ++i)
        {
            int best = INT_MAX;
            int second = INT_MAX;
            int bestIdx = -1;

            for (int j = 0; j < countB; ++j)
            {
                int dist = HammingDistance256(descA[i], descB[j]);
                if (dist < best)
                {
                    second = best;
                    best = dist;
                    bestIdx = j;
                }
                else if (dist < second)
                {
                    second = dist;
                }
            }

            if (bestIdx < 0) continue;

            bool accept = (second == INT_MAX)
                || (static_cast<float>(best) < ratioThreshold * static_cast<float>(second));
            if (!accept) continue;

            if (count < maxMatches)
            {
                outMatches[count].indexA = i;
                outMatches[count].indexB = bestIdx;
                outMatches[count].hammingDistance = best;
            }
            ++count;
        }

        *outMatchCount = count;
    }
}
