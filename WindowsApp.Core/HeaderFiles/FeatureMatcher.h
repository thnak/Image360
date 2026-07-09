#pragma once

#include "ComputeTypes.h"

namespace WindowsApp::Core
{
    // Brute-force Hamming-distance descriptor matching with Lowe's ratio
    // test - replicates
    // WindowsApp::Compute::Kernels::BruteForceMatchKernel's algorithm
    // exactly (best/second-best per descriptor in A, accept if
    // best < ratioThreshold * second, or unconditionally if there's no
    // second candidate). Returns at most maxMatches results in outMatches;
    // outMatchCount may exceed maxMatches if more were found (matching the
    // GPU kernel's own atomicAdd-then-clamp behavior).
    void MatchFeaturesBruteForce(
        const Compute::BriefDescriptor* descA, int countA,
        const Compute::BriefDescriptor* descB, int countB,
        Compute::MatchResult* outMatches, int* outMatchCount, int maxMatches,
        float ratioThreshold = 0.75f);
}
