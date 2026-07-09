#pragma once

namespace WindowsApp::Core
{
    enum class SimdTier
    {
        Scalar,
        Avx2,
        Avx512
    };

    // Runtime CPUID (+ XGETBV OS-support check) detection, cached after
    // the first call. Checked once at CpuComputeBackend::Initialize() and
    // used to pick which Kernels::{Scalar,Avx2,Avx512} function pointers
    // the tiered kernels (MedianStack/WarpPerspective/BayerDemosaic) use -
    // AVX-512 is only reported if the OS actually saves AVX-512 register
    // state (some CPUs expose the CPUID bit with the feature disabled in
    // firmware/OS, which would otherwise crash on first use).
    SimdTier DetectCpuSimdTier();

    const char* SimdTierName(SimdTier tier);
}
