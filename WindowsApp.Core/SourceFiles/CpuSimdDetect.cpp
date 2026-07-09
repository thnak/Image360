#include "pch.h"
#include "HeaderFiles/CpuSimdDetect.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace WindowsApp::Core
{
    namespace
    {
#if defined(_MSC_VER)
        bool CheckAvx512()
        {
            int regs[4] = {};
            __cpuid(regs, 0);
            if (regs[0] < 7) return false;

            __cpuidex(regs, 7, 0);
            bool avx512f = (regs[1] & (1 << 16)) != 0;
            bool avx512dq = (regs[1] & (1 << 17)) != 0;
            bool avx512bw = (regs[1] & (1 << 30)) != 0;
            bool avx512vl = (regs[1] & (1u << 31)) != 0;
            if (!(avx512f && avx512dq && avx512bw && avx512vl)) return false;

            __cpuid(regs, 1);
            bool osxsave = (regs[2] & (1 << 27)) != 0;
            if (!osxsave) return false;

            unsigned long long xcr0 = _xgetbv(0);
            // bits: 1 (SSE), 2 (AVX/YMM), 5 (opmask), 6 (ZMM_Hi256), 7 (Hi16_ZMM)
            const unsigned long long kAvx512StateMask = (1ull << 1) | (1ull << 2) | (1ull << 5) | (1ull << 6) | (1ull << 7);
            return (xcr0 & kAvx512StateMask) == kAvx512StateMask;
        }

        bool CheckAvx2()
        {
            int regs[4] = {};
            __cpuid(regs, 0);
            if (regs[0] < 7) return false;

            __cpuidex(regs, 7, 0);
            bool avx2 = (regs[1] & (1 << 5)) != 0;
            if (!avx2) return false;

            __cpuid(regs, 1);
            bool osxsave = (regs[2] & (1 << 27)) != 0;
            if (!osxsave) return false;

            unsigned long long xcr0 = _xgetbv(0);
            const unsigned long long kAvxStateMask = (1ull << 1) | (1ull << 2);
            return (xcr0 & kAvxStateMask) == kAvxStateMask;
        }
#else
        // Raw opcode bytes (0F 01 D0) instead of a compiler intrinsic -
        // GCC/Clang only expose _xgetbv/__builtin_ia32_xgetbv when the
        // translation unit is compiled with -mxsave, and this file
        // deliberately carries no special arch flags (it just needs to run
        // on every CPU to decide which tier to dispatch to).
        unsigned long long XGetBv(unsigned int index)
        {
            unsigned int eax, edx;
            __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(index));
            return (static_cast<unsigned long long>(edx) << 32) | eax;
        }

        bool CheckAvx512()
        {
            unsigned int eax, ebx, ecx, edx;
            if (!__get_cpuid_max(0, nullptr) || __get_cpuid_max(0, nullptr) < 7) return false;

            if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
            bool avx512f = (ebx & (1 << 16)) != 0;
            bool avx512dq = (ebx & (1 << 17)) != 0;
            bool avx512bw = (ebx & (1 << 30)) != 0;
            bool avx512vl = (ebx & (1u << 31)) != 0;
            if (!(avx512f && avx512dq && avx512bw && avx512vl)) return false;

            if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
            bool osxsave = (ecx & (1 << 27)) != 0;
            if (!osxsave) return false;

            unsigned long long xcr0 = XGetBv(0);
            const unsigned long long kAvx512StateMask = (1ull << 1) | (1ull << 2) | (1ull << 5) | (1ull << 6) | (1ull << 7);
            return (xcr0 & kAvx512StateMask) == kAvx512StateMask;
        }

        bool CheckAvx2()
        {
            unsigned int eax, ebx, ecx, edx;
            if (!__get_cpuid_max(0, nullptr) || __get_cpuid_max(0, nullptr) < 7) return false;

            if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
            bool avx2 = (ebx & (1 << 5)) != 0;
            if (!avx2) return false;

            if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
            bool osxsave = (ecx & (1 << 27)) != 0;
            if (!osxsave) return false;

            unsigned long long xcr0 = XGetBv(0);
            const unsigned long long kAvxStateMask = (1ull << 1) | (1ull << 2);
            return (xcr0 & kAvxStateMask) == kAvxStateMask;
        }
#endif
    }

    SimdTier DetectCpuSimdTier()
    {
        static const SimdTier cached = []() {
            if (CheckAvx512()) return SimdTier::Avx512;
            if (CheckAvx2()) return SimdTier::Avx2;
            return SimdTier::Scalar;
        }();
        return cached;
    }

    const char* SimdTierName(SimdTier tier)
    {
        switch (tier)
        {
        case SimdTier::Avx512: return "CPU (AVX-512)";
        case SimdTier::Avx2: return "CPU (AVX2)";
        default: return "CPU (Scalar)";
        }
    }
}
