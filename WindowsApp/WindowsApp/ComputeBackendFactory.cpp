#include "pch.h"
#include "ComputeBackendFactory.h"

#include "CudaPipeline.h"
#include "NvJpegCodec.h"
#include "CpuComputeBackend.h"
#include "JpegCodec.h"
#include "ProjectManager.h"

#include <algorithm>
#include <thread>

namespace WindowsApp
{
    namespace
    {
        std::wstring WidenAscii(const std::string& s)
        {
            return std::wstring(s.begin(), s.end());
        }

        std::wstring WidenAscii(const char* s)
        {
            return s ? WidenAscii(std::string(s)) : std::wstring();
        }
    }

    ComputeBackendSelection SelectComputeBackend()
    {
        ComputeBackendSelection sel;

        auto cuda = std::make_shared<::WindowsApp::Compute::CudaPipeline>();
        if (cuda->Initialize() == ::WindowsApp::Compute::ComputeResult::SUCCESS)
        {
            auto jpeg = std::make_shared<::WindowsApp::Compute::NvJpegCodec>();
            if (jpeg->Initialize() == ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                auto info = cuda->GetGpuInfo();
                sel.backend = cuda;
                sel.codec = jpeg;
                sel.usedGpu = true;
                sel.recommendedMaxInFlight = 2;
                sel.recommendedChunkSize = ::WindowsApp::Core::RecommendedChunkSize(info.totalMemory);
                sel.statusMessage = L"Using GPU: " + WidenAscii(info.name);
                return sel;
            }
            // GPU initialized but the JPEG codec didn't - fall back to CPU
            // entirely rather than a half-GPU/half-CPU mix.
        }

        std::wstring cudaError = WidenAscii(cuda->GetLastError());

        auto cpu = std::make_shared<::WindowsApp::Core::CpuComputeBackend>();
        cpu->Initialize(); // cannot fail - always some SIMD tier available
        auto cpuJpeg = std::make_shared<::WindowsApp::Core::JpegCodec>();
        cpuJpeg->Initialize(); // rare failure (e.g. OOM) surfaces via each call's own ComputeResult check, same as every other executor guard clause

        auto info = cpu->GetGpuInfo();
        sel.backend = cpu;
        sel.codec = cpuJpeg;
        sel.usedGpu = false;
        sel.recommendedMaxInFlight = (std::max)(size_t(1),
            static_cast<size_t>(std::thread::hardware_concurrency()) - 1);
        sel.recommendedChunkSize = ::WindowsApp::Core::RecommendedChunkSizeForCpu(info.totalMemory);
        sel.statusMessage = L"No compatible GPU (" + cudaError + L") - using " + WidenAscii(info.name);
        return sel;
    }
}
