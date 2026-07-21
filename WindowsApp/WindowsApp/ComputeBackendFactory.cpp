#include "pch.h"
#include "ComputeBackendFactory.h"

#include "CudaPipeline.h"
#include "NvJpegCodec.h"
#include "VulkanPipeline.h"
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

        bool TryCuda(ComputeBackendSelection& sel, std::wstring& lastError)
        {
            auto cuda = std::make_shared<::WindowsApp::Compute::CudaPipeline>();
            if (cuda->Initialize() != ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                lastError = WidenAscii(cuda->GetLastError());
                return false;
            }

            auto jpeg = std::make_shared<::WindowsApp::Compute::NvJpegCodec>();
            if (jpeg->Initialize() != ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                // GPU initialized but the JPEG codec didn't - fall back
                // rather than a half-GPU/half-CPU mix.
                lastError = WidenAscii(jpeg->GetLastError());
                return false;
            }

            auto info = cuda->GetGpuInfo();
            sel.backend = cuda;
            sel.codec = jpeg;
            sel.usedGpu = true;
            sel.kind = ComputeBackendKind::Cuda;
            sel.recommendedMaxInFlight = 2;
            sel.recommendedChunkSize = ::WindowsApp::Core::RecommendedChunkSize(info.totalMemory);
            sel.statusMessage = L"Using GPU (CUDA): " + WidenAscii(info.name);
            return true;
        }

        bool TryVulkan(ComputeBackendSelection& sel, std::wstring& lastError)
        {
            auto vulkan = std::make_shared<::WindowsApp::Compute::VulkanPipeline>();
            if (vulkan->Initialize() != ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                lastError = WidenAscii(vulkan->GetLastError());
                return false;
            }

            // VulkanPipeline has no GPU-accelerated JPEG codec of its own
            // (see WindowsApp.Vulkan/HeaderFiles/VulkanPipeline.h) - pair
            // it with the CPU JpegCodec, same as the CPU-only tier below.
            auto jpeg = std::make_shared<::WindowsApp::Core::JpegCodec>();
            if (jpeg->Initialize() != ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                lastError = WidenAscii(jpeg->GetLastError());
                return false;
            }

            auto info = vulkan->GetGpuInfo();
            sel.backend = vulkan;
            sel.codec = jpeg;
            sel.usedGpu = true;
            sel.kind = ComputeBackendKind::Vulkan;
            sel.recommendedMaxInFlight = 2;
            sel.recommendedChunkSize = ::WindowsApp::Core::RecommendedChunkSize(info.totalMemory);
            sel.statusMessage = L"Using GPU (Vulkan): " + WidenAscii(info.name);
            return true;
        }

        void UseCpu(ComputeBackendSelection& sel, const std::wstring& fallbackReason)
        {
            auto cpu = std::make_shared<::WindowsApp::Core::CpuComputeBackend>();
            cpu->Initialize(); // cannot fail - always some SIMD tier available
            auto cpuJpeg = std::make_shared<::WindowsApp::Core::JpegCodec>();
            cpuJpeg->Initialize(); // rare failure (e.g. OOM) surfaces via each call's own ComputeResult check, same as every other executor guard clause

            auto info = cpu->GetGpuInfo();
            sel.backend = cpu;
            sel.codec = cpuJpeg;
            sel.usedGpu = false;
            sel.kind = ComputeBackendKind::Cpu;
            sel.recommendedMaxInFlight = (std::max)(size_t(1),
                static_cast<size_t>(std::thread::hardware_concurrency()) - 1);
            sel.recommendedChunkSize = ::WindowsApp::Core::RecommendedChunkSizeForCpu(info.totalMemory);
            sel.statusMessage = fallbackReason.empty()
                ? L"Using " + WidenAscii(info.name)
                : L"No compatible GPU (" + fallbackReason + L") - using " + WidenAscii(info.name);
        }
    }

    ComputeBackendSelection SelectComputeBackend(ComputeBackendKind preferred)
    {
        ComputeBackendSelection sel;
        std::wstring lastError;

        if (preferred == ComputeBackendKind::Auto || preferred == ComputeBackendKind::Cuda)
        {
            if (TryCuda(sel, lastError)) return sel;
            if (preferred == ComputeBackendKind::Cuda)
            {
                UseCpu(sel, lastError);
                return sel;
            }
        }

        if (preferred == ComputeBackendKind::Auto || preferred == ComputeBackendKind::Vulkan)
        {
            if (TryVulkan(sel, lastError)) return sel;
            if (preferred == ComputeBackendKind::Vulkan)
            {
                UseCpu(sel, lastError);
                return sel;
            }
        }

        UseCpu(sel, lastError);
        return sel;
    }
}
