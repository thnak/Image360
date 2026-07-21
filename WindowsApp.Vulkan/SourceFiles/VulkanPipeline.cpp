#include "HeaderFiles/VulkanPipeline.h"

#include "HeaderFiles/RgbToGray.h"
#include "HeaderFiles/FastFeatureDetector.h"
#include "HeaderFiles/BriefDescriptorExtractor.h"
#include "HeaderFiles/FeatureMatcher.h"
#include "HeaderFiles/GainColorOps.h"

#include <vulkan/vulkan.h>

// CMake-embedded SPIR-V byte arrays (generated at build time from
// shaders/*.comp by EmbedShader.cmake - see WindowsApp.Vulkan/CMakeLists.txt).
#include "apply_gain.spv.h"
#include "warp_perspective.spv.h"
#include "median_stack.spv.h"
#include "demosaic_bayer.spv.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace WindowsApp { namespace Compute
{
    // All Vulkan usage here is intentionally minimal/headless (no
    // surface/swapchain) and correctness-first over performance: every
    // call marshals plain host pointers in/out via freshly-created
    // host-visible/coherent buffers (no persistent device-resident
    // state), matching the interface's existing per-call contract - the
    // same thing CudaPipeline does with per-call cudaMalloc/cudaMemcpy.
    struct VulkanContext
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;

        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

        // Fills MedianStack's unused input descriptor slots (numInputs < 32)
        // so every dispatch presents a fully valid descriptor set - Vulkan
        // treats a variably-indexed descriptor array as statically used in
        // full, regardless of the push-constant loop bound the shader
        // actually honors at runtime.
        VkBuffer dummyBuffer = VK_NULL_HANDLE;
        VkDeviceMemory dummyMemory = VK_NULL_HANDLE;

        struct PipelineSet
        {
            VkShaderModule shaderModule = VK_NULL_HANDLE;
            VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
            VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
            VkPipeline pipeline = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        };

        PipelineSet applyGain;
        PipelineSet warpPerspective;
        PipelineSet medianStack;
        PipelineSet demosaicBayer;

        VkPhysicalDeviceProperties deviceProperties{};
        VkPhysicalDeviceMemoryProperties memoryProperties{};

        // WarpPerspective/MedianStack/ApplyGain/DemosaicBayer all dispatch
        // through the single shared commandBuffer/fence/computeQueue above
        // and rewrite each PipelineSet's one persistent descriptorSet before
        // every submit. The Vulkan spec requires host access to all of
        // those to be externally synchronized, but callers here are
        // WindowsApp::Core executors running concurrently on
        // PipelineDriver's thread pool - serialize every dispatch through
        // this mutex rather than giving each call its own command
        // buffer/fence/descriptor set, since these calls are not the
        // pipeline's bottleneck and per-call Vulkan object churn would add
        // real complexity for no measured benefit.
        std::mutex dispatchMutex;
    };

    namespace
    {
        constexpr uint32_t kWorkgroupSize = 256;

        uint32_t GroupCount(size_t totalInvocations)
        {
            return static_cast<uint32_t>((totalInvocations + kWorkgroupSize - 1) / kWorkgroupSize);
        }

        std::vector<uint32_t> WidenU16(const unsigned short* src, size_t count)
        {
            std::vector<uint32_t> out(count);
            for (size_t i = 0; i < count; ++i) out[i] = src[i];
            return out;
        }

        void NarrowToU16(const uint32_t* src, size_t count, unsigned short* dst)
        {
            for (size_t i = 0; i < count; ++i) dst[i] = static_cast<unsigned short>(src[i]);
        }

        uint32_t FindMemoryType(VulkanContext* ctx, uint32_t typeFilter, VkMemoryPropertyFlags properties)
        {
            for (uint32_t i = 0; i < ctx->memoryProperties.memoryTypeCount; ++i)
            {
                if ((typeFilter & (1u << i)) &&
                    (ctx->memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                {
                    return i;
                }
            }
            return UINT32_MAX;
        }

        bool CreateBuffer(VulkanContext* ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags memProps, VkBuffer& outBuffer, VkDeviceMemory& outMemory)
        {
            VkBufferCreateInfo bufInfo{};
            bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufInfo.size = size;
            bufInfo.usage = usage;
            bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(ctx->device, &bufInfo, nullptr, &outBuffer) != VK_SUCCESS) return false;

            VkMemoryRequirements memReq{};
            vkGetBufferMemoryRequirements(ctx->device, outBuffer, &memReq);

            uint32_t memType = FindMemoryType(ctx, memReq.memoryTypeBits, memProps);
            if (memType == UINT32_MAX)
            {
                vkDestroyBuffer(ctx->device, outBuffer, nullptr);
                outBuffer = VK_NULL_HANDLE;
                return false;
            }

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memReq.size;
            allocInfo.memoryTypeIndex = memType;
            if (vkAllocateMemory(ctx->device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS)
            {
                vkDestroyBuffer(ctx->device, outBuffer, nullptr);
                outBuffer = VK_NULL_HANDLE;
                return false;
            }

            vkBindBufferMemory(ctx->device, outBuffer, outMemory, 0);
            return true;
        }

        void DestroyBuffer(VulkanContext* ctx, VkBuffer buffer, VkDeviceMemory memory)
        {
            if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(ctx->device, buffer, nullptr);
            if (memory != VK_NULL_HANDLE) vkFreeMemory(ctx->device, memory, nullptr);
        }

        void BindStorageBuffer(VulkanContext* ctx, VkDescriptorSet set, uint32_t binding, uint32_t arrayElement, VkBuffer buffer)
        {
            VkDescriptorBufferInfo info{};
            info.buffer = buffer;
            info.offset = 0;
            info.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set;
            write.dstBinding = binding;
            write.dstArrayElement = arrayElement;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &info;

            vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
        }

        bool CreatePipeline(VulkanContext* ctx, const uint32_t* code, size_t codeSizeBytes,
                             int numBindings, bool firstBindingIsArray32, size_t pushConstantSize,
                             VulkanContext::PipelineSet& out)
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            for (int i = 0; i < numBindings; ++i)
            {
                VkDescriptorSetLayoutBinding b{};
                b.binding = static_cast<uint32_t>(i);
                b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b.descriptorCount = (firstBindingIsArray32 && i == 0) ? 32u : 1u;
                b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                bindings.push_back(b);
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            if (vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, nullptr, &out.descriptorSetLayout) != VK_SUCCESS)
                return false;

            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.offset = 0;
            pcRange.size = static_cast<uint32_t>(pushConstantSize);

            VkPipelineLayoutCreateInfo plInfo{};
            plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plInfo.setLayoutCount = 1;
            plInfo.pSetLayouts = &out.descriptorSetLayout;
            plInfo.pushConstantRangeCount = 1;
            plInfo.pPushConstantRanges = &pcRange;
            if (vkCreatePipelineLayout(ctx->device, &plInfo, nullptr, &out.pipelineLayout) != VK_SUCCESS)
                return false;

            VkShaderModuleCreateInfo smInfo{};
            smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smInfo.codeSize = codeSizeBytes;
            smInfo.pCode = code;
            if (vkCreateShaderModule(ctx->device, &smInfo, nullptr, &out.shaderModule) != VK_SUCCESS)
                return false;

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = out.shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo cpInfo{};
            cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpInfo.stage = stageInfo;
            cpInfo.layout = out.pipelineLayout;
            if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &out.pipeline) != VK_SUCCESS)
                return false;

            VkDescriptorSetAllocateInfo dsAlloc{};
            dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsAlloc.descriptorPool = ctx->descriptorPool;
            dsAlloc.descriptorSetCount = 1;
            dsAlloc.pSetLayouts = &out.descriptorSetLayout;
            if (vkAllocateDescriptorSets(ctx->device, &dsAlloc, &out.descriptorSet) != VK_SUCCESS)
                return false;

            if (firstBindingIsArray32)
            {
                for (uint32_t i = 0; i < 32u; ++i)
                {
                    BindStorageBuffer(ctx, out.descriptorSet, 0, i, ctx->dummyBuffer);
                }
            }

            return true;
        }

        void DestroyPipelineSet(VkDevice device, VulkanContext::PipelineSet& ps)
        {
            if (ps.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, ps.pipeline, nullptr);
            if (ps.pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, ps.pipelineLayout, nullptr);
            if (ps.descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, ps.descriptorSetLayout, nullptr);
            if (ps.shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, ps.shaderModule, nullptr);
            ps = VulkanContext::PipelineSet{};
        }

        bool DispatchAndWait(VulkanContext* ctx, const VulkanContext::PipelineSet& ps,
                              uint32_t groupCountX, const void* pushConstants, uint32_t pushConstantSize)
        {
            vkResetFences(ctx->device, 1, &ctx->fence);
            vkResetCommandBuffer(ctx->commandBuffer, 0);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(ctx->commandBuffer, &beginInfo) != VK_SUCCESS) return false;

            vkCmdBindPipeline(ctx->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ps.pipeline);
            vkCmdBindDescriptorSets(ctx->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     ps.pipelineLayout, 0, 1, &ps.descriptorSet, 0, nullptr);
            if (pushConstantSize > 0)
            {
                vkCmdPushConstants(ctx->commandBuffer, ps.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                                    0, pushConstantSize, pushConstants);
            }
            vkCmdDispatch(ctx->commandBuffer, groupCountX, 1, 1);

            if (vkEndCommandBuffer(ctx->commandBuffer) != VK_SUCCESS) return false;

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &ctx->commandBuffer;

            if (vkQueueSubmit(ctx->computeQueue, 1, &submitInfo, ctx->fence) != VK_SUCCESS) return false;

            return vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        }
    }

    VulkanPipeline::VulkanPipeline() : m_ctx(nullptr)
    {
        m_lastError[0] = '\0';
    }

    VulkanPipeline::~VulkanPipeline()
    {
        Shutdown();
    }

    void VulkanPipeline::SetError(const char* msg)
    {
        std::snprintf(m_lastError, sizeof(m_lastError), "%s", msg);
    }

    ComputeResult VulkanPipeline::Initialize()
    {
        Shutdown();
        m_ctx = new VulkanContext();

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Image360Engine";
        appInfo.apiVersion = VK_API_VERSION_1_2;

        std::vector<const char*> enabledLayers;
#ifndef NDEBUG
        {
            uint32_t layerCount = 0;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
            std::vector<VkLayerProperties> layers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
            for (const auto& layer : layers)
            {
                if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
                {
                    enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
                    break;
                }
            }
        }
#endif

        VkInstanceCreateInfo instInfo{};
        instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo = &appInfo;
        instInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
        instInfo.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();

        if (vkCreateInstance(&instInfo, nullptr, &m_ctx->instance) != VK_SUCCESS)
        {
            SetError("vkCreateInstance failed - no Vulkan runtime/driver available.");
            delete m_ctx;
            m_ctx = nullptr;
            return ComputeResult::NO_GPU;
        }

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_ctx->instance, &deviceCount, nullptr);
        if (deviceCount == 0)
        {
            SetError("No Vulkan-capable physical devices found.");
            Shutdown();
            return ComputeResult::NO_GPU;
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_ctx->instance, &deviceCount, devices.data());

        VkPhysicalDevice chosen = VK_NULL_HANDLE;
        uint32_t chosenQueueFamily = 0;
        bool haveCandidate = false;
        bool haveDiscrete = false;

        for (VkPhysicalDevice dev : devices)
        {
            uint32_t qfCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
            std::vector<VkQueueFamilyProperties> qfs(qfCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfs.data());

            for (uint32_t i = 0; i < qfCount; ++i)
            {
                if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                {
                    VkPhysicalDeviceProperties props{};
                    vkGetPhysicalDeviceProperties(dev, &props);
                    bool isDiscrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

                    if (!haveCandidate || (isDiscrete && !haveDiscrete))
                    {
                        chosen = dev;
                        chosenQueueFamily = i;
                        haveCandidate = true;
                        haveDiscrete = isDiscrete;
                    }
                    break;
                }
            }
        }

        if (!haveCandidate)
        {
            SetError("No Vulkan device with a compute queue family found.");
            Shutdown();
            return ComputeResult::NO_GPU;
        }

        m_ctx->physicalDevice = chosen;
        m_ctx->queueFamilyIndex = chosenQueueFamily;
        vkGetPhysicalDeviceProperties(chosen, &m_ctx->deviceProperties);
        vkGetPhysicalDeviceMemoryProperties(chosen, &m_ctx->memoryProperties);

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = chosenQueueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;

        if (vkCreateDevice(chosen, &deviceInfo, nullptr, &m_ctx->device) != VK_SUCCESS)
        {
            SetError("vkCreateDevice failed.");
            Shutdown();
            return ComputeResult::NO_GPU;
        }

        vkGetDeviceQueue(m_ctx->device, chosenQueueFamily, 0, &m_ctx->computeQueue);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = chosenQueueFamily;
        if (vkCreateCommandPool(m_ctx->device, &poolInfo, nullptr, &m_ctx->commandPool) != VK_SUCCESS)
        {
            SetError("vkCreateCommandPool failed.");
            Shutdown();
            return ComputeResult::KERNEL_LAUNCH_FAILED;
        }

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = m_ctx->commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_ctx->device, &cmdAllocInfo, &m_ctx->commandBuffer) != VK_SUCCESS)
        {
            SetError("vkAllocateCommandBuffers failed.");
            Shutdown();
            return ComputeResult::KERNEL_LAUNCH_FAILED;
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(m_ctx->device, &fenceInfo, nullptr, &m_ctx->fence) != VK_SUCCESS)
        {
            SetError("vkCreateFence failed.");
            Shutdown();
            return ComputeResult::KERNEL_LAUNCH_FAILED;
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 64;

        VkDescriptorPoolCreateInfo descPoolInfo{};
        descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descPoolInfo.poolSizeCount = 1;
        descPoolInfo.pPoolSizes = &poolSize;
        descPoolInfo.maxSets = 4;
        if (vkCreateDescriptorPool(m_ctx->device, &descPoolInfo, nullptr, &m_ctx->descriptorPool) != VK_SUCCESS)
        {
            SetError("vkCreateDescriptorPool failed.");
            Shutdown();
            return ComputeResult::KERNEL_LAUNCH_FAILED;
        }

        VkMemoryPropertyFlags hostProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (!CreateBuffer(m_ctx, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, m_ctx->dummyBuffer, m_ctx->dummyMemory))
        {
            SetError("Failed to create dummy buffer.");
            Shutdown();
            return ComputeResult::OUT_OF_MEMORY;
        }

        bool pipelinesOk =
            CreatePipeline(m_ctx, apply_gain_spv, apply_gain_spv_size, 1, false, 8, m_ctx->applyGain) &&
            CreatePipeline(m_ctx, warp_perspective_spv, warp_perspective_spv_size, 2, false, 60, m_ctx->warpPerspective) &&
            CreatePipeline(m_ctx, median_stack_spv, median_stack_spv_size, 2, true, 12, m_ctx->medianStack) &&
            CreatePipeline(m_ctx, demosaic_bayer_spv, demosaic_bayer_spv_size, 2, false, 80, m_ctx->demosaicBayer);

        if (!pipelinesOk)
        {
            SetError("Failed to create one or more Vulkan compute pipelines.");
            Shutdown();
            return ComputeResult::KERNEL_LAUNCH_FAILED;
        }

        return ComputeResult::SUCCESS;
    }

    void VulkanPipeline::Shutdown()
    {
        if (!m_ctx) return;

        if (m_ctx->device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_ctx->device);

            DestroyPipelineSet(m_ctx->device, m_ctx->applyGain);
            DestroyPipelineSet(m_ctx->device, m_ctx->warpPerspective);
            DestroyPipelineSet(m_ctx->device, m_ctx->medianStack);
            DestroyPipelineSet(m_ctx->device, m_ctx->demosaicBayer);

            if (m_ctx->dummyBuffer != VK_NULL_HANDLE) vkDestroyBuffer(m_ctx->device, m_ctx->dummyBuffer, nullptr);
            if (m_ctx->dummyMemory != VK_NULL_HANDLE) vkFreeMemory(m_ctx->device, m_ctx->dummyMemory, nullptr);

            if (m_ctx->descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_ctx->device, m_ctx->descriptorPool, nullptr);
            if (m_ctx->fence != VK_NULL_HANDLE) vkDestroyFence(m_ctx->device, m_ctx->fence, nullptr);
            if (m_ctx->commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(m_ctx->device, m_ctx->commandPool, nullptr);

            vkDestroyDevice(m_ctx->device, nullptr);
        }

        if (m_ctx->instance != VK_NULL_HANDLE) vkDestroyInstance(m_ctx->instance, nullptr);

        delete m_ctx;
        m_ctx = nullptr;
    }

    bool VulkanPipeline::IsInitialized() const
    {
        return m_ctx != nullptr && m_ctx->device != VK_NULL_HANDLE;
    }

    GpuInfo VulkanPipeline::GetGpuInfo() const
    {
        GpuInfo info{};
        if (!m_ctx || m_ctx->physicalDevice == VK_NULL_HANDLE) return info;

        info.deviceId = static_cast<int>(m_ctx->deviceProperties.deviceID);
        std::snprintf(info.name, sizeof(info.name), "%s", m_ctx->deviceProperties.deviceName);

        VkDeviceSize maxHeap = 0;
        for (uint32_t i = 0; i < m_ctx->memoryProperties.memoryHeapCount; ++i)
        {
            if (m_ctx->memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                maxHeap = (std::max)(maxHeap, m_ctx->memoryProperties.memoryHeaps[i].size);
            }
        }
        info.totalMemory = static_cast<size_t>(maxHeap);
        // Vulkan core has no free-memory query without VK_EXT_memory_budget -
        // report the heap total for both, same cosmetic-best-effort spirit
        // as CpuComputeBackend's non-GPU GpuInfo fields.
        info.freeMemory = static_cast<size_t>(maxHeap);
        info.computeMajor = 0;
        info.computeMinor = 0;
        info.maxThreadsPerBlock = static_cast<int>(m_ctx->deviceProperties.limits.maxComputeWorkGroupInvocations);
        info.multiProcessorCount = 0; // no direct Vulkan equivalent to CUDA's SM count
        info.hasTensorCores = false;
        return info;
    }

    const char* VulkanPipeline::GetLastError() const
    {
        return m_lastError;
    }

    ComputeResult VulkanPipeline::WarpPerspective(
        const unsigned short* srcData, int srcW, int srcH,
        unsigned short* dstData, int dstW, int dstH,
        const float* homography, int offsetX, int offsetY)
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!srcData || !dstData || !homography) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        std::lock_guard<std::mutex> dispatchLock(m_ctx->dispatchMutex);

        size_t srcCount = static_cast<size_t>(srcW) * srcH * 3;
        size_t dstPixels = static_cast<size_t>(dstW) * dstH;
        size_t dstCount = dstPixels * 3;

        VkMemoryPropertyFlags hostProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkBuffer srcBuf = VK_NULL_HANDLE, dstBuf = VK_NULL_HANDLE;
        VkDeviceMemory srcMem = VK_NULL_HANDLE, dstMem = VK_NULL_HANDLE;

        if (!CreateBuffer(m_ctx, srcCount * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, srcBuf, srcMem) ||
            !CreateBuffer(m_ctx, dstCount * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, dstBuf, dstMem))
        {
            SetError("Failed to allocate Vulkan buffers for WarpPerspective.");
            DestroyBuffer(m_ctx, srcBuf, srcMem);
            DestroyBuffer(m_ctx, dstBuf, dstMem);
            return ComputeResult::OUT_OF_MEMORY;
        }

        std::vector<uint32_t> srcWide = WidenU16(srcData, srcCount);
        void* mapped = nullptr;
        vkMapMemory(m_ctx->device, srcMem, 0, VK_WHOLE_SIZE, 0, &mapped);
        std::memcpy(mapped, srcWide.data(), srcCount * sizeof(uint32_t));
        vkUnmapMemory(m_ctx->device, srcMem);

        BindStorageBuffer(m_ctx, m_ctx->warpPerspective.descriptorSet, 0, 0, srcBuf);
        BindStorageBuffer(m_ctx, m_ctx->warpPerspective.descriptorSet, 1, 0, dstBuf);

        struct WarpPC { float invH[9]; int32_t srcW, srcH, dstW, dstH, offsetX, offsetY; } pc{};
        std::memcpy(pc.invH, homography, sizeof(pc.invH));
        pc.srcW = srcW; pc.srcH = srcH; pc.dstW = dstW; pc.dstH = dstH; pc.offsetX = offsetX; pc.offsetY = offsetY;

        bool ok = DispatchAndWait(m_ctx, m_ctx->warpPerspective, GroupCount(dstPixels), &pc, sizeof(pc));

        if (ok)
        {
            vkMapMemory(m_ctx->device, dstMem, 0, VK_WHOLE_SIZE, 0, &mapped);
            NarrowToU16(static_cast<const uint32_t*>(mapped), dstCount, dstData);
            vkUnmapMemory(m_ctx->device, dstMem);
        }

        DestroyBuffer(m_ctx, srcBuf, srcMem);
        DestroyBuffer(m_ctx, dstBuf, dstMem);

        if (!ok) { SetError("WarpPerspective dispatch failed."); return ComputeResult::KERNEL_LAUNCH_FAILED; }
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::MedianStack(
        const unsigned short** inputs, int numInputs,
        unsigned short* output, int width, int height,
        float sigmaThreshold)
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!inputs || !output) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numInputs <= 0 || numInputs > 32) { SetError("numInputs must be 1-32."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        std::lock_guard<std::mutex> dispatchLock(m_ctx->dispatchMutex);

        size_t numScalars = static_cast<size_t>(width) * height * 3;
        VkMemoryPropertyFlags hostProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        std::vector<VkBuffer> inBufs(static_cast<size_t>(numInputs), VK_NULL_HANDLE);
        std::vector<VkDeviceMemory> inMems(static_cast<size_t>(numInputs), VK_NULL_HANDLE);
        VkBuffer outBuf = VK_NULL_HANDLE;
        VkDeviceMemory outMem = VK_NULL_HANDLE;

        bool allocOk = true;
        for (int i = 0; i < numInputs && allocOk; ++i)
        {
            allocOk = CreateBuffer(m_ctx, numScalars * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, inBufs[i], inMems[i]);
            if (allocOk)
            {
                std::vector<uint32_t> wide = WidenU16(inputs[i], numScalars);
                void* mapped = nullptr;
                vkMapMemory(m_ctx->device, inMems[i], 0, VK_WHOLE_SIZE, 0, &mapped);
                std::memcpy(mapped, wide.data(), numScalars * sizeof(uint32_t));
                vkUnmapMemory(m_ctx->device, inMems[i]);
            }
        }
        if (allocOk)
        {
            allocOk = CreateBuffer(m_ctx, numScalars * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, outBuf, outMem);
        }

        if (!allocOk)
        {
            SetError("Failed to allocate Vulkan buffers for MedianStack.");
            for (int i = 0; i < numInputs; ++i) DestroyBuffer(m_ctx, inBufs[i], inMems[i]);
            DestroyBuffer(m_ctx, outBuf, outMem);
            return ComputeResult::OUT_OF_MEMORY;
        }

        for (int i = 0; i < numInputs; ++i)
        {
            BindStorageBuffer(m_ctx, m_ctx->medianStack.descriptorSet, 0, static_cast<uint32_t>(i), inBufs[i]);
        }
        for (int i = numInputs; i < 32; ++i)
        {
            BindStorageBuffer(m_ctx, m_ctx->medianStack.descriptorSet, 0, static_cast<uint32_t>(i), m_ctx->dummyBuffer);
        }
        BindStorageBuffer(m_ctx, m_ctx->medianStack.descriptorSet, 1, 0, outBuf);

        struct MedianPC { uint32_t numInputs; uint32_t numScalars; float sigmaThreshold; } pc{};
        pc.numInputs = static_cast<uint32_t>(numInputs);
        pc.numScalars = static_cast<uint32_t>(numScalars);
        pc.sigmaThreshold = sigmaThreshold;

        bool ok = DispatchAndWait(m_ctx, m_ctx->medianStack, GroupCount(numScalars), &pc, sizeof(pc));

        if (ok)
        {
            void* mapped = nullptr;
            vkMapMemory(m_ctx->device, outMem, 0, VK_WHOLE_SIZE, 0, &mapped);
            NarrowToU16(static_cast<const uint32_t*>(mapped), numScalars, output);
            vkUnmapMemory(m_ctx->device, outMem);
        }

        for (int i = 0; i < numInputs; ++i) DestroyBuffer(m_ctx, inBufs[i], inMems[i]);
        DestroyBuffer(m_ctx, outBuf, outMem);

        if (!ok) { SetError("MedianStack dispatch failed."); return ComputeResult::KERNEL_LAUNCH_FAILED; }
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::ApplyGain(unsigned short* data, int numPixels, float gain)
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!data) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numPixels <= 0) { SetError("Invalid pixel count."); return ComputeResult::INVALID_PARAM; }

        std::lock_guard<std::mutex> dispatchLock(m_ctx->dispatchMutex);

        VkMemoryPropertyFlags hostProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (!CreateBuffer(m_ctx, static_cast<VkDeviceSize>(numPixels) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, buf, mem))
        {
            SetError("Failed to allocate Vulkan buffer for ApplyGain.");
            return ComputeResult::OUT_OF_MEMORY;
        }

        std::vector<uint32_t> wide = WidenU16(data, static_cast<size_t>(numPixels));
        void* mapped = nullptr;
        vkMapMemory(m_ctx->device, mem, 0, VK_WHOLE_SIZE, 0, &mapped);
        std::memcpy(mapped, wide.data(), wide.size() * sizeof(uint32_t));
        vkUnmapMemory(m_ctx->device, mem);

        BindStorageBuffer(m_ctx, m_ctx->applyGain.descriptorSet, 0, 0, buf);

        struct ApplyGainPC { int32_t numPixels; float gain; } pc{ numPixels, gain };
        bool ok = DispatchAndWait(m_ctx, m_ctx->applyGain, GroupCount(static_cast<size_t>(numPixels)), &pc, sizeof(pc));

        if (ok)
        {
            vkMapMemory(m_ctx->device, mem, 0, VK_WHOLE_SIZE, 0, &mapped);
            NarrowToU16(static_cast<const uint32_t*>(mapped), static_cast<size_t>(numPixels), data);
            vkUnmapMemory(m_ctx->device, mem);
        }

        DestroyBuffer(m_ctx, buf, mem);

        if (!ok) { SetError("ApplyGain dispatch failed."); return ComputeResult::KERNEL_LAUNCH_FAILED; }
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::DemosaicBayer(
        const unsigned short* cfaData, int width, int height,
        unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
        uint32_t filters, unsigned short* rgbOut)
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!cfaData || !camMul || !rgbCam || !rgbOut) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        std::lock_guard<std::mutex> dispatchLock(m_ctx->dispatchMutex);

        size_t numPixels = static_cast<size_t>(width) * height;
        VkMemoryPropertyFlags hostProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        VkBuffer cfaBuf = VK_NULL_HANDLE, outBuf = VK_NULL_HANDLE;
        VkDeviceMemory cfaMem = VK_NULL_HANDLE, outMem = VK_NULL_HANDLE;

        if (!CreateBuffer(m_ctx, numPixels * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, cfaBuf, cfaMem) ||
            !CreateBuffer(m_ctx, numPixels * 3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, outBuf, outMem))
        {
            SetError("Failed to allocate Vulkan buffers for DemosaicBayer.");
            DestroyBuffer(m_ctx, cfaBuf, cfaMem);
            DestroyBuffer(m_ctx, outBuf, outMem);
            return ComputeResult::OUT_OF_MEMORY;
        }

        std::vector<uint32_t> wide = WidenU16(cfaData, numPixels);
        void* mapped = nullptr;
        vkMapMemory(m_ctx->device, cfaMem, 0, VK_WHOLE_SIZE, 0, &mapped);
        std::memcpy(mapped, wide.data(), numPixels * sizeof(uint32_t));
        vkUnmapMemory(m_ctx->device, cfaMem);

        BindStorageBuffer(m_ctx, m_ctx->demosaicBayer.descriptorSet, 0, 0, cfaBuf);
        BindStorageBuffer(m_ctx, m_ctx->demosaicBayer.descriptorSet, 1, 0, outBuf);

        struct DemosaicPC
        {
            int32_t width, height;
            uint32_t blackLevel, filters;
            float camMul[4];
            float rgbCam[12];
        } pc{};
        pc.width = width; pc.height = height;
        pc.blackLevel = blackLevel; pc.filters = filters;
        std::memcpy(pc.camMul, camMul, sizeof(pc.camMul));
        for (int row = 0; row < 3; ++row)
        {
            std::memcpy(pc.rgbCam + row * 4, rgbCam[row], sizeof(float) * 4);
        }

        bool ok = DispatchAndWait(m_ctx, m_ctx->demosaicBayer, GroupCount(numPixels), &pc, sizeof(pc));

        if (ok)
        {
            vkMapMemory(m_ctx->device, outMem, 0, VK_WHOLE_SIZE, 0, &mapped);
            NarrowToU16(static_cast<const uint32_t*>(mapped), numPixels * 3, rgbOut);
            vkUnmapMemory(m_ctx->device, outMem);
        }

        DestroyBuffer(m_ctx, cfaBuf, cfaMem);
        DestroyBuffer(m_ctx, outBuf, outMem);

        if (!ok) { SetError("DemosaicBayer dispatch failed."); return ComputeResult::KERNEL_LAUNCH_FAILED; }
        return ComputeResult::SUCCESS;
    }

    // Delegated to Core's portable CPU kernels - see VulkanPipeline.h's
    // class comment for why (branchy/reduction-heavy work that doesn't
    // dominate pipeline runtime, not worth a dedicated shader).

    ComputeResult VulkanPipeline::DetectAndDescribeFeatures(
        const unsigned char* rgbImage, int width, int height,
        FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints)
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgbImage || !outPoints || !outDescriptors || !outCount) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0 || maxPoints <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        std::vector<unsigned char> gray(static_cast<size_t>(width) * height);
        Core::ConvertRgbToGray(rgbImage, width, height, gray.data());

        int detectedCount = 0;
        Core::DetectFastCorners(gray.data(), width, height, outPoints, &detectedCount, maxPoints);
        int clampedCount = (std::min)(detectedCount, maxPoints);

        if (clampedCount > 0)
        {
            Core::ExtractBriefDescriptors(gray.data(), width, height, outPoints, clampedCount, outDescriptors);
        }

        *outCount = clampedCount;
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::MatchFeatures(
        const BriefDescriptor* descA, int countA,
        const BriefDescriptor* descB, int countB,
        MatchResult* outMatches, int* outMatchCount, int maxMatches,
        float ratioThreshold)
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!descA || !descB || !outMatches || !outMatchCount) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }

        int rawCount = 0;
        Core::MatchFeaturesBruteForce(descA, countA, descB, countB, outMatches, &rawCount, maxMatches, ratioThreshold);
        *outMatchCount = (std::min)(rawCount, maxMatches);
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::ComputeLabStats(
        const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3])
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgb || !outMean || !outStd) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        Core::ComputeLabStatsCpu(rgb, width, height, outMean, outStd);
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::ApplyReinhardColorTransfer(
        unsigned short* rgbInOut, int width, int height,
        const double srcMean[3], const double srcStd[3],
        const double refMean[3], const double refStd[3])
    {
        if (!IsInitialized()) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgbInOut || !srcMean || !srcStd || !refMean || !refStd) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        Core::ApplyReinhardColorTransferCpu(rgbInOut, width, height, srcMean, srcStd, refMean, refStd);
        return ComputeResult::SUCCESS;
    }

    ComputeResult VulkanPipeline::BlockMatchAlign(
        const unsigned short* /*refData*/, const unsigned short* /*srcData*/,
        int /*width*/, int /*height*/, int /*tileSize*/, int /*searchRadius*/,
        TileOffset* /*outOffsets*/, int /*tilesX*/, int /*tilesY*/)
    {
        SetError("BlockMatchAlign not implemented on the Vulkan backend yet "
                 "(tracked gap, see docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md Task 2) "
                 "- use the CPU backend for burst-mode projects.");
        return ComputeResult::NOT_SUPPORTED;
    }

    ComputeResult VulkanPipeline::RobustMergeAccumulate(
        const unsigned short* const* /*frames*/, int /*numFrames*/,
        const TileOffset* const* /*perFrameOffsets*/,
        int /*width*/, int /*height*/, int /*tileSize*/, int /*tilesX*/, int /*tilesY*/,
        float /*sigma*/, unsigned short* /*output*/)
    {
        SetError("RobustMergeAccumulate not implemented on the Vulkan backend yet "
                 "(tracked gap, see docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md Task 2) "
                 "- use the CPU backend for burst-mode projects.");
        return ComputeResult::NOT_SUPPORTED;
    }

    ComputeResult VulkanPipeline::TileFftMerge(
        const unsigned short* const* /*frames*/, int /*numFrames*/,
        const TileOffset* const* /*perFrameOffsets*/,
        int /*width*/, int /*height*/, int /*tileSize*/, int /*tilesX*/, int /*tilesY*/,
        float /*noiseVariance*/, unsigned short* /*output*/)
    {
        SetError("TileFftMerge not implemented on the Vulkan backend yet "
                 "(tracked gap, see docs/superpowers/plans/2026-07-21-hdrplus-tile-fft-merge.md Task 1) "
                 "- use the CPU backend for burst-mode projects.");
        return ComputeResult::NOT_SUPPORTED;
    }
}}
