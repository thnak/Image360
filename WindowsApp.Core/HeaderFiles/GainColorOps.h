#pragma once

#include <cstddef>

namespace WindowsApp::Core
{
    // In-place multiplicative gain, clamped to [0,65535] - matches
    // WindowsApp::Compute::Kernels::ApplyGainKernel exactly. No-op if
    // gain ~= 1.0, matching CudaPipeline::ApplyGain's own early-out.
    void ApplyGainCpu(unsigned short* data, int numPixels, float gain);

    // sRGB(16-bit)->linear->XYZ->CIE Lab mean/std over an interleaved RGB48
    // image - matches WindowsApp::Compute::Kernels::RgbToLabKernel +
    // LabStatsKernel + CudaPipeline::ComputeLabStats's mean/variance
    // derivation exactly. outMean/outStd: 3 doubles (L, a, b).
    void ComputeLabStatsCpu(const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3]);

    // Reinhard color transfer (Lab space, per-channel mean/std match) -
    // matches WindowsApp::Compute::Kernels::ReinhardTransferKernel and the
    // RgbToLab -> transfer -> LabToRgb pipeline order in
    // CudaPipeline::ApplyReinhardColorTransfer exactly. rgbInOut:
    // width*height*3, modified in place.
    void ApplyReinhardColorTransferCpu(
        unsigned short* rgbInOut, int width, int height,
        const double srcMean[3], const double srcStd[3],
        const double refMean[3], const double refStd[3]);
}
