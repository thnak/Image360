#pragma once
#include <vector>

namespace WindowsApp::Core::Kernels::PainterlyToneCurve
{
    // Night Sight's "painterly" S-curve finish (docs/COMPUTATIONAL_PHOTOGRAPHY.md
    // SS2.2, docs/superpowers/plans/2026-07-22-night-sight.md Architecture
    // SS2) - distinct from HDR+'s multi-exposure ExposureFusion: this is a
    // single-image tone operator, not a Laplacian-pyramid blend. Per
    // channel, per pixel: shadowGamma > 1 crushes shadows/midtones,
    // highlightRolloff softly compresses highlights (Reinhard-style, avoids
    // hard clipping), vignetteStrength darkens the image radially outward
    // from center (the "darkened surrounds" §2.2 describes). All three are
    // real-but-untuned defaults (kNightSight* in BurstCommon.h), not a
    // scene-adaptive/learned classifier - that's out of scope for this
    // phase (see the plan doc's SS9).
    void Apply(const unsigned short* src, int width, int height,
               float shadowGamma, float highlightRolloff, float vignetteStrength,
               std::vector<unsigned short>& outDst);
}
