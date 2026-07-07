#pragma once
#include "Types.h"
#include "ProjectManager.h"
#include <vector>

namespace WindowsApp::Core
{
    // Pure function, no I/O - projects the image's own pixel-space
    // footprint ([0,0]-[imageWidth,imageHeight], matching
    // WarpPerspectiveKernel's convention that a homography maps an
    // image's own pixel coordinates forward into output-canvas space)
    // through `h`, takes the axis-aligned bounding box of the projected
    // quad, and tests it against `chunk`'s rectangle. A conservative
    // superset test (AABB, not exact polygon intersection) - may
    // include a genuinely-non-overlapping image occasionally (one
    // wasted warp), never wrongly excludes a real contributor.
    bool ImageOverlapsChunk(const Homography& h, int imageWidth, int imageHeight, const ChunkModel& chunk);

    // Not pure - opens each image (ImageLoader::GetMetadata) to learn
    // its pixel dimensions, since InputImageModel doesn't store them.
    // If an image's dimensions can't be determined (e.g. its source
    // file was moved after being added to the project), it's
    // conservatively included as a contributor rather than silently
    // excluded - the safe direction for a culling step.
    std::vector<int> FindOverlappingImages(const ChunkModel& chunk, const std::vector<InputImageModel>& images);
}
