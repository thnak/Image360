#include "CppUnitTest.h"
#include "PanoramaExporter.h"
#include <cmath>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    // Covers docs/superpowers/plans/2026-07-07-nvjpeg-export.md Task 2
    // Step 5: composite placement/scale math is pure arithmetic,
    // testable without a GPU. Encode correctness itself needs real
    // hardware, same standing caveat as every GPU-touching plan.
    TEST_CLASS(PanoramaExporterTests)
    {
    public:
        TEST_METHOD(ComputeExportScaleShrinksToFitLongestEdge)
        {
            using namespace WindowsApp::Core;

            // 8192x4096 panorama, longest edge 8192 -> target 2048.
            float scale = ComputeExportScale(8192, 4096, 2048);
            Assert::IsTrue(std::fabs(scale - 0.25f) < 1e-5f);
        }

        TEST_METHOD(ComputeExportScaleHandlesTallerThanWide)
        {
            using namespace WindowsApp::Core;

            // Longest edge is height (4096), not width (2048).
            float scale = ComputeExportScale(2048, 4096, 1024);
            Assert::IsTrue(std::fabs(scale - 0.25f) < 1e-5f);
        }

        TEST_METHOD(ComputeChunkPlacementScalesOffsetAndSize)
        {
            using namespace WindowsApp::Core;

            ChunkModel chunk;
            chunk.x_offset = 4096;
            chunk.y_offset = 8192;
            chunk.width = 4096;
            chunk.height = 4096;

            ChunkPlacement placement = ComputeChunkPlacement(chunk, 0.25f);

            Assert::AreEqual(1024, placement.destX);
            Assert::AreEqual(2048, placement.destY);
            Assert::AreEqual(1024, placement.destW);
            Assert::AreEqual(1024, placement.destH);
        }

        TEST_METHOD(ComputeChunkPlacementNeverProducesZeroSizedTile)
        {
            using namespace WindowsApp::Core;

            // A very small scale factor could round a chunk's placement
            // down to 0x0 - must clamp to at least 1x1.
            ChunkModel chunk;
            chunk.x_offset = 0;
            chunk.y_offset = 0;
            chunk.width = 100;
            chunk.height = 100;

            ChunkPlacement placement = ComputeChunkPlacement(chunk, 0.001f);

            Assert::IsTrue(placement.destW >= 1);
            Assert::IsTrue(placement.destH >= 1);
        }
    };
}
