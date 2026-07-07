#include "CppUnitTest.h"
#include "OverlapCulling.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    // Covers docs/superpowers/plans/2026-07-07-render-stage.md Task 1.
    // ImageOverlapsChunk is pure (no I/O) and fully testable here.
    // FindOverlappingImages needs real files (ImageLoader::GetMetadata)
    // to determine each image's footprint - no fixture RAW file is
    // committed to this repo yet (same standing gap as
    // ImageLoaderUnpackRawTests), so only its documented conservative
    // fallback ("can't determine dimensions -> include") is tested here.
    TEST_CLASS(OverlapCullingTests)
    {
    public:
        TEST_METHOD(IdentityHomographyOverlapsChunkAtOrigin)
        {
            using namespace WindowsApp::Core;

            Homography identity;
            ChunkModel chunk;
            chunk.x_offset = 0;
            chunk.y_offset = 0;
            chunk.width = 4096;
            chunk.height = 4096;

            Assert::IsTrue(ImageOverlapsChunk(identity, 1000, 800, chunk));
        }

        TEST_METHOD(TranslatedFarAwayDoesNotOverlap)
        {
            using namespace WindowsApp::Core;

            Homography translated;
            translated.h = { 1.0f, 0.0f, 100000.0f,
                              0.0f, 1.0f, 100000.0f,
                              0.0f, 0.0f, 1.0f };

            ChunkModel chunk;
            chunk.x_offset = 0;
            chunk.y_offset = 0;
            chunk.width = 4096;
            chunk.height = 4096;

            Assert::IsFalse(ImageOverlapsChunk(translated, 1000, 800, chunk));
        }

        TEST_METHOD(PartialOverlapAtChunkBoundaryCounts)
        {
            using namespace WindowsApp::Core;

            // Image placed so only its bottom-right corner touches the
            // chunk at (4000,4000)-(8096,8096) - footprint spans
            // (3500,3500)-(4500,4500), overlapping the chunk's top-left corner.
            Homography translated;
            translated.h = { 1.0f, 0.0f, 3500.0f,
                              0.0f, 1.0f, 3500.0f,
                              0.0f, 0.0f, 1.0f };

            ChunkModel chunk;
            chunk.x_offset = 4000;
            chunk.y_offset = 4000;
            chunk.width = 4096;
            chunk.height = 4096;

            Assert::IsTrue(ImageOverlapsChunk(translated, 1000, 1000, chunk));
        }

        TEST_METHOD(AdjacentNonOverlappingImageIsExcluded)
        {
            using namespace WindowsApp::Core;

            // Footprint spans (0,0)-(1000,1000); chunk starts exactly
            // where the image ends - AABBs touch but don't overlap
            // (strict inequality), so this must be excluded.
            Homography identity;
            ChunkModel chunk;
            chunk.x_offset = 1000;
            chunk.y_offset = 0;
            chunk.width = 4096;
            chunk.height = 4096;

            Assert::IsFalse(ImageOverlapsChunk(identity, 1000, 1000, chunk));
        }

        TEST_METHOD(FindOverlappingImagesConservativelyIncludesUndeterminableFootprints)
        {
            using namespace WindowsApp::Core;

            ChunkModel chunk;
            chunk.x_offset = 0;
            chunk.y_offset = 0;
            chunk.width = 4096;
            chunk.height = 4096;

            std::vector<InputImageModel> images;
            InputImageModel img1;
            img1.id = 1;
            img1.file_path = L"C:\\this\\file\\does\\not\\exist_1.dng";
            images.push_back(img1);

            InputImageModel img2;
            img2.id = 2;
            img2.file_path = L"C:\\this\\file\\does\\not\\exist_2.dng";
            images.push_back(img2);

            auto result = FindOverlappingImages(chunk, images);

            // Neither file can be opened, so dimensions can't be
            // determined - both are conservatively included.
            Assert::AreEqual(size_t(2), result.size());
        }
    };
}
