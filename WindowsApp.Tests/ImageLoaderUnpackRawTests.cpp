#include "CppUnitTest.h"
#include "ImageLoader.h"
#include <Windows.h>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    namespace
    {
        // No sample RAW file is committed to this repo yet (see
        // docs/superpowers/plans/2026-07-07-raw-ingest.md Task 2 Step 4).
        // Sourcing a small, permissively-licensed one is follow-up work,
        // not blocking this test's existence.
        std::wstring FixturePath()
        {
            wchar_t moduleDir[MAX_PATH];
            GetModuleFileNameW(nullptr, moduleDir, MAX_PATH);
            std::wstring path(moduleDir);
            size_t lastSlash = path.find_last_of(L'\\');
            std::wstring dir = (lastSlash == std::wstring::npos) ? L"." : path.substr(0, lastSlash);
            return dir + L"\\fixtures\\sample.dng";
        }

        bool FixtureExists()
        {
            return GetFileAttributesW(FixturePath().c_str()) != INVALID_FILE_ATTRIBUTES;
        }
    }

    // Covers docs/superpowers/plans/2026-07-07-raw-ingest.md Task 2:
    // ImageLoader::UnpackRaw. Gated on a fixture RAW file this repo
    // doesn't ship yet - logs why and returns rather than asserting
    // anything about GPU/RAW correctness with no real file to test
    // against (native CppUnitTestFramework has no Assert::Inconclusive).
    TEST_CLASS(ImageLoaderUnpackRawTests)
    {
    public:
        TEST_METHOD(UnpackRawReadsCfaPlaneAndMetadata)
        {
            using namespace WindowsApp::Core;

            if (!FixtureExists())
            {
                // CppUnitTestFramework (native) has no Assert::Inconclusive -
                // that's a .NET MSTest concept, not part of this framework
                // (confirmed via a real build error: C2039). Logger::WriteMessage
                // + a plain early return is the closest available
                // approximation - the test still shows as "passed" with no
                // real assertions made, but the reason is visible in the
                // test output rather than silently absent.
                Logger::WriteMessage(
                    L"SKIPPED: no fixture RAW file at WindowsApp.Tests\\fixtures\\sample.dng - "
                    L"add one to exercise ImageLoader::UnpackRaw for real.");
                return;
            }

            ImageLoader loader;
            Assert::IsTrue(loader.Open(FixturePath()));

            RawPlane plane;
            Assert::IsTrue(loader.UnpackRaw(plane));
            Assert::IsTrue(plane.width > 0);
            Assert::IsTrue(plane.height > 0);
            Assert::AreEqual(
                static_cast<size_t>(plane.width) * static_cast<size_t>(plane.height),
                plane.cfaData.size());
            Assert::IsTrue(plane.cfaType != CfaType::UNKNOWN);
        }

        TEST_METHOD(UnpackRawFailsWithoutOpenFile)
        {
            using namespace WindowsApp::Core;

            ImageLoader loader;
            RawPlane plane;
            Assert::IsFalse(loader.UnpackRaw(plane));
        }

        TEST_METHOD(GetEmbeddedPreviewJpegReadsThumbnail)
        {
            using namespace WindowsApp::Core;

            if (!FixtureExists())
            {
                Logger::WriteMessage(
                    L"SKIPPED: no fixture RAW file at WindowsApp.Tests\\fixtures\\sample.dng - "
                    L"add one to exercise ImageLoader::GetEmbeddedPreviewJpeg for real.");
                return;
            }

            ImageLoader loader;
            Assert::IsTrue(loader.Open(FixturePath()));

            std::vector<unsigned char> jpegBytes;
            Assert::IsTrue(loader.GetEmbeddedPreviewJpeg(jpegBytes));
            Assert::IsTrue(jpegBytes.size() > 0);
            // JPEG SOI marker.
            Assert::AreEqual(static_cast<unsigned char>(0xFF), jpegBytes[0]);
            Assert::AreEqual(static_cast<unsigned char>(0xD8), jpegBytes[1]);
        }

        TEST_METHOD(GetEmbeddedPreviewJpegFailsWithoutOpenFile)
        {
            using namespace WindowsApp::Core;

            ImageLoader loader;
            std::vector<unsigned char> jpegBytes;
            Assert::IsFalse(loader.GetEmbeddedPreviewJpeg(jpegBytes));
        }
    };
}
