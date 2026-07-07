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
    // doesn't ship yet - marks Inconclusive (not a failure, not a
    // silent skip) rather than asserting anything about GPU/RAW
    // correctness with no real file to test against.
    TEST_CLASS(ImageLoaderUnpackRawTests)
    {
    public:
        TEST_METHOD(UnpackRawReadsCfaPlaneAndMetadata)
        {
            using namespace WindowsApp::Core;

            if (!FixtureExists())
            {
                Assert::Inconclusive(
                    L"No fixture RAW file at WindowsApp.Tests\\fixtures\\sample.dng - "
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
    };
}
