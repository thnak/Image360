#include "CppUnitTest.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include <Windows.h>
#include <algorithm>
#include <string>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    namespace
    {
        // Returns a fresh temp directory + a project base name, and
        // removes any leftover project file from a prior failed run.
        std::wstring MakeTempProjectDir(const wchar_t* suffix)
        {
            wchar_t tempDir[MAX_PATH];
            GetTempPathW(MAX_PATH, tempDir);
            std::wstring dir = std::wstring(tempDir) + L"Image360Test_se_" + suffix;
            CreateDirectoryW(dir.c_str(), nullptr);
            return dir;
        }

        void CleanupDir(const std::wstring& dir)
        {
            WIN32_FIND_DATAW findData;
            std::wstring pattern = dir + L"\\*";
            HANDLE find = FindFirstFileW(pattern.c_str(), &findData);
            if (find != INVALID_HANDLE_VALUE)
            {
                do
                {
                    std::wstring name = findData.cFileName;
                    if (name != L"." && name != L"..")
                        DeleteFileW((dir + L"\\" + name).c_str());
                } while (FindNextFileW(find, &findData));
                FindClose(find);
            }
            RemoveDirectoryW(dir.c_str());
        }
    }

    // Covers docs/superpowers/plans/2026-07-07-storage-engine.md Task 2:
    // WriteBlob/ReadBlob round-trip, multi-blob offset bookkeeping,
    // blob_directory integration, and reopen-resumes-at-EOF.
    TEST_CLASS(StorageEngineTests)
    {
    public:
        TEST_METHOD(RoundTripSingleBlob)
        {
            using namespace WindowsApp::Core;

            std::wstring dir = MakeTempProjectDir(L"roundtrip");
            std::wstring vfpPath = dir + L"\\project.vfp";
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(vfpPath, 1024, 1024));

            StorageEngine storage;
            Assert::IsTrue(storage.Open(dir, L"project", pm));

            std::vector<uint8_t> original = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            auto blobId = storage.WriteBlob(original.data(), original.size(), "test_bytes");
            Assert::IsTrue(blobId.has_value());
            Assert::IsTrue(blobId.value() > 0);

            auto readBack = storage.ReadBlob(blobId.value());
            Assert::IsTrue(readBack.has_value());
            Assert::AreEqual(original.size(), readBack->size());
            Assert::IsTrue(std::equal(original.begin(), original.end(), readBack->begin()));

            storage.Close();
            pm.CloseProject();
            CleanupDir(dir);
        }

        TEST_METHOD(MultipleBlobsDoNotCrossContaminate)
        {
            using namespace WindowsApp::Core;

            std::wstring dir = MakeTempProjectDir(L"multi");
            std::wstring vfpPath = dir + L"\\project.vfp";
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(vfpPath, 1024, 1024));

            StorageEngine storage;
            Assert::IsTrue(storage.Open(dir, L"project", pm));

            std::vector<uint8_t> a = { 10, 20, 30 };
            std::vector<uint8_t> b = { 40, 50, 60, 70 };
            std::vector<uint8_t> c = { 80 };

            auto idA = storage.WriteBlob(a.data(), a.size(), "a");
            auto idB = storage.WriteBlob(b.data(), b.size(), "b");
            auto idC = storage.WriteBlob(c.data(), c.size(), "c");
            Assert::IsTrue(idA.has_value() && idB.has_value() && idC.has_value());

            auto readA = storage.ReadBlob(idA.value());
            auto readB = storage.ReadBlob(idB.value());
            auto readC = storage.ReadBlob(idC.value());

            Assert::IsTrue(readA.has_value() && std::equal(a.begin(), a.end(), readA->begin()));
            Assert::IsTrue(readB.has_value() && std::equal(b.begin(), b.end(), readB->begin()));
            Assert::IsTrue(readC.has_value() && std::equal(c.begin(), c.end(), readC->begin()));

            storage.Close();
            pm.CloseProject();
            CleanupDir(dir);
        }

        TEST_METHOD(BlobDirectoryRowMatchesWrittenData)
        {
            using namespace WindowsApp::Core;

            std::wstring dir = MakeTempProjectDir(L"directory");
            std::wstring vfpPath = dir + L"\\project.vfp";
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(vfpPath, 1024, 1024));

            StorageEngine storage;
            Assert::IsTrue(storage.Open(dir, L"project", pm));

            std::vector<uint8_t> data = { 1, 2, 3, 4, 5 };
            auto blobId = storage.WriteBlob(data.data(), data.size(), "my_format_tag");
            Assert::IsTrue(blobId.has_value());

            auto entry = pm.GetBlobDirectoryEntry(blobId.value());
            Assert::IsTrue(entry.has_value());
            Assert::AreEqual(std::string("my_format_tag"), entry->formatTag);
            Assert::AreEqual(int64_t(data.size()), entry->length);

            storage.Close();
            pm.CloseProject();
            CleanupDir(dir);
        }

        TEST_METHOD(ReopenResumesAtEndOfFile)
        {
            using namespace WindowsApp::Core;

            std::wstring dir = MakeTempProjectDir(L"reopen");
            std::wstring vfpPath = dir + L"\\project.vfp";
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(vfpPath, 1024, 1024));

            std::vector<uint8_t> first = { 1, 2, 3 };
            std::vector<uint8_t> second = { 4, 5, 6, 7 };

            {
                StorageEngine storage;
                Assert::IsTrue(storage.Open(dir, L"project", pm));
                auto id1 = storage.WriteBlob(first.data(), first.size(), "first");
                Assert::IsTrue(id1.has_value());
                storage.Close();
            }

            {
                StorageEngine storage;
                Assert::IsTrue(storage.Open(dir, L"project", pm));
                auto id2 = storage.WriteBlob(second.data(), second.size(), "second");
                Assert::IsTrue(id2.has_value());

                // The second write must not have overwritten the first.
                auto entry1 = pm.GetBlobDirectoryEntry(id2.value() - 1);
                auto readBack1 = storage.ReadBlob(id2.value() - 1);
                Assert::IsTrue(readBack1.has_value());
                Assert::IsTrue(std::equal(first.begin(), first.end(), readBack1->begin()));

                auto readBack2 = storage.ReadBlob(id2.value());
                Assert::IsTrue(readBack2.has_value());
                Assert::IsTrue(std::equal(second.begin(), second.end(), readBack2->begin()));

                storage.Close();
            }

            pm.CloseProject();
            CleanupDir(dir);
        }
    };
}
