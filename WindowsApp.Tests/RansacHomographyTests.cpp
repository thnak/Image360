#include "CppUnitTest.h"
#include "RansacHomography.h"
#include <cmath>
#include <random>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    // Covers docs/superpowers/plans/2026-07-07-align-stage.md Task 5.
    // RunRansacHomography now solves via HomographyMath's portable CPU
    // DLT (no CudaPipeline/GPU involved at all - see the CPU compute
    // backend work), so it's fully testable here, not just
    // ReprojectionError.
    TEST_CLASS(RansacHomographyTests)
    {
    public:
        TEST_METHOD(ReprojectionErrorIsZeroForIdentityHomographyMatchingPoint)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            Homography identity; // default-constructed = identity
            FeaturePoint src{ 10.0f, 20.0f };
            FeaturePoint dst{ 10.0f, 20.0f };

            float error = ReprojectionError(identity, src, dst);
            Assert::IsTrue(error < 1e-3f);
        }

        TEST_METHOD(ReprojectionErrorMatchesKnownTranslation)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            // Pure translation homography: x' = x + 5, y' = y + 7.
            Homography h;
            h.h = { 1.0f, 0.0f, 5.0f,
                    0.0f, 1.0f, 7.0f,
                    0.0f, 0.0f, 1.0f };

            FeaturePoint src{ 10.0f, 20.0f };
            FeaturePoint dstExact{ 15.0f, 27.0f };
            FeaturePoint dstOff{ 25.0f, 27.0f }; // 10px off in x

            float errorExact = ReprojectionError(h, src, dstExact);
            float errorOff = ReprojectionError(h, src, dstOff);

            Assert::IsTrue(errorExact < 1e-3f);
            Assert::IsTrue(std::fabs(errorOff - 10.0f) < 1e-3f);
        }

        TEST_METHOD(ReprojectionErrorHandlesDegenerateHomography)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            // h[6..8] = 0 makes the homogeneous denominator zero for any
            // point - must not divide by zero or crash.
            Homography degenerate;
            degenerate.h = { 1.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 0.0f };

            FeaturePoint src{ 1.0f, 1.0f };
            FeaturePoint dst{ 1.0f, 1.0f };

            float error = ReprojectionError(degenerate, src, dst);
            Assert::IsTrue(error > 1e6f); // sentinel "very large", not NaN/inf-crash
        }

        TEST_METHOD(RunRansacHomographyRecoversKnownTranslation)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            // x' = x + 5, y' = y + 7 - simple, exactly-representable
            // homography so the recovered result should match tightly.
            Homography trueH;
            trueH.h = { 1.0f, 0.0f, 5.0f,
                        0.0f, 1.0f, 7.0f,
                        0.0f, 0.0f, 1.0f };

            std::mt19937 rng(123);
            std::uniform_real_distribution<float> coord(0.0f, 500.0f);

            std::vector<std::pair<FeaturePoint, FeaturePoint>> correspondences;
            for (int i = 0; i < 20; ++i)
            {
                FeaturePoint src{ coord(rng), coord(rng) };
                FeaturePoint dst{ src.x + 5.0f, src.y + 7.0f };
                correspondences.emplace_back(src, dst);
            }

            RansacResult result = RunRansacHomography(correspondences, 200, 1.0f);

            Assert::IsTrue(result.success);
            Assert::AreEqual(20, result.inlierCount);
            Assert::IsTrue(std::fabs(result.homography.h[2] - 5.0f) < 0.5f);
            Assert::IsTrue(std::fabs(result.homography.h[5] - 7.0f) < 0.5f);
        }

        TEST_METHOD(RunRansacHomographyRejectsOutliers)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            Homography trueH;
            trueH.h = { 1.0f, 0.0f, 5.0f,
                        0.0f, 1.0f, 7.0f,
                        0.0f, 0.0f, 1.0f };

            std::mt19937 rng(456);
            std::uniform_real_distribution<float> coord(0.0f, 500.0f);

            std::vector<std::pair<FeaturePoint, FeaturePoint>> correspondences;
            for (int i = 0; i < 20; ++i)
            {
                FeaturePoint src{ coord(rng), coord(rng) };
                FeaturePoint dst{ src.x + 5.0f, src.y + 7.0f };
                correspondences.emplace_back(src, dst);
            }
            // 5 gross outliers mixed in - RANSAC should still find the
            // 20-point consensus, not get pulled toward these.
            for (int i = 0; i < 5; ++i)
            {
                correspondences.emplace_back(FeaturePoint{ coord(rng), coord(rng) }, FeaturePoint{ coord(rng), coord(rng) });
            }

            RansacResult result = RunRansacHomography(correspondences, 500, 1.0f);

            Assert::IsTrue(result.success);
            Assert::IsTrue(result.inlierCount >= 20);
            Assert::IsTrue(std::fabs(result.homography.h[2] - 5.0f) < 0.5f);
            Assert::IsTrue(std::fabs(result.homography.h[5] - 7.0f) < 0.5f);
        }

        TEST_METHOD(RunRansacHomographyFailsWithFewerThanFourCorrespondences)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            std::vector<std::pair<FeaturePoint, FeaturePoint>> correspondences = {
                { FeaturePoint{ 0.0f, 0.0f }, FeaturePoint{ 5.0f, 7.0f } },
                { FeaturePoint{ 1.0f, 1.0f }, FeaturePoint{ 6.0f, 8.0f } },
            };

            RansacResult result = RunRansacHomography(correspondences);
            Assert::IsFalse(result.success);
        }
    };
}
