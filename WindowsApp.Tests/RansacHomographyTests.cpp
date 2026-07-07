#include "CppUnitTest.h"
#include "RansacHomography.h"
#include <cmath>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    // Covers docs/superpowers/plans/2026-07-07-align-stage.md Task 5:
    // only ReprojectionError is testable without a GPU -
    // RunRansacHomography's orchestration needs
    // CudaPipeline::TensorEstimateHomography and is only exercised
    // end-to-end via AlignExecutor on real hardware.
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
    };
}
