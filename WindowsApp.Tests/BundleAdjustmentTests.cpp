#include "CppUnitTest.h"
#include "BundleAdjustment.h"
#include <cmath>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    // Covers docs/superpowers/plans/2026-07-07-optimize-stage.md Task 5:
    // only checkpoint serialization is testable without a GPU -
    // RunOneLmIteration needs CudaPipeline::TensorSolveNormalEquations
    // and is only exercised end-to-end via OptimizeExecutor on real
    // hardware.
    TEST_CLASS(BundleAdjustmentTests)
    {
    public:
        TEST_METHOD(CheckpointRoundTrip)
        {
            using namespace WindowsApp::Core;

            BaCheckpoint original;
            original.iteration = 42;
            original.lambda = 0.00125f;
            original.parameters = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                     0.95f, 0.1f, -3.5f, -0.05f, 0.97f, 2.25f, 0.0001f, -0.0002f };

            std::string json = SerializeCheckpoint(original);

            BaCheckpoint restored;
            Assert::IsTrue(DeserializeCheckpoint(json, restored));

            Assert::AreEqual(original.iteration, restored.iteration);
            Assert::IsTrue(std::fabs(original.lambda - restored.lambda) < 1e-6f);
            Assert::AreEqual(original.parameters.size(), restored.parameters.size());
            for (size_t i = 0; i < original.parameters.size(); ++i)
            {
                Assert::IsTrue(std::fabs(original.parameters[i] - restored.parameters[i]) < 1e-4f);
            }
        }

        TEST_METHOD(CheckpointRoundTripEmptyParameters)
        {
            using namespace WindowsApp::Core;

            BaCheckpoint original;
            original.iteration = 0;
            original.lambda = 1e-3f;
            // parameters left empty - the "never started" / single-image-
            // project case.

            std::string json = SerializeCheckpoint(original);

            BaCheckpoint restored;
            Assert::IsTrue(DeserializeCheckpoint(json, restored));
            Assert::AreEqual(0, restored.iteration);
            Assert::IsTrue(restored.parameters.empty());
        }

        TEST_METHOD(DeserializeCheckpointRejectsMalformedInput)
        {
            using namespace WindowsApp::Core;
            BaCheckpoint out;

            Assert::IsFalse(DeserializeCheckpoint("", out));
            Assert::IsFalse(DeserializeCheckpoint("not json at all", out));
            Assert::IsFalse(DeserializeCheckpoint("{\"iteration\":5}", out)); // missing lambda/parameters
            Assert::IsFalse(DeserializeCheckpoint("{\"lambda\":0.1,\"parameters\":[1,2]}", out)); // missing iteration
            Assert::IsFalse(DeserializeCheckpoint("{\"iteration\":5,\"lambda\":0.1,\"parameters\":[1,2}", out)); // unterminated array
        }

        TEST_METHOD(HomographyFromBaParamsFixesH22ToOne)
        {
            using namespace WindowsApp::Core;

            float params[8] = { 1.0f, 0.1f, 5.0f, -0.1f, 0.9f, -3.0f, 0.0001f, -0.0002f };
            Homography h = HomographyFromBaParams(params);

            Assert::IsTrue(std::fabs(h.h[0] - 1.0f) < 1e-6f);
            Assert::IsTrue(std::fabs(h.h[1] - 0.1f) < 1e-6f);
            Assert::IsTrue(std::fabs(h.h[7] - (-0.0002f)) < 1e-6f);
            Assert::IsTrue(std::fabs(h.h[8] - 1.0f) < 1e-6f); // fixed
        }
    };
}
