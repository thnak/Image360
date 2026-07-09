#include "CppUnitTest.h"
#include "BundleAdjustment.h"
#include <cmath>
#include <string>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    // Covers docs/superpowers/plans/2026-07-07-optimize-stage.md Task 5.
    // RunOneLmIteration now solves via LinearSolve's portable CPU normal-
    // equations Cholesky solve (no CudaPipeline/GPU involved at all - see
    // the CPU compute backend work), so it's fully testable here, not
    // just checkpoint serialization.
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

        TEST_METHOD(RunOneLmIterationConvergesToKnownTranslation)
        {
            using namespace WindowsApp::Core;
            using namespace WindowsApp::Compute;

            // Image 0 is the fixed-identity reference; image 1's true
            // (unknown to the solver) homography is a pure translation
            // (dx=3, dy=-2): params = {h00,h01,h02,h10,h11,h12,h20,h21}.
            const float trueDx = 3.0f, trueDy = -2.0f;
            std::vector<int> nonReferenceImageIds = { 1 };

            std::vector<BaCorrespondence> correspondences;
            for (int i = 0; i < 10; ++i)
            {
                float x = static_cast<float>(i * 37 % 200);
                float y = static_cast<float>(i * 53 % 200);
                BaCorrespondence corr;
                corr.imageA = 0;
                corr.imageB = 1;
                corr.pointB = FeaturePoint{ x, y };
                // Reference image applies identity, so worldA = pointA must
                // equal worldB = pointB + (trueDx, trueDy) at convergence.
                corr.pointA = FeaturePoint{ x + trueDx, y + trueDy };
                correspondences.push_back(corr);
            }

            BaCheckpoint checkpoint;
            checkpoint.iteration = 0;
            checkpoint.lambda = 1e-3f;
            checkpoint.parameters = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f }; // identity guess

            bool converged = false;
            for (int iter = 0; iter < 50 && !converged; ++iter)
            {
                LmStepResult step = RunOneLmIteration(nonReferenceImageIds, correspondences, checkpoint);
                checkpoint = step.checkpoint;
                converged = step.converged;
            }

            Assert::IsTrue(converged);
            Assert::IsTrue(std::fabs(checkpoint.parameters[2] - trueDx) < 0.1f); // h02
            Assert::IsTrue(std::fabs(checkpoint.parameters[5] - trueDy) < 0.1f); // h12
        }

        TEST_METHOD(RunOneLmIterationHandlesNoNonReferenceImages)
        {
            using namespace WindowsApp::Core;

            std::vector<int> nonReferenceImageIds; // empty - single-image project
            std::vector<BaCorrespondence> correspondences;

            BaCheckpoint checkpoint;
            LmStepResult step = RunOneLmIteration(nonReferenceImageIds, correspondences, checkpoint);

            Assert::IsTrue(step.converged); // nothing to solve - immediately converged
        }
    };
}
