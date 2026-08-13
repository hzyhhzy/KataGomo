import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
POLICY = ROOT / "cpp" / "cmake" / "C384ExactFfnDownAotPolicy.cmake"
CMAKE_LISTS = ROOT / "cpp" / "CMakeLists.txt"
POLICY_CONTRACT = pathlib.Path(__file__).with_name(
    "c384_b28_production_policy_contract.cmake"
)
FIXED_REGISTRY_HEADER = (
    ROOT / "cpp" / "neuralnet" / "cuda_specialized" / "sm120" /
    "c384" / "fixed_batch" / "kernels.h"
)
CUDA_BACKEND = ROOT / "cpp" / "neuralnet" / "cudabackend.cpp"
CUDA_HELPERS = ROOT / "cpp" / "neuralnet" / "cudahelpers.cu"
NEURALNET = ROOT / "cpp" / "neuralnet"


class C384B28ProductionPolicyTests(unittest.TestCase):
    def run_cmake(self, case: str, expect_success: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["cmake", f"-DCASE={case}", "-P", str(POLICY_CONTRACT)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if expect_success and result.returncode != 0:
            self.fail(result.stdout)
        if not expect_success and result.returncode == 0:
            self.fail("CMake policy unexpectedly accepted invalid input")
        return result

    def test_valid_down_manifest_resolves_exact_b28_id(self) -> None:
        self.run_cmake("valid-down")

    def test_b24_down_manifest_is_rejected(self) -> None:
        result = self.run_cmake("b24-down", expect_success=False)
        self.assertIn("B28/M6300", result.stdout)

    def test_package_cannot_shadow_the_engine_owned_raw_abi(self) -> None:
        result = self.run_cmake("shadowed-abi", expect_success=False)
        self.assertIn("shadows", result.stdout)

    def test_full_production_bundle_must_be_complete_and_b28(self) -> None:
        result = self.run_cmake("incomplete-production", expect_success=False)
        self.assertIn("FFN-down provider is required", result.stdout)
        self.run_cmake("complete-production")

    def test_search_pair_without_down_remains_a_search_configuration(self) -> None:
        self.run_cmake("search-pair")

    def test_cmake_uses_a_separate_down_provider(self) -> None:
        source = CMAKE_LISTS.read_text(encoding="utf-8")
        self.assertIn("c384/fixed_batch/ffn_down_registry.cu", source)
        self.assertIn("c384/fixed_batch/ffn_down_registry_stub.cu", source)
        self.assertIn("KATAGO_C384_FFN_DOWN_EFFECTIVE_REGISTRY_PROVIDER", source)
        self.assertIn('"${CMAKE_CURRENT_SOURCE_DIR}/neuralnet"', source)
        fixed_registry = FIXED_REGISTRY_HEADER.read_text(encoding="utf-8")
        self.assertNotIn("FfnDown", fixed_registry)

    def test_standard_sm120_build_does_not_enable_int8_experiment(self) -> None:
        source = CMAKE_LISTS.read_text(encoding="utf-8")
        declaration = source.split(
            "set(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT", 1,
        )[1].split(")", 1)[0]
        self.assertIn("\n  0 CACHE BOOL", declaration)
        self.assertNotIn("${KATAGO_ENABLE_SM120_TRANSFORMER_WINNER}", declaration)

    def test_exact_transaction_markers_report_typed_dynamic_depth(self) -> None:
        source = CUDA_BACKEND.read_text(encoding="utf-8")
        transaction = source.split("void configureC384ExactFixedAot", 1)[1]
        transaction = transaction.split(
            "#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT)", 1,
        )[0]
        self.assertNotIn("attentionCount != 36", transaction)
        self.assertNotIn("preparedC384ExactQkvFa4 == 36", transaction)
        self.assertNotIn("activeC384ExactQkvFa4 == 36", transaction)
        self.assertNotIn("/36", transaction)
        self.assertIn("KATAGO_C384_EXACT_FIXED_PREPARED batch=", transaction)
        self.assertIn("KATAGO_C384_EXACT_FIXED_ACTIVE batch=", transaction)
        self.assertGreaterEqual(transaction.count(' + " depth=" +'), 2)
        self.assertIn("c384ExactAttentionBlockCount", transaction)
        self.assertIn("c384ExactFfnBlockCount", transaction)

    def test_clip7_dual_semantics_do_not_reuse_unclipped_c256_kernel(self) -> None:
        source = CUDA_BACKEND.read_text(encoding="utf-8")
        ffn = source.split("struct TransformerFFNBlock", 1)[1]
        ffn = ffn.split("//------------------------------------------------------------------------------", 1)[0]
        constructor = ffn.split("TransformerFFNBlock(", 1)[1]
        constructor = constructor.split("void apply(", 1)[0]
        self.assertIn("c256DualSemanticSupported", constructor)
        self.assertIn("c384DualSemanticSupported", constructor)
        c256_create = constructor.split(
            "Sm120C256F768M128N64K32S3Sw4", 1,
        )[1].split("dualFfnKernel =", 1)[0]
        self.assertIn("swigluClip == 0.0f", c256_create)

        apply = ffn.split("void apply(", 1)[1]
        apply = apply.split("// Step 4-5", 1)[0]
        self.assertIn("specializedDualFfnSemanticSupported", apply)
        self.assertIn("!c384DualFfn && swigluClip == 0.0f", apply)
        self.assertNotIn(
            "!usedDualFfn && specializedDualFfn && c384DualFfnRuntimeAllowed &&\n"
            "       (swigluClip == 0.0f || swigluClip == 7.0f)",
            apply,
        )

    def test_generic_clip7_propagates_nan_like_exact_epilogue(self) -> None:
        source = CUDA_HELPERS.read_text(encoding="utf-8")
        clamp = source.split(
            "__device__ __forceinline__ float clampSymmetric", 1,
        )[1].split("}", 1)[0]
        self.assertIn(
            "value > limit ? limit : (value < -limit ? -limit : value)",
            clamp,
        )
        return_line = clamp.split("return", 1)[1]
        self.assertNotIn("fminf", return_line)
        self.assertNotIn("fmaxf", return_line)

    def test_specialized_layout_keeps_generated_include_compatibility(self) -> None:
        forwards = {
            "c384_exact_fixed_aot_kernels.h":
                "cuda_specialized/sm120/c384/fixed_batch/kernels.h",
            "c384_exact_ffn_down_aot.h":
                "cuda_specialized/sm120/c384/fixed_batch/ffn_down.h",
            "c384_h12_fa4_sm120.h":
                "cuda_specialized/sm120/c384/fixed_batch/fa4.h",
        }
        for legacy_name, canonical_rel in forwards.items():
            legacy = (NEURALNET / legacy_name).read_text(encoding="utf-8")
            canonical = (NEURALNET / canonical_rel).read_text(encoding="utf-8")
            self.assertIn(f'#include "{canonical_rel}"', legacy)
            legacy_guard = legacy.splitlines()[0]
            canonical_guard = canonical.splitlines()[0]
            self.assertNotEqual(legacy_guard, canonical_guard)


if __name__ == "__main__":
    unittest.main()
