#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import sys
import unittest


SCRIPT = pathlib.Path(__file__).with_name("renju15_sm120_benchmark_matrix.py")
SPEC = importlib.util.spec_from_file_location("benchmark_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


class BenchmarkMatrixTest(unittest.TestCase):
    def test_preflight_only_is_an_explicit_cli_mode(self):
        args = matrix.parse_args(
            [
                "--binary", "katago",
                "--config", "test.cfg",
                "--model-b36c384", "b36.bin.gz",
                "--model-b24c256", "b24c256.bin.gz",
                "--model-b24c384", "b24c384.bin.gz",
                "--result-dir", "results",
                "--preflight-only",
            ]
        )
        self.assertTrue(args.preflight_only)
        self.assertFalse(args.skip_preflight)

    def test_schedule_is_position_balanced(self):
        matrix.validate_schedule()
        for position in range(6):
            self.assertEqual(
                set(matrix.CASE_IDS),
                {round_cases[position] for round_cases in matrix.BALANCED_ROUNDS},
            )

    def test_argv_uses_real_single_dash_cli_and_s2_override(self):
        argv = matrix.build_argv(
            pathlib.Path("/x/katago"),
            pathlib.Path("/x/test.cfg"),
            pathlib.Path("/x/model.bin.gz"),
            matrix.CASES["A"],
            30,
            150,
            0,
        )
        self.assertIn("-model", argv)
        self.assertIn("-json", argv)
        self.assertFalse(any(arg.startswith("--") for arg in argv))
        override = argv[argv.index("-override-config") + 1]
        self.assertIn("nnMaxBatchSize=28", override)
        self.assertIn("numNNServerThreadsPerModel=2", override)
        self.assertIn("gpuToUseThread0=0", override)
        self.assertIn("gpuToUseThread1=0", override)
        self.assertIn("maxBoardSizeForNNBuffer=15", override)
        self.assertIn("nnCacheSizePowerOfTwo=-1", override)
        self.assertIn("cudaUseINT8=false", override)

    def test_mask_only_changes_case_f_flag(self):
        for case_id, case in matrix.CASES.items():
            argv = matrix.build_argv(
                pathlib.Path("katago"),
                pathlib.Path("cfg"),
                pathlib.Path("model"),
                case,
                3,
                5,
                0,
            )
            self.assertEqual("-force-mask-all-ones" in argv, case_id == "F")

    def test_parse_last_json_line_skips_non_json_and_bad_json(self):
        parsed = matrix.parse_last_json_line(
            'marker\n{bad}\n{"combinedNNEvalsPerSec": 6950.0}\n', "leg"
        )
        self.assertEqual(6950.0, parsed["combinedNNEvalsPerSec"])

    def test_summaries_and_paired_ratios(self):
        report = {"formalLegs": []}
        for round_index, round_cases in enumerate(matrix.BALANCED_ROUNDS):
            for position, case_id in enumerate(round_cases):
                throughput = 100.0 * (ord(case_id) - ord("A") + 1) + round_index
                report["formalLegs"].append(
                    {
                        "roundKey": f"r{round_index}",
                        "caseId": case_id,
                        "position": position,
                        "json": {"combinedNNEvalsPerSec": throughput},
                    }
                )
        matrix.summarize_report(report, 5.0, 10.0)
        self.assertEqual(6, report["caseSummaries"]["A"]["count"])
        self.assertEqual(1.0, report["pairedRatiosVsA"]["A"]["caseOverAGeometricMean"])
        expected = math.exp(
            sum(math.log((200.0 + i) / (100.0 + i)) for i in range(6)) / 6.0
        )
        self.assertAlmostEqual(
            expected,
            report["pairedRatiosVsA"]["B"]["caseOverAGeometricMean"],
        )

    def test_validate_exact_marker_and_json_contract(self):
        case = matrix.CASES["C"]
        result = {
            "modelFile": "model.bin.gz",
            "revision": "test+CUDA",
            "batchSize": 28,
            "numServerThreads": 2,
            "numIterations": 5,
            "forceMaskAllOnes": False,
            "gpuIdxs": [0],
            "perServerMedianMs": [1.0, 1.1],
            "perServerNNEvalsPerSec": [100.0, 101.0],
            "combinedWallSeconds": 1.0,
            "combinedNNEvalsPerSec": 200.0,
        }
        markers = [
            "benchmarknn board size 15",
            "After dedups: nnModelFile0 = model.bin.gz useFP16 true useNHWC true useINT8 false",
            "Initializing neural net buffer to be size 15 * 15 exactly",
        ]
        for marker, minimum in matrix.required_markers(case):
            markers.extend([marker] * minimum)
        matrix.validate_benchmark(case, result, "\n".join(markers), 5, 0, "C")


if __name__ == "__main__":
    unittest.main()
