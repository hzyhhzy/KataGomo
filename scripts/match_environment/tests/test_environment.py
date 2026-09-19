import copy
import csv
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import auto_match_config as config
from calculate_elo import calculate, calculate_all, collect_results, fit_elo, normalize_name
from match_tools import check_assets, pair_state, result_for_pair, verify_pair, write_json
from script_generator import generate
from types import SimpleNamespace


FAKE_ENGINE = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
if os.environ.get("FAKE_FAIL"):
    sys.exit(7)
args = sys.argv[1:]
assert args[0] == "match"
params = {}
for i, arg in enumerate(args):
    if arg == "-override-config":
        k, v = args[i+1].split("=", 1)
        params[k] = v
for k in ("nnModelFile0", "nnModelFile1", "matchOpeningFile"):
    if params[k]:
        assert pathlib.Path(params[k]).is_file(), params[k]
assert params["cudaUseINT8"] == "false"
n = int(params["numGamesTotal"])
pathlib.Path("matchresult").mkdir()
pathlib.Path("matchresult/result.json").write_text(json.dumps({
    "bot0name": params["botName0"], "bot1name": params["botName1"],
    "total": n, "win0": n // 2, "lose0": n - n // 2, "draw": 0,
    "params": params, "visible": os.environ["CUDA_VISIBLE_DEVICES"],
    "libs": os.environ.get("LD_LIBRARY_PATH", ""),
}))
'''


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ab-match-tests-")
        self.root = Path(self.temp.name) / "folder with 'quotes' $dollar"
        self.root.mkdir()
        self.cfg = SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(config).items() if k.isupper()})
        self.cfg.GAMES_PER_PAIR = 4
        for name in ("match_tools.py", "calculate_elo.py", "script_generator.py", "auto_match_config.py", "match.cfg"):
            shutil.copyfile(SOURCE / name, self.root / name)
        for d in ("engine", "lib", "openings", "modelsGroupA/alpha", "modelsGroupB/beta"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        (self.root / "engine/katago").write_text(FAKE_ENGINE, encoding="utf-8", newline="\n")
        (self.root / "engine/katago").chmod(0o755)
        (self.root / "modelsGroupA/alpha/model.bin.gz").write_bytes(b"model-A")
        (self.root / "modelsGroupB/beta/model.bin.gz").write_bytes(b"model-B")
        (self.root / self.cfg.OPENING_FILE).write_text("15 H8 J8 H9 J9 G8\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def alias(self, group, name):
        (self.root / group / name).mkdir()

    def save_result(self, pair, **changes):
        data = {"bot0name": pair["a"], "bot1name": pair["b"], "total": pair["games"],
                "win0": 2, "lose0": 1, "draw": 1}
        data.update(changes)
        write_json(self.root / pair["directory"] / "matchresult" / "one.json", data)

    def test_cross_group_aliases_and_order(self):
        self.alias("modelsGroupA", "alpha_200po")
        self.alias("modelsGroupB", "beta_50po")
        self.alias("modelsGroupB", ".ipynb_checkpoints")
        (self.root / "modelsGroupA/alpha_200po/model.bin.gz").write_bytes(b"MUST NOT USE")
        plan = generate(self.root, self.cfg)
        self.assertEqual([(p["a"], p["b"]) for p in plan["pairs"]], [
            ("A/alpha_100po", "B/beta_100po"), ("A/alpha_100po", "B/beta_50po"),
            ("A/alpha_200po", "B/beta_100po"), ("A/alpha_200po", "B/beta_50po")])
        last = plan["pairs"][-1]
        self.assertEqual((last["playouts_a"], last["playouts_b"]), (200, 50))
        self.assertEqual(last["model_a"], "modelsGroupA/alpha/model.bin.gz")
        self.assertEqual(len(set(p["directory"] for p in plan["pairs"])), 4)

    def test_missing_base_rejected(self):
        self.alias("modelsGroupA", "missing_200po")
        with self.assertRaisesRegex(ValueError, "missing base"):
            generate(self.root, self.cfg)

    def test_eleven_each_group_optional_dedup_and_group_local_models(self):
        for group in ("modelsGroupA", "modelsGroupB"):
            for name in ("alpha", "beta") + tuple(f"model{i}" for i in range(9)):
                folder = self.root / group / name
                folder.mkdir(exist_ok=True)
                (folder / "model.bin.gz").write_bytes(name.encode())
        dedup = generate(self.root, self.cfg)
        self.assertEqual(len(dedup["pairs"]), 55)
        self.assertEqual(len(dedup["participants"]), 11)
        seen = set()
        for pair in dedup["pairs"]:
            self.assertNotEqual(pair["elo_a"], pair["elo_b"])
            key = frozenset((pair["elo_a"], pair["elo_b"]))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertTrue(pair["model_a"].startswith("modelsGroupA/"))
            self.assertTrue(pair["model_b"].startswith("modelsGroupB/"))
        self.cfg.DEDUPLICATE_MATCHES = False
        all_pairs = generate(self.root, self.cfg)
        self.assertEqual(len(all_pairs["pairs"]), 121)
        self.assertEqual(len(all_pairs["participants"]), 11)
        self.assertEqual(sum(p["elo_a"] == p["elo_b"] for p in all_pairs["pairs"]), 11)
        for pair in all_pairs["pairs"]:
            self.save_result(pair)
            verify_pair(self.root, pair)
        result = calculate(self.root, all_pairs)
        self.assertEqual(len(result["ratings"]), 11)
        self.assertEqual(result["self_play_games_excluded_from_ratings"], 44)
        self.assertEqual(len(result["matches"]), 121)
        self.assertEqual(sum(not row["included_in_elo"] for row in result["matches"]), 11)
        self.assertEqual(result["total_scheduled_games"], 484)

    def test_different_weights_same_name_are_the_same_participant(self):
        folder = self.root / "modelsGroupB/alpha"
        folder.mkdir()
        (folder / "model.bin.gz").write_bytes(b"different-from-A-alpha")
        self.assertEqual(len(generate(self.root, self.cfg)["pairs"]), 1)
        self.cfg.DEDUPLICATE_MATCHES = False
        plan = generate(self.root, self.cfg)
        self.assertEqual(len(plan["pairs"]), 2)
        self.assertEqual({p["id"] for p in plan["participants"]}, {"alpha_100po", "beta_100po"})

    def test_custom_default_playouts_recorded_names_and_explicit_alias(self):
        self.cfg.DEFAULT_PLAYOUTS = 150
        self.alias("modelsGroupA", "alpha_200po")
        plan = generate(self.root, self.cfg)
        self.assertEqual({p["id"] for p in plan["participants"]}, {"alpha_150po", "alpha_200po", "beta_150po"})
        self.assertEqual([(p["playouts_a"], p["playouts_b"]) for p in plan["pairs"]], [(150, 150), (200, 150)])
        script = (self.root / "run_matches.sh").read_text()
        self.assertIn("botName0=A/alpha_150po", script)
        self.assertIn("maxPlayouts0=150", script)
        self.assertIn('"$ROOT/calculate_elo.py"\n', script)
        self.assertNotIn('calculate_elo.py" --plan', script)

    def test_zero_alias_rejected(self):
        self.alias("modelsGroupA", "alpha_0po")
        with self.assertRaisesRegex(ValueError, "alias playouts"):
            generate(self.root, self.cfg)

    def test_missing_model_rejected(self):
        self.alias("modelsGroupB", "empty")
        with self.assertRaisesRegex(ValueError, "No supported model"):
            generate(self.root, self.cfg)

    def test_opening_empty_forms_and_missing(self):
        for empty in ("", '""', None, "  "):
            self.cfg.OPENING_FILE = empty
            self.assertEqual(generate(self.root, self.cfg)["opening_file"], "")
            self.assertIn("matchOpeningFile=", (self.root / "run_matches.sh").read_text())
        self.cfg.OPENING_FILE = "openings/missing.txt"
        with self.assertRaisesRegex(ValueError, "Missing asset"):
            generate(self.root, self.cfg)
        self.cfg.OPENING_FILE = "openings/empty.txt"
        (self.root / self.cfg.OPENING_FILE).write_text("# only comment\n")
        with self.assertRaisesRegex(ValueError, "empty"):
            generate(self.root, self.cfg)

    def test_gpu_mapping_and_overrides(self):
        self.cfg.VISIBLE_GPUS = [2, 5]
        self.cfg.EXTRA_OVERRIDES = {"basicRules": "FREESTYLE"}
        p = generate(self.root, self.cfg)
        self.assertEqual([p["overrides"][f"gpuToUseThread{i}"] for i in range(4)], [0, 1, 0, 1])
        self.assertEqual(p["overrides"]["basicRules"], "FREESTYLE")
        self.assertIn("CUDA_VISIBLE_DEVICES=2,5", (self.root / "run_matches.sh").read_text())
        self.cfg.EXTRA_OVERRIDES = {"maxPlayouts0": 1}
        with self.assertRaisesRegex(ValueError, "Reserved"):
            generate(self.root, self.cfg)

    def test_config_validation(self):
        for name, bad in (("GAMES_PER_PAIR", 0), ("GAME_THREADS", True), ("VISIBLE_GPUS", [0, 0]),
                          ("NN_CACHE_POWER", -2), ("MODEL_FILE_PREFERENCE", ["../a"]), ("GROUP_B", "modelsGroupA")):
            cfg = copy.deepcopy(self.cfg)
            setattr(cfg, name, bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                generate(self.root, cfg)
        self.cfg.EXTRA_OVERRIDES = {"key": "a,b"}
        with self.assertRaises(ValueError):
            generate(self.root, self.cfg)

    def test_native_preference_and_onnx(self):
        for group, model in (("modelsGroupA", "alpha"), ("modelsGroupB", "beta")):
            (self.root / group / model / "model.onnx").write_bytes(b"onnx")
        self.assertEqual(generate(self.root, self.cfg)["pairs"][0]["engine"], "engine/katago")
        self.cfg.MODEL_FILE_PREFERENCE = ["model.onnx", "model.bin.gz"]
        with self.assertRaisesRegex(ValueError, "katago_onnx"):
            generate(self.root, self.cfg)
        shutil.copyfile(self.root / "engine/katago", self.root / "engine/katago_onnx")
        self.assertEqual(generate(self.root, self.cfg)["pairs"][0]["engine"], "engine/katago_onnx")
        (self.root / "modelsGroupB/beta/model.onnx").unlink()
        with self.assertRaisesRegex(ValueError, "Cannot mix"):
            generate(self.root, self.cfg)

    def test_plan_stability_and_hash_check(self):
        first = generate(self.root, self.cfg)
        self.assertEqual(generate(self.root, self.cfg)["run_id"], first["run_id"])
        check_assets(self.root, first)
        (self.root / "modelsGroupA/alpha/model.bin.gz").write_bytes(b"new-model")
        with self.assertRaisesRegex(ValueError, "changed asset"):
            check_assets(self.root, first)
        self.assertNotEqual(generate(self.root, self.cfg)["run_id"], first["run_id"])

    def test_verified_completion_and_tampering(self):
        pair = generate(self.root, self.cfg)["pairs"][0]
        self.assertEqual(pair_state(self.root, pair), 1)
        self.save_result(pair)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            pair_state(self.root, pair)
        verify_pair(self.root, pair)
        self.assertEqual(pair_state(self.root, pair), 0)
        self.save_result(pair, win0=1, lose0=2)
        with self.assertRaisesRegex(ValueError, "differs"):
            pair_state(self.root, pair)

    def test_result_rejects_wrong_identity_partial_and_duplicates(self):
        pair = generate(self.root, self.cfg)["pairs"][0]
        for bad in ({"bot0name": "intruder"}, {"total": 3}, {"win0": True}, {"draw": -1}):
            self.save_result(pair, **bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                result_for_pair(self.root, pair)
        self.save_result(pair)
        write_json(self.root / pair["directory"] / "matchresult" / "two.json", {})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            result_for_pair(self.root, pair)

    def test_elo_requires_all_pairs_and_exports_all(self):
        self.alias("modelsGroupA", "alpha_200po")
        plan = generate(self.root, self.cfg)
        with self.assertRaisesRegex(ValueError, "not complete"):
            calculate(self.root, plan)
        for pair in plan["pairs"]:
            self.save_result(pair)
            verify_pair(self.root, pair)
        result = calculate(self.root, plan)
        self.assertEqual(len(result["ratings"]), 3)
        self.assertEqual(result["total_games"], 8)
        self.assertTrue((self.root / "results" / plan["run_id"] / "elo.csv").is_file())
        self.assertTrue((self.root / "results" / plan["run_id"] / "matches.csv").is_file())
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual({r["pair_id"] for r in result["matches"]}, {p["id"] for p in plan["pairs"]})

    def test_match_details_and_elo_table_without_bot_win_rates(self):
        records = [
            ("round1.json", "A/alpha", "B/beta", 7, 1, 2),
            ("round2.json", "B/beta", "A/alpha", 3, 5, 2),
            ("self.json", "A/alpha", "B/alpha", 1, 1, 2),
        ]
        for filename, a, b, wa, wb, draws in records:
            write_json(self.root / "matchresult" / filename,
                       {"bot0name": a, "bot1name": b, "total": wa + wb + draws,
                        "win0": wa, "lose0": wb, "draw": draws})
        console = io.StringIO()
        with redirect_stdout(console):
            result = calculate_all(self.root)
        self.assertEqual(result["total_games"], 20)
        self.assertEqual(result["total_scheduled_games"], 24)
        self.assertEqual(len(result["matches"]), 3)
        self.assertEqual(sum(row["included_in_elo"] for row in result["matches"]), 2)
        expected = [("alpha_100po", "beta_100po", 7, 1, 2),
                    ("beta_100po", "alpha_100po", 3, 5, 2),
                    ("alpha_100po", "alpha_100po", 1, 1, 2)]
        self.assertEqual([(r["bot_a"], r["bot_b"], r["wins_a"], r["wins_b"], r["draws"])
                          for r in result["matches"]], expected)
        self.assertTrue(all(set(r) == {"rank", "participant", "elo", "games"} for r in result["ratings"]))
        self.assertIn("Match details (3 result JSONs)", console.getvalue())
        self.assertIn("Rank Bot", console.getvalue())
        self.assertNotIn("score=", console.getvalue())
        self.assertNotIn("%", console.getvalue())
        with (self.root / "elo_all/elo.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["rank", "participant", "elo", "games"])
            self.assertEqual(len(list(reader)), 2)
        with (self.root / "elo_all/matches.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual([Path(r["result_file"]).name for r in rows], [r[0] for r in records])
        self.assertEqual(rows[-1]["included_in_elo"], "False")
        self.assertEqual(calculate_all(self.root)["result_files"], 3)  # Output JSON must not become input.

    def test_multiple_rounds_same_name_changed_weights_and_budgets(self):
        first = generate(self.root, self.cfg)
        self.save_result(first["pairs"][0])
        verify_pair(self.root, first["pairs"][0])
        (self.root / "modelsGroupA/alpha/model.bin.gz").write_bytes(b"new weights same name")
        second = generate(self.root, self.cfg)
        self.save_result(second["pairs"][0], win0=1, lose0=2)
        verify_pair(self.root, second["pairs"][0])
        result = calculate_all(self.root)
        self.assertEqual(result["total_games"], 8)
        self.assertEqual(result["result_files"], 2)
        self.assertEqual({r["participant"] for r in result["ratings"]}, {"alpha_100po", "beta_100po"})
        self.assertTrue(all(abs(r["elo"]) < 1e-6 for r in result["ratings"]))
        # Both old rounds remain readable without the old weights, and without any model files.
        (self.root / "modelsGroupA/alpha/model.bin.gz").unlink()
        self.assertEqual(calculate_all(self.root)["total_games"], 8)

    def test_pooled_results_skip_incomplete_and_refuse_tampered_complete(self):
        self.alias("modelsGroupA", "alpha_200po")
        plan = generate(self.root, self.cfg)
        self.save_result(plan["pairs"][0])
        verify_pair(self.root, plan["pairs"][0])
        self.save_result(plan["pairs"][1], total=1, win0=1, lose0=0, draw=0)
        result = calculate_all(self.root)
        self.assertEqual(result["total_games"], 4)
        self.assertEqual(len(result["skipped_files"]), 1)
        self.save_result(plan["pairs"][0], win0=1, lose0=2)
        with self.assertRaisesRegex(ValueError, "differs"):
            calculate_all(self.root)

    def test_loose_mixed_json_names_default_suffix_and_copies(self):
        first = {"bot0name": "A/alpha", "bot1name": "B/beta_200po", "total": 10,
                 "win0": 5, "lose0": 3, "draw": 2, "bot0model": "old weights"}
        second = dict(first, bot0name="alpha_100po", bot1name="beta_200po", bot0model="new weights")
        write_json(self.root / "matchresult/round1.json", first)
        write_json(self.root / "matchresult/round2.json", second)
        shutil.copyfile(self.root / "matchresult/round1.json", self.root / "lib/round1.json")
        result = calculate_all(self.root, ["matchresult", "lib"])
        self.assertEqual(result["total_games"], 20)
        self.assertEqual(len(result["skipped_files"]), 1)
        self.assertEqual({r["participant"] for r in result["ratings"]}, {"alpha_100po", "beta_200po"})
        self.assertEqual(calculate_all(self.root, ["matchresult", "lib"], deduplicate_copies=False)["total_games"], 30)

    def test_independent_identical_scores_are_not_deduplicated(self):
        data = {"bot0name": "alpha", "bot1name": "beta", "total": 4, "win0": 2, "lose0": 1, "draw": 1}
        write_json(self.root / "matchresult/time1.json", data)
        write_json(self.root / "matchresult/time2.json", data)
        self.assertEqual(calculate_all(self.root)["total_games"], 8)

    def test_old_plan_restores_non100_budget_to_old_names(self):
        plan = generate(self.root, self.cfg)
        pair = plan["pairs"][0]
        pair.update(a="A/alpha", b="B/beta", elo_a="alpha", elo_b="beta", playouts_a=150, playouts_b=200)
        write_json(self.root / "plans" / (plan["run_id"] + ".json"), plan)
        write_json(self.root / "match_plan.json", plan)
        self.save_result(pair)
        verify_pair(self.root, pair)
        self.assertEqual({r["participant"] for r in calculate_all(self.root)["ratings"]}, {"alpha_150po", "beta_200po"})

    def test_loose_invalid_incomplete_and_empty_results_rejected(self):
        with self.assertRaisesRegex(ValueError, "No completed"):
            calculate_all(self.root)
        data = {"bot0name": "alpha", "bot1name": "beta", "total": 4, "win0": 2, "lose0": 1, "draw": 1}
        for change in ({"total": 8}, {"numGamesRequested": 10}, {"win0": -1}, {"bot0name": ""}):
            write_json(self.root / "matchresult/bad.json", dict(data, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                calculate_all(self.root)

    def test_name_normalization(self):
        self.assertEqual(normalize_name("A/model", 150), "model_150po")
        self.assertEqual(normalize_name("B/model_400po", 100), "model_400po")
        self.assertEqual(normalize_name("model"), "model_100po")
        with self.assertRaises(ValueError):
            normalize_name("model", 0)

    @unittest.skipUnless(os.name == "posix", "Full Bash execution tested on Linux")
    def test_generated_bash_exec_quoting_environment_resume_and_elo(self):
        self.alias("modelsGroupA", "alpha_200po")
        self.cfg.VISIBLE_GPUS = [2, 5]
        self.cfg.LIBRARY_PATHS = ["lib", "folder with 'quotes' $space"]
        (self.root / self.cfg.LIBRARY_PATHS[1]).mkdir()
        self.cfg.PYTHON_COMMAND = sys.executable
        plan = generate(self.root, self.cfg)
        script = self.root / "run_matches.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        env = dict(os.environ, LD_LIBRARY_PATH="/previous/path")
        first = subprocess.run(["bash", str(script)], cwd="/tmp", env=env, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        for pair in plan["pairs"]:
            self.assertEqual(pair_state(self.root, pair), 0)
            r, _ = result_for_pair(self.root, pair)
            self.assertEqual(r["visible"], "2,5")
            self.assertTrue(r["libs"].endswith(":/previous/path"))
            self.assertEqual(int(r["params"]["maxPlayouts0"]), pair["playouts_a"])
        second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(second.stdout.count("Skip verified"), 2)

    @unittest.skipUnless(os.name == "posix", "Full Bash execution tested on Linux")
    def test_bash_failure_does_not_mark_or_continue(self):
        self.alias("modelsGroupA", "alpha_200po")
        self.cfg.PYTHON_COMMAND = sys.executable
        plan = generate(self.root, self.cfg)
        r = subprocess.run(["bash", str(self.root / "run_matches.sh")], env=dict(os.environ, FAKE_FAIL="1"), capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.root / plan["pairs"][0]["directory"] / "complete.json").exists())
        self.assertFalse((self.root / plan["pairs"][1]["directory"]).exists())


class EloTests(unittest.TestCase):
    def test_known_two_player_likelihood(self):
        result = fit_elo(["a", "b"], [("a", "b", 445, 224, 331)], pseudo_wins=0)
        by_name = {r["participant"]: r["elo"] for r in result["ratings"]}
        expected = 400 * math.log10((445 + 331 / 2) / (224 + 331 / 2))
        self.assertAlmostEqual(by_name["a"] - by_name["b"], expected, places=6)
        self.assertAlmostEqual(sum(by_name.values()), 0, places=6)

    def test_cross_group_global_fit_and_reference(self):
        r = fit_elo(["a", "b", "c", "d"], [("a", "c", 80, 20, 0), ("a", "d", 90, 10, 0),
                                           ("b", "c", 30, 70, 0), ("b", "d", 70, 30, 0)], reference="c", reference_rating=1500)
        ratings = {x["participant"]: x["elo"] for x in r["ratings"]}
        self.assertEqual([x["participant"] for x in r["ratings"]], ["a", "c", "b", "d"])
        self.assertAlmostEqual(ratings["c"], 1500)

    def test_draws_sweeps_and_disconnected(self):
        for match in (("a", "b", 0, 0, 20), ("a", "b", 20, 0, 0)):
            r = fit_elo(["a", "b"], [match])
            self.assertTrue(all(math.isfinite(x["elo"]) for x in r["ratings"]))
        with self.assertRaisesRegex(ValueError, "disconnected"):
            fit_elo(["a", "b", "c"], [("a", "b", 1, 1, 0)])
        with self.assertRaises(ValueError):
            fit_elo(["a", "b"], [("a", "b", 20, 0, 0)], pseudo_wins=0)


if __name__ == "__main__":
    unittest.main()
