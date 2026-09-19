"""Explicit real-engine smoke test; 2x2 pairs, 8 games/pair, in validation/cuda_smoke.

Requires the default two native models and engine installed in the environment.
Does not change the main configuration or main results. Refuses an existing test folder.
"""
import copy
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import auto_match_config as config
from match_tools import read_json, result_for_pair, write_json
from script_generator import generate


def main():
    if os.name != "posix":
        raise ValueError("Real CUDA smoke test must run on Linux")
    test = ROOT / "validation" / "cuda_smoke"
    test.mkdir(parents=True, exist_ok=False)
    for name in ("match_tools.py", "calculate_elo.py", "script_generator.py", "auto_match_config.py", "match.cfg"):
        shutil.copyfile(ROOT / name, test / name)
    cfg = SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(config).items() if k.isupper()})
    a_name = "b14c192h6tflrs"
    b_name = "b16c128h4tflrs"
    for group, name in ((cfg.GROUP_A, a_name), (cfg.GROUP_B, b_name)):
        shutil.copytree(ROOT / group / name, test / group / name)
    (test / "engine").mkdir()
    shutil.copyfile(ROOT / cfg.ENGINE_NATIVE, test / cfg.ENGINE_NATIVE)
    (test / "openings").mkdir()
    shutil.copyfile(ROOT / cfg.OPENING_FILE, test / cfg.OPENING_FILE)
    (test / "lib").mkdir()
    (test / cfg.GROUP_A / (a_name + "_8po")).mkdir()
    (test / cfg.GROUP_B / (b_name + "_12po")).mkdir()
    cfg.GAMES_PER_PAIR = 8
    cfg.GAME_THREADS = 4
    cfg.DEFAULT_PLAYOUTS = 4
    cfg.NN_CACHE_POWER = 20
    cfg.PYTHON_COMMAND = sys.executable
    cfg.EXPERIMENT_TAG = "cuda-smoke-32-games-not-a-strength-benchmark"
    cfg.LIBRARY_PATHS = [str(ROOT / p) if not p.startswith(("/", "~")) else p for p in cfg.LIBRARY_PATHS]
    plan = generate(test, cfg)
    with (test / "sweep.log").open("w") as log:
        subprocess.run(["bash", str(test / "run_matches.sh")], cwd="/tmp", stdout=log, stderr=subprocess.STDOUT, check=True)
    openings = []
    for line in (test / cfg.OPENING_FILE).read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("#", 1)[0].split()
        if parts:
            openings.append([( "B" if i % 2 == 0 else "W", chr(97 + "ABCDEFGHJKLMNOP".index(move[0])) + chr(97 + 15 - int(move[1:]))) for i, move in enumerate(parts[1:])])
    audited = []
    for pair in plan["pairs"]:
        result, _ = result_for_pair(test, pair)
        folder = test / pair["directory"]
        log = (folder / "match.log").read_text()
        assert "C384 INT8 ACTIVE" not in log
        assert "useINT8 false" in log
        assert f"maxPlayouts0 = {pair['playouts_a']}" in log
        assert f"maxPlayouts1 = {pair['playouts_b']}" in log
        assert "Match opening finished" not in log
        # Audit persisted SGFs instead of relying on noisy per-game log lines.
        expected = Counter()
        for game in range(pair["games"]):
            black, white = (pair["a"], pair["b"]) if game % 2 == 0 else (pair["b"], pair["a"])
            expected[(tuple(openings[(game // 2) % len(openings)]), black, white)] += 1
        sgfs = [line for path in (folder / "sgfs").glob("*.sgfs") for line in path.read_text().splitlines() if line.strip()]
        assert len(sgfs) == 8
        sgf_hashes = set()
        actual = Counter()
        for sgf in sgfs:
            game_hash = re.search(r"gameHash=([0-9A-F]+)", sgf)[1]
            sgf_hashes.add(game_hash)
            moves = re.findall(r";([BW])\[([a-o]{2})\]", sgf)
            black = re.search(r"PB\[([^\]]+)\]", sgf)[1]
            white = re.search(r"PW\[([^\]]+)\]", sgf)[1]
            actual[(tuple(moves[:5]), black, white)] += 1
            assert "startTurnIdx=5" in sgf
        assert len(sgf_hashes) == 8
        assert actual == expected
        audited.append({"pair": pair["id"], "a": pair["a"], "b": pair["b"],
                        "playouts": [pair["playouts_a"], pair["playouts_b"]], "games": result["total"],
                        "ordered_openings_and_swapped_colors": "PASS", "sgf_prefixes": "PASS", "int8": False})
    retry = subprocess.run(["bash", str(test / "run_matches.sh")], capture_output=True, text=True, check=True)
    assert retry.stdout.count("Skip verified") == 4
    (test / "retry.log").write_text(retry.stdout)
    elo = read_json(test / "elo_all" / "elo.json")
    assert len(elo["ratings"]) == 4 and elo["total_games"] == 32
    summary = {"status": "PASS", "pairs": audited, "total_games": 32, "elo_participants": 4,
               "safe_resume": "4 verified pairs skipped", "note": "Small playout smoke test only; not a strength benchmark"}
    write_json(test / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
