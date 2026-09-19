"""Generate sequential A x B Bash matches with explicit, safely quoted overrides."""
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys

from match_tools import asset_path, sha256, write_json
from calculate_elo import normalize_name


def scalar(value):
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value)
    # The engine splits override values on commas and interprets # as a comment.
    if any(c in text for c in ",\r\n\x00#") or text != text.strip():
        raise ValueError(f"Unsupported engine override value: {text!r}")
    return text


def positive(name, value, limit=2147483647):
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError(f"{name} must be an integer in [1, {limit}]")
    return value


def scan_group(root, directory, label, cfg):
    base = asset_path(root, directory)
    if not base.is_dir():
        raise ValueError(f"Missing group directory: {directory}")
    entries = []
    for folder in sorted(base.iterdir(), key=lambda p: p.name):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        alias = re.fullmatch(r"(.+)_([0-9]+)po", folder.name)
        playouts = cfg.DEFAULT_PLAYOUTS
        model_folder = folder
        if alias:
            playouts = positive("alias playouts", int(alias[2]))
            model_folder = base / alias[1]
            if not model_folder.is_dir():
                raise ValueError(f"{directory}/{folder.name}: missing base directory {alias[1]}")
        files = [model_folder / f for f in cfg.MODEL_FILE_PREFERENCE if (model_folder / f).is_file()]
        if not files:
            raise ValueError(f"No supported model in {model_folder}")
        model = files[0].relative_to(root).as_posix()
        scalar(model)
        asset_path(root, model)
        display_name = normalize_name(folder.name, playouts)
        entries.append({"id": scalar(f"{label}/{display_name}"), "name": folder.name, "rating_id": display_name, "model": model,
                        "format": "onnx" if files[0].suffix == ".onnx" else "native",
                        "playouts": playouts})
    if not entries:
        raise ValueError(f"No participants in {directory}")
    return entries


def rooted_shell(path):
    return '"$ROOT"/' + shlex.quote(path)


def library_shell(path):
    if not isinstance(path, str) or not path or any(c in path for c in ":\r\n\x00"):
        raise ValueError("Library paths must be nonempty Linux paths without ':'")
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    if path.startswith("/"):
        return shlex.quote(path)
    return rooted_shell(path)


def generate(root, cfg):
    root = Path(root).resolve()
    for key in ("GAMES_PER_PAIR", "DEFAULT_PLAYOUTS", "GAME_THREADS", "SEARCH_THREADS", "BATCH_SIZE", "NN_SERVER_THREADS_PER_MODEL"):
        positive(key, getattr(cfg, key))
    if type(cfg.NN_CACHE_POWER) is not int or not -1 <= cfg.NN_CACHE_POWER <= 48:
        raise ValueError("NN_CACHE_POWER must be -1..48")
    if type(cfg.GRAPH_SEARCH) is not bool:
        raise ValueError("GRAPH_SEARCH must be bool")
    if type(cfg.DEDUPLICATE_MATCHES) is not bool:
        raise ValueError("DEDUPLICATE_MATCHES must be bool")
    if not isinstance(cfg.ELO_PSEUDO_WINS, (int, float)) or not math.isfinite(cfg.ELO_PSEUDO_WINS) or cfg.ELO_PSEUDO_WINS < 0:
        raise ValueError("ELO_PSEUDO_WINS must be finite and nonnegative")
    if not isinstance(cfg.ELO_REFERENCE_RATING, (int, float)) or not math.isfinite(cfg.ELO_REFERENCE_RATING):
        raise ValueError("ELO_REFERENCE_RATING must be finite")
    if not cfg.VISIBLE_GPUS or any(type(i) is not int or i < 0 for i in cfg.VISIBLE_GPUS) or len(set(cfg.VISIBLE_GPUS)) != len(cfg.VISIBLE_GPUS):
        raise ValueError("VISIBLE_GPUS must be distinct nonnegative integer GPU IDs")
    if cfg.NN_SERVER_THREADS_PER_MODEL < len(cfg.VISIBLE_GPUS):
        raise ValueError("NN_SERVER_THREADS_PER_MODEL must cover all visible GPUs")
    supported = {"model.bin.gz", "model.bin", "model.onnx"}
    if not cfg.MODEL_FILE_PREFERENCE or not set(cfg.MODEL_FILE_PREFERENCE) <= supported or len(set(cfg.MODEL_FILE_PREFERENCE)) != len(cfg.MODEL_FILE_PREFERENCE):
        raise ValueError("MODEL_FILE_PREFERENCE must be an ordered subset of supported filenames")
    if asset_path(root, cfg.GROUP_A).resolve() == asset_path(root, cfg.GROUP_B).resolve():
        raise ValueError("Group directories must be distinct")
    a = scan_group(root, cfg.GROUP_A, "A", cfg)
    b = scan_group(root, cfg.GROUP_B, "B", cfg)
    opening = (cfg.OPENING_FILE or "").strip()
    if opening == '""':
        opening = ""
    common = {
        "numBots": 2, "numGamesTotal": cfg.GAMES_PER_PAIR, "numGameThreads": cfg.GAME_THREADS,
        "numSearchThreads": cfg.SEARCH_THREADS, "nnMaxBatchSize": cfg.BATCH_SIZE,
        "numNNServerThreadsPerModel": cfg.NN_SERVER_THREADS_PER_MODEL,
        "nnCacheSizePowerOfTwo": cfg.NN_CACHE_POWER, "useGraphSearch": cfg.GRAPH_SEARCH,
        "cudaUseINT8": False,
    }
    for i in range(cfg.NN_SERVER_THREADS_PER_MODEL):
        common[f"gpuToUseThread{i}"] = i % len(cfg.VISIBLE_GPUS)
    for key, value in cfg.EXTRA_OVERRIDES.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid override key: {key!r}")
        if key in common or key.startswith(("botName", "nnModelFile", "maxPlayouts", "maxVisits", "maxTime", "gpu", "device", "cudaGpu", "cudaDevice", "onnxGpu", "onnxDevice", "numBots", "numNNServer", "includeBot")) or key in {"matchOpeningFile", "matchRepFactor", "extraPairs", "secondaryBots", "blackPriority0", "blackPriority1"}:
            raise ValueError(f"Reserved override: {key}")
        common[key] = scalar(value)
    assets = {}

    def add_asset(name):
        scalar(name)
        path = asset_path(root, name)
        if not path.is_file():
            raise ValueError(f"Missing asset: {name}")
        if name not in assets:
            assets[name] = sha256(path)

    add_asset(cfg.BASE_CONFIG)
    if opening:
        add_asset(opening)
        lines = [l.split("#", 1)[0].strip() for l in (root / opening).read_text(encoding="utf-8-sig").splitlines()]
        if not any(lines):
            raise ValueError("Opening library is empty")
    # Pin the runtime helpers too, so an old plan cannot silently change validation.
    for helper in ("match_tools.py", "calculate_elo.py"):
        add_asset(helper)
    # Rating identity is explicitly name + playouts, NOT model weights.
    # Still retain independent physical A/B paths and per-run asset checksums.
    participants = []
    for bot in a + b:
        add_asset(bot["model"])
        if not any(p["id"] == bot["rating_id"] for p in participants):
            participants.append({"id": bot["rating_id"], "name": bot["name"]})
    reference = normalize_name(cfg.ELO_REFERENCE, cfg.DEFAULT_PLAYOUTS) if cfg.ELO_REFERENCE is not None else None
    if reference is not None and reference not in {p["id"] for p in participants}:
        raise ValueError(f"Unknown ELO_REFERENCE: {cfg.ELO_REFERENCE}")
    pairs = []
    seen_pairs = set()
    for bot_a in a:
        for bot_b in b:
            key = tuple(sorted((bot_a["rating_id"], bot_b["rating_id"])))
            if cfg.DEDUPLICATE_MATCHES and (bot_a["rating_id"] == bot_b["rating_id"] or key in seen_pairs):
                continue
            seen_pairs.add(key)
            if bot_a["format"] != bot_b["format"]:
                raise ValueError(f"Cannot mix native/ONNX in one match: {bot_a['id']} vs {bot_b['id']}. Export both to the same format.")
            engine = cfg.ENGINE_ONNX if bot_a["format"] == "onnx" else cfg.ENGINE_NATIVE
            for path in (engine, bot_a["model"], bot_b["model"]):
                add_asset(path)
            pairs.append({"id": f"pair_{len(pairs) + 1:04d}", "a": bot_a["id"], "b": bot_b["id"],
                          "elo_a": bot_a["rating_id"], "elo_b": bot_b["rating_id"],
                          "model_a": bot_a["model"], "model_b": bot_b["model"], "engine": engine,
                          "playouts_a": bot_a["playouts"], "playouts_b": bot_b["playouts"], "games": cfg.GAMES_PER_PAIR})
    if not pairs:
        raise ValueError("No matches remain after deduplication")
    # Validate library paths now, but existence is checked on the Linux host, not Windows.
    lib_exprs = [library_shell(p) for p in cfg.LIBRARY_PATHS]
    if not isinstance(cfg.PYTHON_COMMAND, str) or not cfg.PYTHON_COMMAND or any(c in cfg.PYTHON_COMMAND for c in "\r\n\x00"):
        raise ValueError("PYTHON_COMMAND must be a single executable name/path")
    plan = {"schema": 1, "tag": str(cfg.EXPERIMENT_TAG), "participants": participants, "entries": a + b, "pairs": pairs,
            "deduplicate_matches": cfg.DEDUPLICATE_MATCHES,
            "assets": assets, "base_config": cfg.BASE_CONFIG, "opening_file": opening,
            "overrides": common, "visible_gpus": cfg.VISIBLE_GPUS, "library_paths": cfg.LIBRARY_PATHS,
            "elo": {"reference": reference, "reference_rating": cfg.ELO_REFERENCE_RATING,
                    "pseudo_wins": cfg.ELO_PSEUDO_WINS}}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    plan["run_id"] = digest
    for pair in pairs:
        pair["directory"] = f"results/{digest}/{pair['id']}"
    plan_name = f"plans/{digest}.json"
    write_json(root / plan_name, plan)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"', 'cd "$ROOT"',
             f"PYTHON={shlex.quote(cfg.PYTHON_COMMAND)}", "export PYTHONIOENCODING=utf-8",
             f"export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str, cfg.VISIBLE_GPUS)))}",
             "export KATAGO_C384_INT8_MODE=off"]
    if lib_exprs:
        lines += ["LIB_DIRS=(" + " ".join(lib_exprs) + ")", 'for lib in "${LIB_DIRS[@]}"; do',
                  '  [[ -d "$lib" ]] || { echo "Missing library directory: $lib" >&2; exit 2; }', "done",
                  'LIB_PREFIX="$(IFS=:; echo "${LIB_DIRS[*]}")"',
                  'export LD_LIBRARY_PATH="$LIB_PREFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"']
    lines += ['mkdir -p "$ROOT/results"', 'exec 9>"$ROOT/results/.match.lock"',
              'flock -n 9 || { echo "Another match sweep is already running" >&2; exit 2; }',
              f'PLAN={shlex.quote(plan_name)}',
              '"$PYTHON" "$ROOT/match_tools.py" check-plan --plan "$PLAN"']
    for engine in sorted(set(p["engine"] for p in pairs)):
        lines += ["chmod +x " + rooted_shell(engine)]
    for pair in pairs:
        lines += ["", "# " + pair["id"], f'if "$PYTHON" "$ROOT/match_tools.py" pair-state --plan "$PLAN" --pair {pair["id"]}; then',
                  f'  echo "Skip verified {pair["id"]}"', "else", "  status=$?",
                  '  [[ "$status" -eq 1 ]] || exit "$status"', "  (",
                  "    mkdir -p " + rooted_shell(pair["directory"]),
                  "    cd " + rooted_shell(pair["directory"])]
        args = [rooted_shell(pair["engine"]), "match", "-config", rooted_shell(cfg.BASE_CONFIG),
                "-sgf-output-dir", "sgfs", "-log-file", "match.log"]
        overrides = dict(common, botName0=pair["a"], botName1=pair["b"],
                         maxPlayouts0=pair["playouts_a"], maxPlayouts1=pair["playouts_b"])
        for key, value in overrides.items():
            args += ["-override-config", shlex.quote(f"{key}={scalar(value)}")]
        for key, path in (("nnModelFile0", pair["model_a"]), ("nnModelFile1", pair["model_b"]), ("matchOpeningFile", opening)):
            args += ["-override-config", shlex.quote(key + "=") + (rooted_shell(path) if path else "")]
        separator = " " + chr(92) + "\n      "
        lines += ["    " + separator.join(args) + " 2>&1 | tee stdout.log", "  )",
                  f'  "$PYTHON" "$ROOT/match_tools.py" verify-pair --plan "$PLAN" --pair {pair["id"]}', "fi"]
    if cfg.CALCULATE_ELO_AT_END:
        lines += ['', '"$PYTHON" "$ROOT/calculate_elo.py"']
    lines += ["", 'echo "All cross-group matches verified."']
    (root / "run_matches.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    write_json(root / "match_plan.json", plan)
    print(f"Generated A={len(a)}, B={len(b)}, deduplicate={cfg.DEDUPLICATE_MATCHES}: {len(pairs)} matches, {cfg.GAMES_PER_PAIR} games/pair, run {digest}")
    print("Linux: bash run_matches.sh")
    return plan


if __name__ == "__main__":
    try:
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("auto_match_config", root / "auto_match_config.py")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        generate(root, cfg)
    except (ValueError, OSError, AttributeError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
