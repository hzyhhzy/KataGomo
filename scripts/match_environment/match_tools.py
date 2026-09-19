"""Shared plan/result checks. Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    # Atomic for the small control files; never overwrites engine result files.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def asset_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Asset paths must be portable relative paths: {relative!r}")
    if not relative or not (root / p).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset escapes environment folder: {relative!r}")
    return root / p


def load_plan(root, name):
    path = asset_path(root, name)
    plan = read_json(path)
    if plan.get("schema") != 1:
        raise ValueError("Unknown match plan schema")
    return plan


def check_assets(root, plan):
    for name, digest in plan["assets"].items():
        path = asset_path(root, name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing/changed asset: {name}; regenerate the match script")


def result_for_pair(root, pair):
    folder = asset_path(root, pair["directory"])
    files = sorted((folder / "matchresult").glob("*.json"))
    if len(files) != 1:
        raise ValueError(f"{pair['id']}: expected exactly one result JSON, found {len(files)}")
    result = read_json(files[0])
    for key, expected in (("bot0name", pair["a"]), ("bot1name", pair["b"])):
        if result.get(key) != expected:
            raise ValueError(f"{pair['id']}: {key} mismatch")
    for key in ("total", "win0", "lose0", "draw"):
        if type(result.get(key)) is not int or result[key] < 0:
            raise ValueError(f"{pair['id']}: invalid {key}")
    if result["total"] != pair["games"] or sum(result[k] for k in ("win0", "lose0", "draw")) != pair["games"]:
        raise ValueError(f"{pair['id']}: incomplete or inconsistent game counts")
    return result, files[0]


def verify_pair(root, pair):
    result, path = result_for_pair(root, pair)
    write_json(asset_path(root, pair["directory"]) / "complete.json", {
        "pair": pair["id"], "total": result["total"], "result": path.name,
        "sha256": sha256(path),
    })


def pair_state(root, pair):
    """0=verified complete, 1=not started. Reject partial/unmarked folders."""
    folder = asset_path(root, pair["directory"])
    if not folder.exists():
        return 1
    marker = folder / "complete.json"
    if not marker.is_file():
        raise ValueError(f"{pair['id']}: unfinished folder {folder}. Inspect it and move it aside before retrying; no partial results are reused.")
    result, path = result_for_pair(root, pair)
    check = read_json(marker)
    if check != {"pair": pair["id"], "total": result["total"], "result": path.name, "sha256": sha256(path)}:
        raise ValueError(f"{pair['id']}: result differs from completion marker")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check-plan", "pair-state", "verify-pair"])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pair")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = load_plan(root, args.plan)
    if args.action == "check-plan":
        check_assets(root, plan)
        print(f"Assets verified. {len(plan['pairs'])} cross-group pairs.")
        return 0
    pair = next((p for p in plan["pairs"] if p["id"] == args.pair), None)
    if pair is None:
        raise ValueError("Unknown pair ID")
    if args.action == "pair-state":
        return pair_state(root, pair)
    verify_pair(root, pair)
    print(f"Verified {pair['id']}: {pair['games']} games")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
