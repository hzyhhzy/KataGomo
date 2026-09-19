"""Pool all match-result JSONs across rounds, by name/playouts (not model weights)."""
import argparse
import csv
import math
from pathlib import Path
import re
import runpy
import sys

from match_tools import asset_path, load_plan, pair_state, read_json, result_for_pair, sha256, write_json


def normalize_name(name, playouts=100):
    """Group prefixes are bookkeeping only; every rating name includes its budget."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing participant name")
    name = name.strip()
    if name.startswith(("A/", "B/")):
        name = name[2:]
    suffix = re.fullmatch(r"(.+)_([0-9]+)po", name)
    if suffix:
        name, playouts = suffix[1], int(suffix[2])
    if not name or type(playouts) is not int or playouts <= 0:
        raise ValueError("Playouts must be a positive integer")
    return f"{name}_{playouts}po"


def fit_elo(names, matches, pseudo_wins=0.5, reference=None, reference_rating=0.0):
    """matches = [(A, B, A wins, B wins, draws), ...]. Draw = half a win."""
    if not math.isfinite(pseudo_wins) or pseudo_wins < 0:
        raise ValueError("pseudo_wins must be finite and >= 0")
    if not math.isfinite(reference_rating):
        raise ValueError("reference_rating must be finite")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Need at least two unique participants")
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    edges = {}
    stats = [{"participant": name, "games": 0} for name in names]
    for a, b, wa, wb, draws in matches:
        if a not in index or b not in index or a == b:
            raise ValueError("Unknown participant or self match")
        if any(type(v) is not int or v < 0 for v in (wa, wb, draws)) or wa + wb + draws == 0:
            raise ValueError("Invalid outcome counts")
        i, j = index[a], index[b]
        for k in (i, j):
            stats[k]["games"] += wa + wb + draws
        if i > j:
            i, j, wa, wb = j, i, wb, wa
        s = edges.setdefault((i, j), [0.0, 0.0])
        s[0] += wa + draws * 0.5
        s[1] += wb + draws * 0.5
    adj = [[] for _ in names]
    scores = [0.0] * n
    for (i, j), (si, sj) in edges.items():
        si += pseudo_wins
        sj += pseudo_wins
        scores[i] += si
        scores[j] += sj
        adj[i].append((j, si + sj))
        adj[j].append((i, si + sj))
    connected = {0}
    pending = [0]
    while pending:
        for j, _ in adj[pending.pop()]:
            if j not in connected:
                connected.add(j)
                pending.append(j)
    if len(connected) != n:
        raise ValueError("Match graph is disconnected; global Elo is not identifiable for all participants")
    if min(scores) <= 0:
        raise ValueError("No finite unregularized Elo for an all-loss player; use pseudo_wins > 0")
    # Minorization-maximization for the Bradley-Terry likelihood, fixed mean log strength.
    strengths = [1.0] * n
    for iteration in range(100000):
        new = [scores[i] / sum(games / (strengths[i] + strengths[j]) for j, games in adj[i]) for i in range(n)]
        if min(new) <= 0 or not all(math.isfinite(x) for x in new):
            raise ValueError("Elo fit diverged; use positive pseudo_wins")
        center = sum(math.log(x) for x in new) / n
        new = [math.exp(math.log(x) - center) for x in new]
        change = max(abs(math.log(new[i] / strengths[i])) for i in range(n))
        strengths = new
        if change < 1e-10:
            break
    else:
        raise ValueError("Elo fit did not converge; check extreme outcomes or use larger pseudo_wins")
    ratings = [400.0 * math.log10(x) for x in strengths]
    shift = 0.0
    if reference is not None:
        if reference not in index:
            raise ValueError(f"Unknown Elo reference: {reference}")
        shift = reference_rating - ratings[index[reference]]
    for i, row in enumerate(stats):
        row["elo"] = ratings[i] + shift
    stats.sort(key=lambda r: (-r["elo"], r["participant"]))
    for rank, row in enumerate(stats, 1):
        row["rank"] = rank
    return {"method": "Bradley-Terry; draw=0.5; 400-point logistic scale",
            "pseudo_wins_per_side_per_played_pair": pseudo_wins,
            "reference": reference, "reference_rating": reference_rating if reference else None,
            "gauge": "reference" if reference else "mean Elo = 0", "iterations": iteration + 1,
            "total_games": sum(r["games"] for r in stats) // 2, "ratings": stats,
            "note": "Relative ratings, not calibrated external Elo. Smoothing is not a confidence interval. Outcomes are listed per match, not as opponent-unadjusted bot win rates."}


def calculate(root, plan):
    matches = []
    records = []
    self_play_games = 0
    for pair in plan["pairs"]:
        if pair_state(root, pair) != 0:
            raise ValueError(f"{pair['id']}: not complete; finish all planned matches before calculating global Elo")
        r, path = result_for_pair(root, pair)
        a = normalize_name(pair["elo_a"], pair["playouts_a"])
        b = normalize_name(pair["elo_b"], pair["playouts_b"])
        records.append({"path": str(path), "event": (plan["run_id"], pair["id"]),
                        "a": a, "b": b, "wins_a": r["win0"], "wins_b": r["lose0"],
                        "draws": r["draw"], "games": r["total"]})
        if a == b:
            self_play_games += r["total"]
        else:
            matches.append((a, b, r["win0"], r["lose0"], r["draw"]))
    names = sorted({name for match in matches for name in match[:2]})
    options = dict(plan["elo"])
    if options["reference"] is not None:
        options["reference"] = normalize_name(options["reference"])
    result = fit_elo(names, matches, **options)
    result["run_id"] = plan["run_id"]
    result["self_play_games_excluded_from_ratings"] = self_play_games
    result["total_scheduled_games"] = result["total_games"] + self_play_games
    result["matches"] = match_rows(records)
    output = root / "results" / plan["run_id"]
    save_ratings(output, result)
    return result


def match_rows(records):
    """Keep each accepted result JSON as one inspectable matchup, including self-play."""
    rows = []
    for number, record in enumerate(records, 1):
        event = record.get("event") or ("", "")
        rows.append({"match_no": number, "bot_a": record["a"], "bot_b": record["b"],
                     "games": record["games"], "wins_a": record["wins_a"], "draws": record["draws"],
                     "wins_b": record["wins_b"], "included_in_elo": record["a"] != record["b"],
                     "run_id": event[0], "pair_id": event[1], "result_file": record["path"]})
    return rows


def save_ratings(output, result):
    write_json(output / "elo.json", result)
    with (output / "elo.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "participant", "elo", "games"])
        writer.writeheader()
        writer.writerows(result["ratings"])
    with (output / "matches.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_no", "bot_a", "bot_b", "games", "wins_a", "draws", "wins_b",
                                              "included_in_elo", "run_id", "pair_id", "result_file"])
        writer.writeheader()
        writer.writerows(result["matches"])
    print(f"Match details ({len(result['matches'])} result JSONs):")
    print(f"{'#':>4}  {'Bot A':40} {'A wins':>7} {'Draws':>7} {'B wins':>7}  {'Bot B':40} {'Games':>7}")
    for row in result["matches"]:
        note = " (same-name self-play; excluded from Elo)" if not row["included_in_elo"] else ""
        print(f"{row['match_no']:4}  {row['bot_a']:40} {row['wins_a']:7} {row['draws']:7} {row['wins_b']:7}  {row['bot_b']:40} {row['games']:7}{note}")
    print()
    print(f"Elo: {result['total_games']} games; pseudo-wins/side/pair={result['pseudo_wins_per_side_per_played_pair']}; {result['gauge']}")
    if result["self_play_games_excluded_from_ratings"]:
        print(f"Same-name self-play excluded from ratings: {result['self_play_games_excluded_from_ratings']} games")
    print(f"{'Rank':>4} {'Bot':40} {'Elo':>9} {'Games':>7}")
    for row in result["ratings"]:
        print(f"{row['rank']:4} {row['participant']:40} {row['elo']:9.2f} {row['games']:7}")
    print(f"Saved: {output / 'matches.csv'} ; {output / 'elo.csv'}")


def validate_result(data, path):
    for key in ("bot0name", "bot1name"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{path}: missing {key}")
    for key in ("total", "win0", "lose0", "draw"):
        if type(data.get(key)) is not int or data[key] < 0:
            raise ValueError(f"{path}: invalid {key}")
    if data["total"] <= 0 or sum(data[k] for k in ("win0", "lose0", "draw")) != data["total"]:
        raise ValueError(f"{path}: inconsistent/empty game counts")
    if "numGamesRequested" in data and data["total"] != data["numGamesRequested"]:
        raise ValueError(f"{path}: incomplete result ({data['total']}/{data['numGamesRequested']})")


def collect_results(root, directories=None, default_playouts=100, deduplicate_copies=True):
    """Read saved JSON only. Old models and current model files are never needed."""
    root = Path(root).resolve()
    normalize_name("validate", default_playouts)
    folders = [root / "results", root / "matchresult"] if directories is None else [
        Path(d) if Path(d).is_absolute() else root / d for d in directories]
    paths = sorted({p.resolve() for folder in folders if folder.is_dir() for p in folder.rglob("*.json")})
    if directories is not None:
        for folder in folders:
            if not folder.is_dir():
                raise ValueError(f"Result directory does not exist: {folder}")
    # Plans are optional. When retained, they supply old budgets and enforce completion markers.
    planned = {}
    for path in sorted(set((root / "plans").glob("*.json")) | ({root / "match_plan.json"} if (root / "match_plan.json").is_file() else set())):
        plan = read_json(path)
        if not isinstance(plan, dict) or plan.get("schema") != 1:
            continue
        for pair in plan["pairs"]:
            directory = asset_path(root, pair["directory"]).resolve()
            if directory in planned and planned[directory][1] != pair:
                raise ValueError(f"Conflicting plans for {directory}")
            planned[directory] = (plan["run_id"], pair)
    records, skipped, ignored = [], [], 0
    for path in paths:
        known = planned.get(path.parent.parent) if path.parent.name == "matchresult" else None
        if known:
            run_id, pair = known
            if not (path.parent.parent / "complete.json").is_file():
                skipped.append({"path": str(path), "reason": "planned pair not yet verified complete"})
                continue
            pair_state(root, pair)  # Raises on changed/corrupt results, never silently trusts them.
            data, _ = result_for_pair(root, pair)
            budgets = [pair["playouts_a"], pair["playouts_b"]]
            event = (run_id, pair["id"])
        else:
            data = read_json(path)
            budgets = [default_playouts, default_playouts]
            event = None
        if not isinstance(data, dict) or not any(k in data for k in ("bot0name", "bot1name", "win0", "lose0")):
            ignored += 1  # elo.json, complete.json, manifests, etc. are not match results.
            continue
        validate_result(data, path)
        names = [normalize_name(data[f"bot{i}name"], budgets[i]) for i in range(2)]
        records.append({"path": str(path), "file_name": path.name, "sha256": sha256(path),
                        "event": event, "a": names[0], "b": names[1], "wins_a": data["win0"],
                        "wins_b": data["lose0"], "draws": data["draw"], "games": data["total"]})
    # An independent round with the same score is NOT a copy. Managed run/pair IDs
    # take precedence; for loose JSONs only identical filename+bytes count as copies.
    records.sort(key=lambda r: (r["event"] is None, r["path"]))
    unique, seen_events, seen_files = [], set(), set()
    for record in records:
        fingerprint = (record["file_name"], record["sha256"])
        event = record["event"]
        duplicate = event in seen_events if event is not None else deduplicate_copies and fingerprint in seen_files
        if duplicate:
            skipped.append({"path": record["path"], "reason": "duplicate result copy"})
            continue
        if event is not None:
            seen_events.add(event)
        seen_files.add(fingerprint)
        unique.append(record)
    return unique, skipped, ignored


def calculate_all(root, directories=None, default_playouts=100, deduplicate_copies=True, options=None):
    root = Path(root).resolve()
    records, skipped, ignored = collect_results(root, directories, default_playouts, deduplicate_copies)
    if not records:
        raise ValueError("No completed match-result JSONs found yet; no engine was started")
    matches, self_play_games = [], 0
    for record in records:
        if record["a"] == record["b"]:
            self_play_games += record["games"]
        else:
            matches.append((record["a"], record["b"], record["wins_a"], record["wins_b"], record["draws"]))
    names = sorted({name for match in matches for name in match[:2]})
    if not names:
        raise ValueError("Only same-name self-play results found; no relative Elo information")
    reference_playouts = default_playouts
    if options is None:
        cfg = runpy.run_path(str(root / "auto_match_config.py")) if (root / "auto_match_config.py").is_file() else {}
        options = {"pseudo_wins": cfg.get("ELO_PSEUDO_WINS", 0.5), "reference": cfg.get("ELO_REFERENCE"),
                   "reference_rating": cfg.get("ELO_REFERENCE_RATING", 0.0)}
        reference_playouts = cfg.get("DEFAULT_PLAYOUTS", default_playouts)
    options = dict(options)
    if options.get("reference") is not None:
        options["reference"] = normalize_name(options["reference"], reference_playouts)
    result = fit_elo(names, matches, **options)
    result.update({"scope": "all collected rounds", "identity": "name + playout suffix; weights ignored",
                   "result_files": len(records), "self_play_games_excluded_from_ratings": self_play_games,
                   "total_scheduled_games": result["total_games"] + self_play_games,
                   "ignored_non_result_jsons": ignored, "skipped_files": skipped,
                   "input_files": records, "matches": match_rows(records), "legacy_default_playouts": default_playouts})
    output = root / "elo_all"
    save_ratings(output, result)
    print(f"Pooled {len(records)} result JSONs, skipped {len(skipped)} incomplete/copy files; model weights are not compared.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--plan", help="Optional: calculate just one plan instead of all results")
    selection.add_argument("--results-dir", nargs="+", help="Optional folders of mixed result JSONs, absolute or relative to this environment")
    parser.add_argument("--default-playouts", type=int, default=100, help="Budget for old bare names when no plan is available (default 100)")
    parser.add_argument("--keep-copies", action="store_true", help="Count identical loose files separately instead of deduplicating filename+bytes")
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parent
        if args.plan:
            calculate(root, load_plan(root, args.plan))
        else:
            calculate_all(root, args.results_dir, args.default_playouts, not args.keep_copies)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
