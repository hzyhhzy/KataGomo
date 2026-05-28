#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path


LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sgf_escape(s):
    return s.replace("\\", "\\\\").replace("]", "\\]")


def sgf_coord(x, y):
    return LETTERS[x] + LETTERS[y]


def sgf_size(x, y):
    return str(x) if x == y else f"{x}:{y}"


def parse_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def exp_mean(rng, mean):
    return rng.expovariate(1.0 / mean)


def neighbors(x, y):
    return [
        (x, y - 1),
        (x - 1, y),
        (x + 1, y),
        (x, y + 1),
        (x - 1, y + 1),
        (x + 1, y - 1),
    ]


def hex_legal_cells(xsize, ysize):
    if xsize != ysize:
        return None, "shape 6 is not allowed unless xsize == ysize"
    if xsize % 2 == 0:
        return None, "shape 6 is not allowed unless xsize is odd"

    t = (xsize - 1) // 2
    legal = set()
    for y in range(ysize):
        for x in range(xsize):
            upper_left_cut = x + y < t
            lower_right_cut = (xsize - x - 1) + (ysize - y - 1) < t
            if not (upper_left_cut or lower_right_cut):
                legal.add((x, y))
    return legal, None


def all_groups_have_liberties(stones, legal):
    seen = set()
    for start, color in list(stones.items()):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        has_liberty = False
        while stack:
            point = stack.pop()
            for adj in neighbors(*point):
                if adj not in legal:
                    continue
                adj_color = stones.get(adj)
                if adj_color is None:
                    has_liberty = True
                elif adj_color == color and adj not in seen:
                    seen.add(adj)
                    stack.append(adj)
        if not has_liberty:
            return False
    return True


def try_place(stones, legal, x, y, color):
    point = (x, y)
    if point not in legal or point in stones:
        return False
    stones[point] = color
    if not all_groups_have_liberties(stones, legal):
        del stones[point]
        return False
    return True


def make_para_opening(rng, xsize, ysize, variant):
    legal = {(x, y) for y in range(ysize) for x in range(xsize)}
    stones = {}
    black_y = 0 if variant == 0 else 1
    black_x = xsize - 1 if variant == 0 else xsize - 2
    if black_y < 0 or black_y >= ysize or black_x < 0 or black_x >= xsize:
        return False, stones, legal, "para opening line out of range", "W"

    for x in range(xsize):
        if not try_place(stones, legal, x, black_y, "B"):
            return False, stones, legal, f"failed placing black at {sgf_coord(x, black_y)}", "W"
    for y in range(ysize):
        if (black_x, y) not in stones and not try_place(stones, legal, black_x, y, "B"):
            return False, stones, legal, f"failed placing black at {sgf_coord(black_x, y)}", "W"

    p = 0.05 + exp_mean(rng, 0.1)
    for y in range(ysize):
        for x in range(xsize):
            if (x, y) in stones:
                continue
            d = math.hypot(x, y - (ysize - 1))
            prob = min(0.15, p * math.exp(-d / 3.0))
            if rng.random() < prob:
                try_place(stones, legal, x, y, "W")

    return True, stones, legal, f"success p={p:.6g}", "W"


def make_hex_opening(rng, xsize, ysize):
    legal, err = hex_legal_cells(xsize, ysize)
    stones = {}
    if legal is None:
        return False, stones, set(), err, "B"

    for x, y in list(legal):
        if any(adj not in legal for adj in neighbors(x, y)):
            if not try_place(stones, legal, x, y, "B"):
                return False, stones, legal, f"failed placing outer black at {sgf_coord(x, y)}", "B"

    empty_area = len(legal) - len(stones)
    if empty_area <= 0:
        return False, stones, legal, "empty area <= 0 after outer black stones", "B"

    white_prob = min(0.25, (10.0 + exp_mean(rng, 10.0)) / empty_area)
    for x, y in list(legal):
        if (x, y) not in stones and rng.random() < white_prob:
            try_place(stones, legal, x, y, "W")

    next_pla = "B" if rng.random() < 0.5 else "W"
    return True, stones, legal, f"success whiteProb={white_prob:.6g} emptyArea={empty_area}", next_pla


def sgf_record(sample_idx, seed, xsize, ysize, opening, ok, stones, legal, msg, next_pla):
    opening_name = ["para-y0-xmax", "para-y1-xmaxminus1", "hex-outer"][opening]
    ordered = sorted(stones.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ab = "".join(f"[{sgf_coord(x, y)}]" for (x, y), color in ordered if color == "B")
    aw = "".join(f"[{sgf_coord(x, y)}]" for (x, y), color in ordered if color == "W")

    props = [
        "FF[4]",
        "GM[1]",
        f"SZ[{sgf_size(xsize, ysize)}]",
        "PB[random-initial]",
        "PW[random-initial]",
    ]
    if ab:
        props.append("AB" + ab)
    if aw:
        props.append("AW" + aw)
    props.append(f"PL[{next_pla}]")

    comment = (
        f"sample={sample_idx}, seed={seed}, opening={opening_name}, ok={int(ok)}, "
        f"size={xsize}x{ysize}, stonesB={sum(1 for c in stones.values() if c == 'B')}, "
        f"stonesW={sum(1 for c in stones.values() if c == 'W')}, legalArea={len(legal)}, msg={msg}"
    )
    props.append("C[" + sgf_escape(comment) + "]")
    return "(;" + "".join(props) + ")\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate .sgfs samples for KataGo random specified initial board openings."
    )
    parser.add_argument("--output", default="random_initial_board_samples.sgfs")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--b-sizes",
        default="7,8,9,10,11,12,13,14,15,16,17,18,19",
    )
    parser.add_argument(
        "--b-size-rel-probs",
        default="1,2,10,20,30,40,50,70,90,110,130,150,400",
    )
    parser.add_argument("--allow-rectangle-prob", type=float, default=0.50)
    args = parser.parse_args()

    b_sizes = parse_ints(args.b_sizes)
    rel_probs = parse_floats(args.b_size_rel_probs)
    if len(b_sizes) != len(rel_probs):
        raise ValueError("--b-sizes and --b-size-rel-probs must have the same length")

    rng = random.Random(args.seed)

    def choose_edge():
        return rng.choices(b_sizes, weights=rel_probs, k=1)[0]

    def choose_size():
        if rng.random() < args.allow_rectangle_prob:
            return choose_edge(), choose_edge()
        edge = choose_edge()
        return edge, edge

    records = []
    failures = 0
    for i in range(args.num_samples):
        xsize, ysize = choose_size()
        opening = rng.randrange(3)
        if opening in (0, 1):
            ok, stones, legal, msg, next_pla = make_para_opening(rng, xsize, ysize, opening)
        else:
            ok, stones, legal, msg, next_pla = make_hex_opening(rng, xsize, ysize)
        failures += 0 if ok else 1
        records.append(sgf_record(i, args.seed, xsize, ysize, opening, ok, stones, legal, msg, next_pla))

    output = Path(args.output)
    output.write_text("".join(records), encoding="utf-8")
    print(f"wrote={output}")
    print(f"samples={args.num_samples} failures={failures}")


if __name__ == "__main__":
    main()
