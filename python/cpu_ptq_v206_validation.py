#!/usr/bin/env python3
"""Prepare raw CPU-PTQ inputs and score native p0/value logits.

The engine benchmark format is deliberately headerless little-endian FP32:
each input row has 22*7*7 NCHW spatial values followed by 19 globals, and
each output row has 50 policy logits, 3 value logits, 4 misc logits, and 2
more-misc logits. This utility keeps the fixed NPZ as the source of truth.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


BOARD_SIZE = 7
SPATIAL_CHANNELS = 22
GLOBAL_CHANNELS = 19
POLICY_SIZE = 50
VALUE_SIZE = 3
INPUT_FLOATS = SPATIAL_CHANNELS * BOARD_SIZE * BOARD_SIZE + GLOBAL_CHANNELS
OUTPUT_FLOATS = POLICY_SIZE + VALUE_SIZE + 4 + 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_inputs(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        packed = archive["binaryInputNCHWPacked"]
        global_input = archive["globalInputNC"]
        rows = packed.shape[0] if args.rows is None else args.rows
        if not (0 < rows <= packed.shape[0]):
            raise ValueError(f"--rows must be in [1,{packed.shape[0]}]")
        if packed.shape[1:] != (SPATIAL_CHANNELS, 7):
            raise ValueError(f"unexpected packed input shape {packed.shape}")
        if global_input.shape != (packed.shape[0], GLOBAL_CHANNELS):
            raise ValueError(f"unexpected global input shape {global_input.shape}")

        with destination.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                compresslevel=args.compression_level,
                mtime=0,
            ) as output:
                for start in range(0, rows, args.chunk_rows):
                    end = min(rows, start + args.chunk_rows)
                    spatial = np.unpackbits(
                        packed[start:end], axis=2, count=BOARD_SIZE * BOARD_SIZE
                    ).astype("<f4", copy=False)
                    combined = np.concatenate(
                        (
                            spatial.reshape(end - start, -1),
                            np.asarray(global_input[start:end], dtype="<f4"),
                        ),
                        axis=1,
                    )
                    if combined.shape[1] != INPUT_FLOATS:
                        raise AssertionError(combined.shape)
                    output.write(combined.tobytes(order="C"))

    report = {
        "kind": "cpu-ptq-v206-native-inputs-v1",
        "source": str(source),
        "sourceSha256": sha256_file(source),
        "output": str(destination),
        "outputSha256": sha256_file(destination),
        "rows": rows,
        "floatsPerRow": INPUT_FLOATS,
        "uncompressedBytes": rows * INPUT_FLOATS * 4,
        "compressedBytes": destination.stat().st_size,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def log_softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = maximum + np.log(
        np.sum(np.exp(logits - maximum), axis=1, keepdims=True)
    )
    return np.sum(targets * (log_sum_exp - logits), axis=1)


def score_outputs(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    output_path = args.output.resolve()
    raw = np.fromfile(output_path, dtype="<f4")
    if raw.size % OUTPUT_FLOATS != 0:
        raise ValueError(
            f"{output_path} has {raw.size} floats, not a multiple of {OUTPUT_FLOATS}"
        )
    output = raw.reshape(-1, OUTPUT_FLOATS)
    rows = output.shape[0]
    with np.load(source, allow_pickle=False) as archive:
        if rows > archive["globalTargetsNC"].shape[0]:
            raise ValueError("native output has more rows than the source NPZ")
        policy_targets = np.asarray(
            archive["policyTargetsNCMove"][:rows, 0, :], dtype=np.float64
        )
        global_targets = np.asarray(
            archive["globalTargetsNC"][:rows], dtype=np.float64
        )
    policy_targets /= np.sum(policy_targets, axis=1, keepdims=True)
    global_weight = global_targets[:, 25]
    weight_sum = np.sum(global_weight)
    policy_weight = global_targets[:, 26]
    value_weight = 1.0 - global_targets[:, 35]
    policy_loss = log_softmax_cross_entropy(
        output[:, :POLICY_SIZE].astype(np.float64), policy_targets
    )
    value_loss = log_softmax_cross_entropy(
        output[:, POLICY_SIZE : POLICY_SIZE + VALUE_SIZE].astype(np.float64),
        global_targets[:, :VALUE_SIZE],
    )
    report = {
        "kind": "cpu-ptq-v206-native-loss-v1",
        "data": str(source),
        "dataSha256": sha256_file(source),
        "nativeOutput": str(output_path),
        "nativeOutputSha256": sha256_file(output_path),
        "samples": rows,
        "weightSum": float(weight_sum),
        "p0LossPerWeight": float(
            np.sum(global_weight * policy_weight * policy_loss) / weight_sum
        ),
        "valueLossPerWeight": float(
            1.20 * np.sum(global_weight * value_weight * value_loss) / weight_sum
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack-inputs")
    pack_parser.add_argument("--data", type=Path, required=True)
    pack_parser.add_argument("--output", type=Path, required=True)
    pack_parser.add_argument("--rows", type=int)
    pack_parser.add_argument("--chunk-rows", type=int, default=1024)
    pack_parser.add_argument("--compression-level", type=int, default=1)
    pack_parser.set_defaults(function=pack_inputs)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--data", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(function=score_outputs)

    args = parser.parse_args()
    if getattr(args, "chunk_rows", 1) <= 0:
        raise ValueError("--chunk-rows must be positive")
    if not 0 <= getattr(args, "compression_level", 1) <= 9:
        raise ValueError("--compression-level must be in [0,9]")
    args.function(args)


if __name__ == "__main__":
    main()
