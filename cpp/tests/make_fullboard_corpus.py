#!/usr/bin/env python3
"""Convert labeled KataGo NPZ rows into an identity-bound FBVCORP1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"FBVCORP1"
SPATIAL_FEATURES = 22
GLOBAL_FEATURES = 39
GLOBAL_TARGET_DIM = 64


def little_endian_contiguous(values: np.ndarray, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.dtype(dtype))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source_path: Path, output_path: Path, board: int, rows: int) -> dict[str, object]:
    if board not in (15, 19):
        raise ValueError("--board must be 15 or 19")
    area = board * board
    packed_width = (area + 7) // 8
    policy_dim = area + 1
    with np.load(source_path) as source:
        required = {
            "binaryInputNCHWPacked",
            "globalInputNC",
            "policyTargetsNCMove",
            "globalTargetsNC",
        }
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"source NPZ is missing arrays: {sorted(missing)}")
        packed_all = source["binaryInputNCHWPacked"]
        global_all = source["globalInputNC"]
        policy_all = source["policyTargetsNCMove"]
        target_all = source["globalTargetsNC"]
        expected_shapes = {
            "binaryInputNCHWPacked": (SPATIAL_FEATURES, packed_width),
            "globalInputNC": (GLOBAL_FEATURES,),
            "policyTargetsNCMove": (2, policy_dim),
            "globalTargetsNC": (GLOBAL_TARGET_DIM,),
        }
        arrays = {
            "binaryInputNCHWPacked": packed_all,
            "globalInputNC": global_all,
            "policyTargetsNCMove": policy_all,
            "globalTargetsNC": target_all,
        }
        population = packed_all.shape[0]
        for name, values in arrays.items():
            if values.shape[0] != population or values.shape[1:] != expected_shapes[name]:
                raise ValueError(
                    f"{name} shape {values.shape} does not match expected trailing "
                    f"shape {expected_shapes[name]}"
                )

        mask = np.unpackbits(packed_all[:, 0, :], axis=1)[:, :area]
        selected = np.flatnonzero(mask.sum(axis=1) == area)
        if rows <= 0:
            rows = int(selected.size)
        if selected.size < rows:
            raise ValueError(
                f"only {selected.size} full-board rows are available, requested {rows}"
            )
        selected = selected[:rows]
        packed = little_endian_contiguous(packed_all[selected], "u1")
        global_input = little_endian_contiguous(global_all[selected], "<f4")
        policy = little_endian_contiguous(policy_all[selected], "<i2")
        global_target = little_endian_contiguous(target_all[selected], "<f4")

    header = struct.pack(
        "<7I",
        rows,
        board,
        SPATIAL_FEATURES,
        GLOBAL_FEATURES,
        packed_width,
        policy_dim,
        GLOBAL_TARGET_DIM,
    )
    payloads = (
        packed.tobytes(order="C"),
        global_input.tobytes(order="C"),
        policy.tobytes(order="C"),
        global_target.tobytes(order="C"),
    )
    identity_hash = hashlib.sha256()
    identity_hash.update(MAGIC)
    identity_hash.update(header)
    for payload in payloads:
        identity_hash.update(payload)
    identity = identity_hash.digest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(MAGIC)
        output.write(header)
        output.write(identity)
        for payload in payloads:
            output.write(payload)

    index_bytes = little_endian_contiguous(selected, "<u8").tobytes(order="C")
    return {
        "format": MAGIC.decode("ascii"),
        "source": str(source_path.resolve()),
        "sourceSha256": file_sha256(source_path),
        "output": str(output_path.resolve()),
        "outputSha256": file_sha256(output_path),
        "corpusId": identity.hex(),
        "board": board,
        "numRows": rows,
        "selection": "first full-board rows in source order",
        "firstSourceRow": int(selected[0]),
        "lastSourceRow": int(selected[-1]),
        "sourceIndicesSha256": hashlib.sha256(index_bytes).hexdigest(),
        "allMaskSums": [area],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board", type=int, required=True)
    parser.add_argument(
        "--rows",
        type=int,
        default=8192,
        help="first N full-board rows; use 0 for all full-board rows",
    )
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = convert(args.npz, args.output, args.board, args.rows)
    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
