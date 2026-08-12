#!/usr/bin/env python3
"""Compare exporter NumPy PT bytes with the production C++ quantizer gate."""

import argparse
import json
import struct
import subprocess

import numpy as np


WEIGHTS = np.asarray(
    [
        -4.25, -2.5, -1.5, -0.5, 0.5,
        1.5, 2.5, 3.75, 4.25, -3.125,
        0.0, 0.03125, -0.03125, 2.0, -2.0,
        1.0, -1.0, 0.25, -0.25, 3.0,
    ],
    dtype=np.float32,
).reshape(4, 5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", help="path to katago_renju15_int8_weight_gate")
    args = parser.parse_args()

    cpp = json.loads(subprocess.check_output([args.gate], text=True))
    max_abs = np.max(np.abs(WEIGHTS)).astype(np.float32)
    scale = np.float32(max_abs / np.float32(127.0))
    quantized = np.clip(np.rint(WEIGHTS / scale), -127, 127).astype(np.int8)
    # CUTLASS B is ColumnMajor [K,N], equivalent to N-major packed bytes.
    packed = quantized.T.copy().reshape(-1)
    scale_bits = struct.unpack("<I", scale.tobytes())[0]
    expected = [int(value) for value in packed]
    if cpp["scale_bits"] != scale_bits or cpp["packed"] != expected:
        raise SystemExit(
            f"mismatch: cpp={cpp}, numpy={{'scale_bits': {scale_bits}, "
            f"'packed': {expected}}}"
        )
    print(json.dumps({"pass": True, "scale_bits": scale_bits, "bytes": len(expected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
