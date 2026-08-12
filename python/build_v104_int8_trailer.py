#!/usr/bin/env python3
"""Convert a native v102 transformer model to explicit-INT8 native v104.

This is a CPU-only bridge for the production training exporter. The v102 body
and every FP32 master stay byte-for-byte intact except for the fixed-width
version token (102 -> 104). A mandatory, hash-bound trailer carries the 24 QK,
24 FFN-up, and 24 FFN-gate per-tensor signed-int8 matrices.

The authoritative training exporter should call ``convert`` only when its
explicit ``-int8-pt-clip4`` flag is present. There is intentionally no implicit
upgrade path here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np


TRAILER_MARKER = b"@KATAGO_QUANT_TRAILER@"
BINARY_MARKER = b"@BIN@"
HEADER_SCHEMA = 1
PAYLOAD_MAGIC = b"KQPT104\0"
PAYLOAD_SCHEMA = 1
ENTRY_SCHEMA = 1
ENTRY_COUNT = 72
ROLE_QK = 1
ROLE_FFN_UP = 2
ROLE_FFN_GATE = 3
LAYOUT_OUTPUT_MAJOR_K_CONTIGUOUS = 1
QUANT_SYMMETRIC_SIGNED_INT8_PER_TENSOR = 1
ROUND_TIES_TO_EVEN_SATURATE_127 = 1
CLIP = np.float32(4.0)
ACTIVATION_SCALE = np.float32(CLIP / np.float32(127.0))


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _i32(value: int) -> bytes:
    return struct.pack("<i", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _f32_bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _name(value: str) -> bytes:
    raw = value.encode("utf-8")
    if not raw or len(raw) > 4096 or b"\0" in raw:
        raise ValueError(f"invalid native layer name {value!r}")
    return _u32(len(raw)) + raw


def _require_native_suffix(path: Path, field: str) -> None:
    name = path.name.lower()
    if not (name.endswith(".bin") or name.endswith(".bin.gz")):
        raise ValueError(f"{field} must end in .bin or .bin.gz")


def _read_native(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    gzip_magic = raw.startswith(b"\x1f\x8b")
    gzip_suffix = path.name.lower().endswith(".gz")
    if gzip_magic != gzip_suffix:
        raise ValueError("input .gz suffix and gzip magic disagree")
    if gzip_magic:
        return gzip.decompress(raw), artifact_sha256
    return raw, artifact_sha256


def _write_native(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            if path.name.lower().endswith(".gz"):
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
                    out.write(data)
            else:
                raw.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            out.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class Matrix:
    name: str
    k: int
    n: int
    values: Optional[np.ndarray]


@dataclass(frozen=True)
class Target:
    topology_index: int
    role: int
    matrices: tuple[Matrix, ...]


class NativeV102Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.stream: BinaryIO = io.BytesIO(data)
        self.version_span = (-1, -1)
        self.targets: list[Target] = []

    def token_with_span(self) -> tuple[str, int, int]:
        first = self.stream.read(1)
        while first and first.isspace():
            first = self.stream.read(1)
        if not first:
            raise EOFError("unexpected EOF while reading native model token")
        start = self.stream.tell() - 1
        raw = bytearray(first)
        byte = self.stream.read(1)
        while byte and not byte.isspace():
            raw.extend(byte)
            byte = self.stream.read(1)
        end = start + len(raw)
        try:
            return raw.decode("ascii"), start, end
        except UnicodeDecodeError as exc:
            raise ValueError("non-ASCII native model token") from exc

    def token(self) -> str:
        return self.token_with_span()[0]

    def integer(self) -> int:
        return int(self.token())

    def floating(self) -> float:
        value = float(self.token())
        if not np.isfinite(value):
            raise ValueError("non-finite scalar in native model")
        return value

    def weights(self, count: int, retain: bool) -> Optional[np.ndarray]:
        if count <= 0:
            raise ValueError("nonpositive native weight count")
        byte = self.stream.read(1)
        whitespace = 0
        while byte != b"@":
            if not byte or not byte.isspace() or whitespace >= 100:
                raise ValueError("missing @BIN@ native float marker")
            whitespace += 1
            byte = self.stream.read(1)
        if self.stream.read(4) != b"BIN@":
            raise ValueError("invalid @BIN@ native float marker")
        raw = self.stream.read(count * 4)
        if len(raw) != count * 4:
            raise EOFError("truncated native float block")
        values = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("non-finite native FP32 master")
        return values.copy() if retain else None

    def conv(self) -> None:
        self.token()
        y, x, k, n, dy, dx = [self.integer() for _ in range(6)]
        if min(y, x, k, n, dy, dx) <= 0 or y % 2 == 0 or x % 2 == 0:
            raise ValueError("invalid native convolution descriptor")
        self.weights(y * x * k * n, False)

    def matmul(self, retain: bool = False) -> Matrix:
        name = self.token()
        k, n = self.integer(), self.integer()
        if k <= 0 or n <= 0:
            raise ValueError(f"{name}: invalid native matmul descriptor")
        values = self.weights(k * n, retain)
        if values is not None:
            values = values.reshape(k, n)
        return Matrix(name, k, n, values)

    def matbias(self) -> None:
        self.token()
        channels = self.integer()
        self.weights(channels, False)

    def batch_norm(self) -> None:
        self.token()
        channels = self.integer()
        epsilon = self.floating()
        has_scale, has_bias = self.integer(), self.integer()
        if channels <= 0 or epsilon <= 0 or has_scale not in (0, 1) or has_bias not in (0, 1):
            raise ValueError("invalid native batch-normalization descriptor")
        self.weights(channels, False)
        self.weights(channels, False)
        if has_scale:
            self.weights(channels, False)
        if has_bias:
            self.weights(channels, False)

    def activation(self) -> None:
        self.token()
        kind = self.token()
        if kind not in {
            "ACTIVATION_IDENTITY", "ACTIVATION_RELU",
            "ACTIVATION_MISH", "ACTIVATION_SILU",
        }:
            raise ValueError(f"unsupported native activation {kind!r}")

    def transformer_norm(self) -> int:
        self.token()
        channels = self.integer()
        epsilon = self.floating()
        if channels <= 0 or epsilon <= 0:
            raise ValueError("invalid transformer RMSNorm descriptor")
        self.weights(channels, False)
        return channels

    def attention(self, topology_index: int) -> None:
        self.token()  # block name
        heads, kv_heads, q_dim, v_dim = [self.integer() for _ in range(4)]
        use_rope, learned_rope = self.integer(), self.integer()
        norm_channels = self.transformer_norm()
        q = self.matmul(True)
        k = self.matmul(True)
        v = self.matmul(False)
        out = self.matmul(False)
        if (
            heads != 8 or kv_heads != 8 or q_dim != 32 or v_dim != 32
            or use_rope != 1 or learned_rope != 1
            or norm_channels != 256
            or (q.k, q.n) != (256, 256)
            or (k.k, k.n) != (256, 256)
            or (v.k, v.n) != (256, 256)
            or (out.k, out.n) != (256, 256)
        ):
            raise ValueError("v104 explicit INT8 exporter requires C256/H8 learned-RoPE attention")
        rope_name = self.token()
        rope_heads, rope_pairs, coordinates = [self.integer() for _ in range(3)]
        if (rope_heads, rope_pairs, coordinates) != (8, 16, 2) or not rope_name:
            raise ValueError("invalid learned-RoPE tensor for v104")
        self.weights(rope_heads * rope_pairs * coordinates, False)
        self.targets.append(Target(topology_index, ROLE_QK, (q, k)))

    def ffn(self, topology_index: int) -> None:
        self.token()  # block name
        channels, ffn_channels, swiglu = [self.integer() for _ in range(3)]
        norm_channels = self.transformer_norm()
        up = self.matmul(True)
        gate = self.matmul(True) if swiglu else None
        down = self.matmul(False)
        if (
            channels != 256 or ffn_channels != 768 or swiglu != 1
            or norm_channels != 256
            or (up.k, up.n) != (256, 768)
            or gate is None or (gate.k, gate.n) != (256, 768)
            or (down.k, down.n) != (768, 256)
        ):
            raise ValueError("v104 explicit INT8 exporter requires C256/F768 SwiGLU FFN")
        self.targets.append(Target(topology_index, ROLE_FFN_UP, (up,)))
        self.targets.append(Target(topology_index, ROLE_FFN_GATE, (gate,)))

    def parse(self) -> tuple[bytes, list[Target]]:
        self.token()  # model name
        version, start, end = self.token_with_span()
        self.version_span = (start, end)
        if version != "102" or end - start != 3:
            raise ValueError("input must be a native v102 model")
        self.integer()  # spatial features
        self.integer()  # global features

        if self.token() != "trunk":
            raise ValueError("unexpected native trunk name")
        block_count = self.integer()
        self.integer()  # trunk channels
        self.integer()  # mid channels
        self.integer()  # regular channels
        self.integer()  # dilated channels
        self.integer()  # gpool channels
        self.conv()       # topology 0
        self.matmul()     # topology 1
        topology_index = 2
        attention_count = 0
        ffn_count = 0
        for _ in range(block_count):
            kind = self.token()
            if kind == "transformer_attention_block":
                self.attention(topology_index)
                attention_count += 1
            elif kind == "transformer_ffn_block":
                self.ffn(topology_index)
                ffn_count += 1
            else:
                raise ValueError(f"v104 bridge rejects unsupported trunk block {kind!r}")
            topology_index += 1
        self.batch_norm()
        self.activation()

        self.token()  # policy head name
        self.conv()
        self.conv()
        self.batch_norm()
        self.activation()
        self.matmul()
        self.batch_norm()
        self.activation()
        self.conv()
        self.matmul()

        self.token()  # value head name
        self.conv()
        self.batch_norm()
        self.activation()
        self.matmul()
        self.matbias()
        self.activation()
        self.matmul()
        self.matbias()
        self.matmul()
        self.matbias()
        self.conv()

        tail = self.stream.read()
        if not tail or tail.strip():
            raise ValueError("native v102 body must end in whitespace and strict EOF")
        if attention_count != 24 or ffn_count != 24 or len(self.targets) != ENTRY_COUNT:
            raise ValueError("v104 explicit INT8 requires 24 attention + 24 FFN blocks and 72 entries")
        upgraded = bytearray(self.data)
        upgraded[start:end] = b"104"
        return bytes(upgraded), self.targets


def _canonical_master(target: Target) -> np.ndarray:
    matrices = target.matrices
    if any(matrix.values is None for matrix in matrices):
        raise AssertionError("retained target matrix has no values")
    if len(matrices) == 1:
        return np.asarray(matrices[0].values, dtype="<f4", order="C")
    if len(matrices) == 2 and matrices[0].k == matrices[1].k:
        # Logical [K,N] concat: each K row contains all Q outputs, then K.
        return np.ascontiguousarray(
            np.concatenate((matrices[0].values, matrices[1].values), axis=1),
            dtype="<f4",
        )
    raise ValueError("invalid QK canonical concatenation")


def _quantize(target: Target) -> tuple[np.float32, bytes, bytes, bytes]:
    master = _canonical_master(target)
    max_abs = np.max(np.abs(master), initial=np.float32(0.0)).astype(np.float32)
    scale = np.float32(max_abs / np.float32(127.0))
    scale = np.maximum(scale, np.finfo(np.float32).tiny).astype(np.float32)
    if not np.isfinite(scale) or not scale > 0:
        raise ValueError("invalid per-tensor INT8 weight scale")
    # Both operands remain float32, matching C++ float division before its
    # explicit round-ties-to-even implementation.
    scaled = np.asarray(master / scale, dtype=np.float32)
    quantized = np.rint(scaled)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    packed = np.ascontiguousarray(quantized.T).tobytes(order="C")
    if b"\x80" in packed:
        raise AssertionError("symmetric signed-int8 contract emitted -128")
    master_bytes = np.ascontiguousarray(master, dtype="<f4").tobytes(order="C")
    return (
        scale,
        hashlib.sha256(master_bytes).digest(),
        hashlib.sha256(packed).digest(),
        packed,
    )


def build_payload(targets: list[Target]) -> tuple[bytes, list[dict[str, object]]]:
    if len(targets) != ENTRY_COUNT:
        raise ValueError("v104 payload requires exactly 72 targets")
    payload = bytearray(PAYLOAD_MAGIC)
    payload += _u32(PAYLOAD_SCHEMA)
    payload += _u32(len(targets))
    payload += _u32(QUANT_SYMMETRIC_SIGNED_INT8_PER_TENSOR)
    payload += _u32(QUANT_SYMMETRIC_SIGNED_INT8_PER_TENSOR)
    payload += _u32(ROUND_TIES_TO_EVEN_SATURATE_127)
    payload += _i32(0)
    payload += _u32(_f32_bits(CLIP))
    payload += _u32(_f32_bits(ACTIVATION_SCALE))
    payload += _u32(0) + _u32(0)
    manifest_entries: list[dict[str, object]] = []

    for target in targets:
        master = _canonical_master(target)
        k, n = master.shape
        scale, master_sha, packed_sha, packed = _quantize(target)
        names = [matrix.name for matrix in target.matrices]
        record = bytearray()
        record += _u32(ENTRY_SCHEMA)
        record += _u32(target.topology_index)
        record += _u32(target.role)
        record += _u32(LAYOUT_OUTPUT_MAJOR_K_CONTIGUOUS)
        record += _u32(k) + _u32(n)
        record += _i32(0)
        record += _u32(_f32_bits(ACTIVATION_SCALE))
        record += _u32(_f32_bits(scale))
        record += _u32(len(names))
        record += _u64(len(packed))
        for name in names:
            record += _name(name)
        record += master_sha + packed_sha + packed
        payload += _u64(len(record)) + record
        manifest_entries.append({
            "topology_index": target.topology_index,
            "role": {ROLE_QK: "qk", ROLE_FFN_UP: "ffn_up", ROLE_FFN_GATE: "ffn_gate"}[target.role],
            "layer_names": names,
            "k": k,
            "n": n,
            "layout": "output-major-k-contiguous",
            "zero_point": 0,
            "activation_clip": float(CLIP),
            "activation_scale_bits": f"0x{_f32_bits(ACTIVATION_SCALE):08x}",
            "weight_scale_bits": f"0x{_f32_bits(scale):08x}",
            "master_sha256": master_sha.hex(),
            "packed_sha256": packed_sha.hex(),
            "packed_bytes": len(packed),
        })
    return bytes(payload), manifest_entries


def convert(input_path: Path, output_path: Path, manifest_path: Optional[Path]) -> dict[str, object]:
    _require_native_suffix(input_path, "input native model")
    _require_native_suffix(output_path, "output native model")
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if input_resolved == output_resolved:
        raise ValueError("input and output must be different paths")
    if manifest_path is not None and manifest_path.resolve() in {
        input_resolved, output_resolved
    }:
        raise ValueError("manifest must be different from input and output paths")
    source, source_artifact_sha256 = _read_native(input_path)
    body_v104, targets = NativeV102Reader(source).parse()
    payload, entries = build_payload(targets)
    payload_sha = hashlib.sha256(payload).hexdigest()
    header = (
        TRAILER_MARKER + b" " + str(HEADER_SCHEMA).encode("ascii") + b" "
        + str(len(payload)).encode("ascii") + b" " + payload_sha.encode("ascii")
        + b" " + BINARY_MARKER
    )
    output = body_v104 + header + payload
    _write_native(output_path, output)
    output_artifact_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "status": "PASS",
        "format": "native-v104-int8-pt-clip4",
        "source_path": str(input_path),
        "source_artifact_sha256": source_artifact_sha256,
        "source_uncompressed_sha256": hashlib.sha256(source).hexdigest(),
        "v104_body_bytes": len(body_v104),
        "v104_body_sha256": hashlib.sha256(body_v104).hexdigest(),
        "output_path": str(output_path),
        "output_artifact_sha256": output_artifact_sha256,
        "output_uncompressed_sha256": hashlib.sha256(output).hexdigest(),
        "payload_schema": PAYLOAD_SCHEMA,
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
        "entry_count": len(entries),
        "entries": entries,
    }
    if manifest_path is not None:
        _write_text_atomic(manifest_path, json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append mandatory explicit clip4/PT INT8 metadata to a native v102 model"
    )
    parser.add_argument("-input-native-v102", required=True, type=Path)
    parser.add_argument("-output-native", required=True, type=Path)
    parser.add_argument("-manifest", type=Path)
    parser.add_argument(
        "-int8-pt-clip4", action="store_true",
        help="required explicit opt-in; without it no v104 artifact is emitted",
    )
    args = parser.parse_args()
    if not args.int8_pt_clip4:
        parser.error("-int8-pt-clip4 is required; default exporter output remains native v102")
    try:
        report = convert(args.input_native_v102, args.output_native, args.manifest)
    except (OSError, EOFError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in (
        "status", "output_path", "payload_bytes", "payload_sha256", "entry_count"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
