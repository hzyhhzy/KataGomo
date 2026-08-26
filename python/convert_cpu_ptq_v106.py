#!/usr/bin/env python3
"""Convert a strict native v105 Transformer model to CPU-PTQ v106.

Version 106 replaces each Transformer Q/K/V/O and SwiGLU up/gate/down FP32
matrix with a canonical per-output-channel symmetric S7 or S8 block. The
canonical layout is deliberately independent of MLAS: scales are little-endian
FP32 and weights are output-major with input channels contiguous. The CPU
backend only has to transpose and pack these already-quantized weights at load
time.

All other native weights remain byte-for-byte identical to v105. This tool is
strictly scoped to the three compiled CPU profiles and validates the complete
native body before writing an atomic, deterministic .bin.gz artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import pathlib
import tempfile
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


FP32_MARKER = b"@BIN@"
S8_MARKER = b"@S8P@"
S7_MARKER = b"@S7P@"


@dataclass(frozen=True)
class Profile:
    name: str
    logical_blocks: int
    transformer_layers: int
    channels: int
    heads: int
    head_dim: int
    ffn_channels: int
    value_hidden_channels: int


PROFILES = (
    Profile("b11c96h3-f256", 22, 11, 96, 3, 32, 256, 64),
    Profile("b16c128h4-f384", 32, 16, 128, 4, 32, 384, 96),
    Profile("b24c192h6-f512", 48, 24, 192, 6, 32, 512, 96),
)


@dataclass(frozen=True)
class Matrix:
    name: str
    inputs: int
    outputs: int
    values: np.ndarray | None
    payload_span: tuple[int, int] | None


@dataclass(frozen=True)
class QuantizedMatrix:
    name: str
    inputs: int
    outputs: int
    quantized_max: int
    scales: np.ndarray
    values: np.ndarray


class NativeReader:
    """Shape-checking reader for one complete native v105 or v106 body."""

    def __init__(self, data: bytes, expected_version: int):
        self.data = data
        self.stream: BinaryIO = io.BytesIO(data)
        self.expected_version = expected_version
        self.version_span = (-1, -1)
        self.projections: list[Matrix] = []
        self.quantized_projections: list[QuantizedMatrix] = []
        self.quantized_projection_count = 0
        self.profile: Profile | None = None

    def token_with_span(self) -> tuple[str, int, int]:
        byte = self.stream.read(1)
        while byte and byte.isspace():
            byte = self.stream.read(1)
        if not byte:
            raise EOFError("unexpected EOF while reading native model token")
        start = self.stream.tell() - 1
        raw = bytearray(byte)
        byte = self.stream.read(1)
        while byte and not byte.isspace():
            raw.extend(byte)
            byte = self.stream.read(1)
        try:
            return raw.decode("ascii"), start, start + len(raw)
        except UnicodeDecodeError as exc:
            raise ValueError("non-ASCII native model token") from exc

    def token(self) -> str:
        return self.token_with_span()[0]

    def integer(self) -> int:
        return int(self.token())

    def flag(self, field: str) -> bool:
        value = self.integer()
        if value not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1")
        return bool(value)

    def floating(
        self,
        field: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> float:
        value = float(self.token())
        if not np.isfinite(value):
            raise ValueError(f"{field} must be finite")
        if positive and not value > 0.0:
            raise ValueError(f"{field} must be positive")
        if nonnegative and value < 0.0:
            raise ValueError(f"{field} must be nonnegative")
        return value

    def _marker_start(self, expected: bytes, field: str) -> int:
        byte = self.stream.read(1)
        whitespace = 0
        while byte != b"@":
            if not byte or not byte.isspace() or whitespace >= 100:
                raise ValueError(f"{field}: missing {expected.decode()} marker")
            whitespace += 1
            byte = self.stream.read(1)
        start = self.stream.tell() - 1
        marker = byte + self.stream.read(len(expected) - 1)
        if marker != expected:
            raise ValueError(
                f"{field}: expected {expected.decode()}, got {marker!r}"
            )
        return start

    def fp32_weights(
        self, count: int, field: str, *, retain: bool = False
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        if count <= 0:
            raise ValueError(f"{field}: nonpositive native weight count")
        start = self._marker_start(FP32_MARKER, field)
        raw = self.stream.read(count * 4)
        if len(raw) != count * 4:
            raise EOFError(f"{field}: truncated native FP32 block")
        values = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"{field}: non-finite native FP32 weight")
        return (values.copy() if retain else None), (start, self.stream.tell())

    def quantized_weights(self, inputs: int, outputs: int, field: str) -> None:
        marker_start = self.stream.tell()
        try:
            self._marker_start(S8_MARKER, field)
            quantized_max = 127
        except ValueError:
            self.stream.seek(marker_start)
            self._marker_start(S7_MARKER, field)
            quantized_max = 63
        scale_raw = self.stream.read(outputs * 4)
        if len(scale_raw) != outputs * 4:
            raise EOFError(f"{field}: truncated v106 scale block")
        scales = np.frombuffer(scale_raw, dtype="<f4")
        if not np.isfinite(scales).all() or not np.all(scales > 0.0):
            raise ValueError(f"{field}: v106 scales must be finite and positive")
        quantized = self.stream.read(inputs * outputs)
        if len(quantized) != inputs * outputs:
            raise EOFError(f"{field}: truncated v106 S7/S8 block")
        if b"\x80" in quantized:
            raise ValueError(f"{field}: symmetric v106 block contains -128")
        self.quantized_projections.append(
            QuantizedMatrix(
                field,
                inputs,
                outputs,
                quantized_max,
                scales.copy(),
                np.frombuffer(quantized, dtype=np.int8)
                .reshape(outputs, inputs)
                .copy(),
            )
        )
        maximum_code = int(np.max(np.abs(
            self.quantized_projections[-1].values.astype(np.int16)
        )))
        if maximum_code > quantized_max:
            raise ValueError(f"{field}: quantized code exceeds declared S7/S8 range")
        self.quantized_projection_count += 1

    def conv(self) -> tuple[int, int, int, int, int, int]:
        name = self.token()
        dims = tuple(self.integer() for _ in range(6))
        y, x, inputs, outputs, dilation_y, dilation_x = dims
        if min(dims) <= 0 or y % 2 == 0 or x % 2 == 0:
            raise ValueError(f"{name}: invalid native convolution descriptor")
        self.fp32_weights(y * x * inputs * outputs, name)
        return dims

    def matmul(self, *, projection: bool = False) -> Matrix:
        name = self.token()
        inputs, outputs = self.integer(), self.integer()
        if inputs <= 0 or outputs <= 0:
            raise ValueError(f"{name}: invalid native matmul descriptor")
        if projection and self.expected_version == 106:
            self.quantized_weights(inputs, outputs, name)
            return Matrix(name, inputs, outputs, None, None)
        values, span = self.fp32_weights(
            inputs * outputs, name, retain=projection
        )
        if values is not None:
            values = values.reshape(inputs, outputs)
        matrix = Matrix(name, inputs, outputs, values, span)
        if projection:
            self.projections.append(matrix)
        return matrix

    def matbias(self) -> int:
        name = self.token()
        channels = self.integer()
        if channels <= 0:
            raise ValueError(f"{name}: invalid native bias descriptor")
        self.fp32_weights(channels, name)
        return channels

    def batch_norm(self) -> int:
        name = self.token()
        channels = self.integer()
        epsilon = self.floating(name + ".epsilon", positive=True)
        has_scale = self.flag(name + ".hasScale")
        has_bias = self.flag(name + ".hasBias")
        if channels <= 0 or epsilon > 1.0:
            raise ValueError(
                f"{name}: invalid native batch-normalization descriptor"
            )
        self.fp32_weights(channels, name + ".mean")
        self.fp32_weights(channels, name + ".variance")
        if has_scale:
            self.fp32_weights(channels, name + ".scale")
        if has_bias:
            self.fp32_weights(channels, name + ".bias")
        return channels

    def activation(self) -> None:
        name = self.token()
        kind = self.token()
        if kind not in {
            "ACTIVATION_IDENTITY",
            "ACTIVATION_RELU",
            "ACTIVATION_MISH",
            "ACTIVATION_SILU",
        }:
            raise ValueError(f"{name}: unsupported native activation {kind!r}")

    def transformer_norm(self) -> int:
        name = self.token()
        channels = self.integer()
        epsilon = self.floating(name + ".epsilon", positive=True)
        if channels <= 0 or epsilon > 1.0:
            raise ValueError(f"{name}: invalid Transformer RMSNorm descriptor")
        self.fp32_weights(channels, name)
        return channels

    def attention(self, profile: Profile) -> None:
        name = self.token()
        heads, kv_heads, q_dim, v_dim = [self.integer() for _ in range(4)]
        use_rope = self.flag(name + ".useRope")
        learned_rope = self.flag(name + ".learnedRope")
        use_qk_norm = self.flag(name + ".useQKNorm")
        self.floating(name + ".attentionInputQuantMaxAbs", positive=True)
        self.floating(name + ".attentionOutputQuantMaxAbs", positive=True)
        norm_channels = self.transformer_norm()
        q = self.matmul(projection=True)
        k = self.matmul(projection=True)
        v = self.matmul(projection=True)
        out = self.matmul(projection=True)
        if use_qk_norm:
            q_norm = self.transformer_norm()
            k_norm = self.transformer_norm()
            if q_norm != q_dim or k_norm != q_dim:
                raise ValueError(f"{name}: QK norm geometry mismatch")

        expected = (
            heads == profile.heads
            and kv_heads == profile.heads
            and q_dim == profile.head_dim
            and v_dim == profile.head_dim
            and norm_channels == profile.channels
            and (q.inputs, q.outputs) == (profile.channels, profile.channels)
            and (k.inputs, k.outputs) == (profile.channels, profile.channels)
            and (v.inputs, v.outputs) == (profile.channels, profile.channels)
            and (out.inputs, out.outputs) == (profile.channels, profile.channels)
        )
        if not expected:
            raise ValueError(f"{name}: attention does not match {profile.name}")
        if learned_rope and not use_rope:
            raise ValueError(f"{name}: learned RoPE requires RoPE")
        if use_rope:
            if learned_rope:
                rope_name = self.token()
                rope_heads, rope_pairs, coordinates = [
                    self.integer() for _ in range(3)
                ]
                if (rope_heads, rope_pairs, coordinates) != (
                    profile.heads,
                    profile.head_dim // 2,
                    2,
                ):
                    raise ValueError(
                        f"{rope_name}: learned RoPE geometry mismatch"
                    )
                self.fp32_weights(
                    rope_heads * rope_pairs * coordinates, rope_name
                )
            else:
                theta_name = self.token()
                self.floating(theta_name, positive=True)

    def ffn(self, profile: Profile) -> None:
        name = self.token()
        channels, ffn_channels = self.integer(), self.integer()
        use_swiglu = self.flag(name + ".useSwiGLU")
        self.floating(name + ".swigluClip", nonnegative=True)
        self.floating(name + ".ffnInputQuantMaxAbs", positive=True)
        self.floating(name + ".productQuantMaxAbs", positive=True)
        norm_channels = self.transformer_norm()
        up = self.matmul(projection=True)
        gate = self.matmul(projection=True) if use_swiglu else None
        down = self.matmul(projection=True)
        expected = (
            channels == profile.channels
            and ffn_channels == profile.ffn_channels
            and norm_channels == profile.channels
            and use_swiglu
            and (up.inputs, up.outputs)
            == (profile.channels, profile.ffn_channels)
            and gate is not None
            and (gate.inputs, gate.outputs)
            == (profile.channels, profile.ffn_channels)
            and (down.inputs, down.outputs)
            == (profile.ffn_channels, profile.channels)
        )
        if not expected:
            raise ValueError(f"{name}: FFN does not match {profile.name}")

    @staticmethod
    def _conv_is(
        dims: tuple[int, int, int, int, int, int],
        y: int,
        x: int,
        inputs: int,
        outputs: int,
    ) -> bool:
        return dims == (y, x, inputs, outputs, 1, 1)

    def policy_head(self, profile: Profile) -> None:
        self.token()
        p1 = self.conv()
        g1 = self.conv()
        g1_norm = self.batch_norm()
        self.activation()
        gpool_bias = self.matmul()
        p1_norm = self.batch_norm()
        self.activation()
        p2 = self.conv()
        pass_mul = self.matmul()
        if not (
            self._conv_is(p1, 1, 1, profile.channels, 32)
            and self._conv_is(g1, 1, 1, profile.channels, 32)
            and g1_norm == 32
            and (gpool_bias.inputs, gpool_bias.outputs) == (96, 32)
            and p1_norm == 32
            and self._conv_is(p2, 1, 1, 32, 1)
            and (pass_mul.inputs, pass_mul.outputs) == (96, 1)
        ):
            raise ValueError("policy head does not match the CPU-PTQ profile")

    def value_head(self, profile: Profile) -> None:
        self.token()
        v1 = self.conv()
        v1_norm = self.batch_norm()
        self.activation()
        v2 = self.matmul()
        v2_bias = self.matbias()
        self.activation()
        value = self.matmul()
        value_bias = self.matbias()
        score = self.matmul()
        score_bias = self.matbias()
        ownership = self.conv()
        hidden = profile.value_hidden_channels
        if not (
            self._conv_is(v1, 1, 1, profile.channels, 32)
            and v1_norm == 32
            and (v2.inputs, v2.outputs) == (96, hidden)
            and v2_bias == hidden
            and (value.inputs, value.outputs) == (hidden, 3)
            and value_bias == 3
            and (score.inputs, score.outputs) == (hidden, 6)
            and score_bias == 6
            and self._conv_is(ownership, 1, 1, 32, 1)
        ):
            raise ValueError("value head does not match the CPU-PTQ profile")

    def parse(self) -> Profile:
        model_name = self.token()
        if not model_name:
            raise ValueError("empty native model name")
        version, start, end = self.token_with_span()
        self.version_span = (start, end)
        if version != str(self.expected_version) or end - start != 3:
            raise ValueError(
                f"expected native v{self.expected_version}, got version {version!r}"
            )
        if self.integer() != 22 or self.integer() != 39:
            raise ValueError("CPU-PTQ v106 requires the v101 22/39 input ABI")
        if self.token() != "trunk":
            raise ValueError("unexpected native trunk name")
        logical_blocks = self.integer()
        channels = self.integer()
        mid, regular, dilated, gpool = [self.integer() for _ in range(4)]
        if min(mid, regular, dilated, gpool) <= 0:
            raise ValueError("invalid native trunk channel header")
        profile = next(
            (
                candidate
                for candidate in PROFILES
                if candidate.logical_blocks == logical_blocks
                and candidate.channels == channels
            ),
            None,
        )
        if profile is None:
            raise ValueError(
                f"unsupported CPU-PTQ profile: blocks={logical_blocks}, "
                f"C={channels}"
            )
        self.profile = profile

        stem = self.conv()
        global_mul = self.matmul()
        if not self._conv_is(stem, 3, 3, 22, channels):
            raise ValueError("stem does not match the CPU-PTQ profile")
        if (global_mul.inputs, global_mul.outputs) != (39, channels):
            raise ValueError(
                "global projection does not match the CPU-PTQ profile"
            )

        attention_count = 0
        ffn_count = 0
        for block_index in range(logical_blocks):
            kind = self.token()
            if (
                block_index % 2 == 0
                and kind == "transformer_attention_block"
            ):
                self.attention(profile)
                attention_count += 1
            elif block_index % 2 == 1 and kind == "transformer_ffn_block":
                self.ffn(profile)
                ffn_count += 1
            else:
                raise ValueError(
                    f"block {block_index}: expected alternating attention/FFN, "
                    f"got {kind!r}"
                )
        if self.batch_norm() != channels:
            raise ValueError("trunk tip norm does not match the CPU-PTQ profile")
        self.activation()
        self.policy_head(profile)
        self.value_head(profile)

        tail = self.stream.read()
        if tail.strip():
            raise ValueError("native model has trailing non-whitespace data")
        if (
            attention_count != profile.transformer_layers
            or ffn_count != profile.transformer_layers
        ):
            raise ValueError(
                "Transformer layer count does not match the CPU-PTQ profile"
            )
        expected_projections = profile.transformer_layers * 7
        actual_projections = (
            len(self.projections)
            if self.expected_version == 105
            else self.quantized_projection_count
        )
        if actual_projections != expected_projections:
            raise ValueError(
                f"expected {expected_projections} Transformer projections, "
                f"found {actual_projections}"
            )
        return profile


def _read_model(path: pathlib.Path) -> bytes:
    if path.name.endswith(".bin.gz"):
        with gzip.open(path, "rb") as model_file:
            return model_file.read()
    if path.suffix == ".bin":
        return path.read_bytes()
    raise ValueError("input must end in .bin or .bin.gz (ONNX is not accepted)")


def _quantize(matrix: Matrix, quantized_max: int) -> tuple[bytes, int, int]:
    if matrix.values is None or matrix.payload_span is None:
        raise AssertionError("source projection has no retained FP32 payload")
    values = np.asarray(matrix.values, dtype=np.float32, order="C")
    max_abs = np.max(np.abs(values), axis=0).astype(np.float32)
    scales = np.asarray(max_abs / np.float32(quantized_max), dtype=np.float32)
    scales[max_abs == np.float32(0.0)] = np.float32(1.0)
    if not np.isfinite(scales).all() or not np.all(scales > 0.0):
        raise ValueError(f"{matrix.name}: invalid per-output quantized scale")
    scaled = np.asarray(values / scales.reshape(1, -1), dtype=np.float32)
    quantized_kn = np.clip(
        np.rint(scaled), -quantized_max, quantized_max
    ).astype(np.int8)
    quantized_nk = np.ascontiguousarray(quantized_kn.T)
    packed = quantized_nk.tobytes(order="C")
    if b"\x80" in packed:
        raise AssertionError(f"{matrix.name}: symmetric quantizer emitted -128")
    replacement = (
        (S7_MARKER if quantized_max == 63 else S8_MARKER)
        + np.ascontiguousarray(scales, dtype="<f4").tobytes(order="C")
        + packed
    )
    return replacement, len(packed), scales.size * 4


def _replace(data: bytes, replacements: list[tuple[int, int, bytes]]) -> bytes:
    output = bytearray()
    cursor = 0
    for start, end, replacement in sorted(replacements):
        if start < cursor or end < start or end > len(data):
            raise ValueError("overlapping or invalid native payload replacement")
        output += data[cursor:start]
        output += replacement
        cursor = end
    output += data[cursor:]
    return bytes(output)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    source: pathlib.Path, destination: pathlib.Path, projection_bits: int = 8
) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise ValueError(f"source model does not exist: {source}")
    if source == destination:
        raise ValueError("source and destination must differ")
    if not destination.name.endswith(".bin.gz"):
        raise ValueError("destination must end in .bin.gz")
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing output: {destination}")
    if projection_bits not in (7, 8):
        raise ValueError("projection_bits must be 7 or 8")
    quantized_max = 63 if projection_bits == 7 else 127

    source_payload = _read_model(source)
    reader = NativeReader(source_payload, expected_version=105)
    profile = reader.parse()
    replacements: list[tuple[int, int, bytes]] = [
        (*reader.version_span, b"106")
    ]
    quantized_bytes = 0
    scale_bytes = 0
    fp32_projection_bytes = 0
    for matrix in reader.projections:
        replacement, matrix_quantized_bytes, matrix_scale_bytes = _quantize(
            matrix, quantized_max
        )
        assert matrix.payload_span is not None
        replacements.append((*matrix.payload_span, replacement))
        quantized_bytes += matrix_quantized_bytes
        scale_bytes += matrix_scale_bytes
        fp32_projection_bytes += matrix.inputs * matrix.outputs * 4

    destination_payload = _replace(source_payload, replacements)
    verified_profile = NativeReader(
        destination_payload, expected_version=106
    ).parse()
    if verified_profile != profile:
        raise AssertionError("v106 verification selected a different profile")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as raw_file:
            temporary_path = pathlib.Path(raw_file.name)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_file, mtime=0
            ) as gzip_file:
                gzip_file.write(destination_payload)
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    round_trip = _read_model(destination)
    if round_trip != destination_payload:
        destination.unlink(missing_ok=True)
        raise RuntimeError("v106 deterministic gzip round-trip verification failed")

    canonical_bytes = quantized_bytes + scale_bytes
    print(f"profile={profile.name}")
    print(f"projection_count={len(reader.projections)}")
    print(f"projection_bits={projection_bits}")
    print(f"projection_fp32_bytes={fp32_projection_bytes}")
    print(f"projection_quantized_bytes={quantized_bytes}")
    print(f"projection_scale_bytes={scale_bytes}")
    print(
        f"projection_storage_ratio="
        f"{canonical_bytes / fp32_projection_bytes:.9f}"
    )
    print(f"source_uncompressed_bytes={len(source_payload)}")
    print(f"v106_uncompressed_bytes={len(destination_payload)}")
    print(f"source_uncompressed_sha256={_sha256(source_payload)}")
    print(f"v106_uncompressed_sha256={_sha256(destination_payload)}")
    print(f"v106_file_sha256={_sha256_file(destination)}")
    print(f"output={destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize the seven Transformer projections per layer in a strict "
            "native v105 b11c96, b16c128, or b24c192 model and write "
            "CPU-PTQ v106"
        )
    )
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument(
        "--projection-bits", type=int, choices=(7, 8), default=8,
        help="symmetric per-output projection weight bit width (default: 8)",
    )
    args = parser.parse_args()
    convert(args.source, args.destination, args.projection_bits)


if __name__ == "__main__":
    main()
