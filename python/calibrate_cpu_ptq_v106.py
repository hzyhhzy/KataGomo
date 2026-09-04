#!/usr/bin/env python3
"""Generate GPTQ projection overrides for a native CPU-PTQ v106 model.

The output is an intermediate NPZ manifest consumed by
``convert_cpu_ptq_v106.py --gptq-overrides``. The final ``.bin.gz`` remains
board-size agnostic: the board size here only selects calibration activations
for a particular runtime profile.

The calibration forward pass mirrors the CPU-PTQ projection and attention
quantizers. Projection activations use one symmetric S8 scale per token. S8
weights use the VNNI Q8/K8/P8/V8 attention recipe; S7 weights use the AVX2
Q7/K8/P8/V7 recipe. GPTQ changes only the stored codes and per-output weight
scales, so it has no inference-time cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


PROJECTION_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "ffn_linear1",
    "ffn_linear_gate",
    "ffn_linear2",
)
GROUP_ROLES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "out": ("out_proj",),
    "upgate": ("ffn_linear1", "ffn_linear_gate"),
    "down": ("ffn_linear2",),
}
GROUP_HOOK = {
    "qkv": "q_proj",
    "out": "out_proj",
    "upgate": "ffn_linear1",
    "down": "ffn_linear2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibrationData:
    def __init__(
        self,
        path: Path,
        board_size: int,
        offset: int,
        samples: int,
    ) -> None:
        if board_size < 1 or offset < 0 or samples < 1:
            raise ValueError("board size and calibration slice must be positive")
        area = board_size * board_size
        packed_bytes = (area + 7) // 8
        with np.load(path) as archive:
            packed_source = archive["binaryInputNCHWPacked"]
            global_source = archive["globalInputNC"]
            end = offset + samples
            if end > packed_source.shape[0] or end > global_source.shape[0]:
                raise ValueError(
                    f"calibration slice [{offset},{end}) exceeds dataset rows"
                )
            self.packed = np.asarray(
                packed_source[offset:end], dtype=np.uint8
            )
            self.global_input = np.asarray(
                global_source[offset:end], dtype=np.float32
            )
        if self.packed.ndim != 3 or self.packed.shape[2] < packed_bytes:
            raise ValueError(
                f"packed spatial input cannot represent {board_size}x{board_size}"
            )
        if self.global_input.ndim != 2:
            raise ValueError("globalInputNC must be rank two")
        self.rows = samples
        self.board_size = board_size
        self.area = area
        spatial_bits = np.unpackbits(self.packed[:, 0, :], axis=1)
        if not np.all(spatial_bits[:, :area] == 1):
            raise ValueError("CPU-PTQ calibration requires an unmasked full board")

    def inputs(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        spatial = np.unpackbits(self.packed[start:end], axis=2)
        spatial = spatial[:, :, : self.area]
        spatial = np.ascontiguousarray(
            spatial.reshape(
                -1,
                self.packed.shape[1],
                self.board_size,
                self.board_size,
            ).astype(np.float32)
        )
        return spatial, np.ascontiguousarray(self.global_input[start:end])


def symmetric_codes(
    values: torch.Tensor, quantized_maximum: float, reduce_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = values.abs().amax(dim=reduce_dim, keepdim=True)
    scale = maximum / quantized_maximum
    scale = torch.where(
        torch.isfinite(scale) & (scale > 0.0),
        scale,
        torch.ones_like(scale),
    )
    codes = torch.clamp(
        torch.round(values / scale),
        -quantized_maximum,
        quantized_maximum,
    )
    return codes, scale


class ProjectionQuantController:
    """Fake-quantize the operations that feed GPTQ activation moments."""

    def __init__(self, model: Any, qmax: int) -> None:
        self.qmax = qmax
        self.original_sdpa = functional.scaled_dot_product_attention
        self.original_forwards: list[tuple[Any, Any]] = []
        self.linear_names: list[str] = []
        self._install_linears(model)
        functional.scaled_dot_product_attention = self.attention

    def close(self) -> None:
        functional.scaled_dot_product_attention = self.original_sdpa
        while self.original_forwards:
            module, original = self.original_forwards.pop()
            module.forward = original

    def _install_linears(self, model: Any) -> None:
        for block_index, block in enumerate(model.blocks):
            missing = [
                role for role in PROJECTION_ROLES if not hasattr(block, role)
            ]
            if missing:
                raise ValueError(
                    f"block {block_index} does not match the CPU-PTQ topology: "
                    f"{missing}"
                )
            for role in PROJECTION_ROLES:
                module = getattr(block, role)
                if not isinstance(module, torch.nn.Linear):
                    raise TypeError(f"blocks.{block_index}.{role} is not Linear")
                name = f"blocks.{block_index}.{role}"
                weight = module.weight.detach().float()
                maximum = weight.abs().amax(dim=1, keepdim=True)
                weight_scale = torch.where(
                    maximum > 0.0,
                    maximum / float(self.qmax),
                    torch.ones_like(maximum),
                )
                weight_codes = torch.clamp(
                    torch.round(weight / weight_scale),
                    -self.qmax,
                    self.qmax,
                )
                bias = module.bias
                self.original_forwards.append((module, module.forward))

                def fake_forward(
                    values: torch.Tensor,
                    *,
                    qweight: torch.Tensor = weight_codes,
                    wscale: torch.Tensor = weight_scale,
                    linear_bias: torch.Tensor | None = bias,
                ) -> torch.Tensor:
                    input_codes, input_scale = symmetric_codes(
                        values.float(), 127.0, values.ndim - 1
                    )
                    accumulators = functional.linear(
                        input_codes, qweight, None
                    )
                    output = (
                        accumulators
                        * input_scale
                        * wscale.transpose(0, 1)
                    )
                    if linear_bias is not None:
                        output = output + linear_bias
                    return output

                module.forward = fake_forward
                self.linear_names.append(name)
        expected = len(model.blocks) * len(PROJECTION_ROLES)
        if len(self.linear_names) != expected:
            raise AssertionError(
                f"installed {len(self.linear_names)} projections, expected {expected}"
            )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if dropout_p != 0.0 or is_causal or enable_gqa or attn_mask is not None:
            raise ValueError("CPU-PTQ calibration supports only unmasked plain MHA")
        multiplier = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
        query_maximum = 127.0 if self.qmax == 127 else 63.0
        value_maximum = 127.0 if self.qmax == 127 else 63.0
        query_codes, query_scale = symmetric_codes(query, query_maximum, -1)
        key_codes, key_scale = symmetric_codes(key, 127.0, -1)
        scores = torch.matmul(query_codes, key_codes.transpose(-2, -1))
        scores = (
            scores
            * query_scale
            * key_scale.transpose(-2, -1)
            * multiplier
        )

        shifted = scores - scores.amax(dim=-1, keepdim=True)
        exponentials = torch.exp(shifted)
        exponential_sum = exponentials.sum(dim=-1, keepdim=True)
        probability_codes = torch.clamp(
            torch.round(exponentials * 255.0), 0.0, 255.0
        )
        value_channel_maximum = value.abs().amax(dim=-2, keepdim=True)
        value_scale = torch.where(
            value_channel_maximum > 0.0,
            value_channel_maximum / value_maximum,
            torch.zeros_like(value_channel_maximum),
        )
        value_inverse = torch.where(
            value_channel_maximum > 0.0,
            value_maximum / value_channel_maximum,
            torch.zeros_like(value_channel_maximum),
        )
        value_codes = torch.clamp(
            torch.round(value * value_inverse),
            -value_maximum,
            value_maximum,
        )
        accumulators = torch.matmul(probability_codes, value_codes)
        # Keep the exact operation ordering used while selecting the accepted
        # recipes. Although division is algebraically equivalent, its last-bit
        # rounding can perturb later-block GPTQ moments and flip borderline
        # codes.
        return accumulators * value_scale * (
            1.0 / (255.0 * exponential_sum)
        )


class Moments:
    def __init__(self, channels: int, device: torch.device) -> None:
        self.xq_xq = torch.zeros(
            (channels, channels), dtype=torch.float64, device=device
        )
        self.xq_x = torch.zeros(
            (channels, channels), dtype=torch.float64, device=device
        )
        self.rows = 0

    def observe(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        codes, scale = symmetric_codes(values, 127.0, values.ndim - 1)
        original = values.reshape(-1, values.shape[-1])
        quantized = (codes * scale).reshape(-1, values.shape[-1])
        self.xq_xq.add_(torch.matmul(quantized.T, quantized).double())
        self.xq_x.add_(torch.matmul(quantized.T, original).double())
        self.rows += original.shape[0]


def collect_moments(
    model: Any,
    data: CalibrationData,
    device: torch.device,
    batch_size: int,
    qmax: int,
) -> list[dict[str, Moments]]:
    all_moments: list[dict[str, Moments]] = []
    handles = []
    for block in model.blocks:
        block_moments: dict[str, Moments] = {}
        all_moments.append(block_moments)
        for group, hook_role in GROUP_HOOK.items():
            channels = getattr(block, hook_role).in_features
            item = Moments(channels, device)
            block_moments[group] = item

            def pre_hook(_module: Any, args: tuple[Any, ...], *, target=item):
                target.observe(args[0])

            handles.append(
                getattr(block, hook_role).register_forward_pre_hook(pre_hook)
            )

    controller = ProjectionQuantController(model, qmax)
    try:
        with torch.inference_mode():
            for start in range(0, data.rows, batch_size):
                end = min(start + batch_size, data.rows)
                spatial_np, global_np = data.inputs(start, end)
                spatial = torch.from_numpy(spatial_np).to(device)
                global_input = torch.from_numpy(global_np).to(device)
                model(spatial, global_input, disable_mask=True)
                print(f"calibration rows={end}/{data.rows}", flush=True)
    finally:
        controller.close()
        for handle in handles:
            handle.remove()
    return all_moments


def gptq_matrix(
    weight: torch.Tensor,
    moments: Moments,
    qmax: int,
    damp: float,
    act_order: bool,
    compensation_blend: float,
    weight_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    original = weight.detach().float()
    hessian = (moments.xq_xq / float(moments.rows)).float()
    cross = (moments.xq_x / float(moments.rows)).float()
    diagonal_mean = torch.mean(torch.diagonal(hessian))
    regularizer = max(damp, 1.0e-5) * diagonal_mean
    identity = torch.eye(
        hessian.shape[0], dtype=hessian.dtype, device=hessian.device
    )

    target = original
    if compensation_blend != 0.0:
        transform = torch.linalg.solve(
            hessian + regularizer * identity,
            cross + regularizer * identity,
        )
        compensated = torch.matmul(original, transform.T)
        target = original + compensation_blend * (compensated - original)

    maximum = original.abs().amax(dim=1, keepdim=True)
    scale = torch.where(
        maximum > 0.0,
        maximum / float(qmax),
        torch.ones_like(maximum),
    )
    scale = scale * weight_scale_factor
    hessian = hessian.clone()
    hessian.diagonal().add_(damp * diagonal_mean)
    permutation = (
        torch.argsort(torch.diagonal(hessian), descending=True)
        if act_order
        else torch.arange(hessian.shape[0], device=hessian.device)
    )
    hessian = hessian[permutation][:, permutation]
    working = target[:, permutation].clone()

    inverse_factor = torch.linalg.cholesky(
        torch.linalg.inv(hessian), upper=True
    )
    codes_permuted = torch.zeros_like(working)
    block_size = 64
    for first in range(0, working.shape[1], block_size):
        last = min(first + block_size, working.shape[1])
        block = working[:, first:last].clone()
        errors = torch.zeros_like(block)
        for relative in range(last - first):
            column = block[:, relative]
            diagonal = inverse_factor[first + relative, first + relative]
            codes = torch.clamp(
                torch.round(column / scale[:, 0]), -qmax, qmax
            )
            quantized = codes * scale[:, 0]
            codes_permuted[:, first + relative] = codes
            error = (column - quantized) / diagonal
            block[:, relative:] -= (
                error.unsqueeze(1)
                * inverse_factor[
                    first + relative, first + relative:last
                ].unsqueeze(0)
            )
            errors[:, relative] = error
        if last < working.shape[1]:
            working[:, last:] -= torch.matmul(
                errors, inverse_factor[first:last, last:]
            )

    codes = torch.zeros_like(codes_permuted)
    codes[:, permutation] = codes_permuted
    return codes, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--qmax", type=int, choices=(63, 127), required=True)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--calib-offset", type=int, default=0)
    parser.add_argument("--calib-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--damp", type=float)
    parser.add_argument(
        "--act-order", choices=("auto", "yes", "no"), default="auto"
    )
    parser.add_argument("--compensation-blend", type=float, default=0.0)
    parser.add_argument("--weight-scale-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.damp is not None and args.damp <= 0.0:
        raise ValueError("--damp must be positive")
    if not 0.0 <= args.compensation_blend <= 1.0:
        raise ValueError("--compensation-blend must be in [0,1]")
    if not 0.5 <= args.weight_scale_factor <= 1.5:
        raise ValueError("--weight-scale-factor must be in [0.5,1.5]")

    training_root = args.training_repo.resolve()
    training_dir = training_root / "train"
    exporter = training_dir / "export_onnx.py"
    if not exporter.is_file():
        raise ValueError(
            f"--training-repo does not contain train/export_onnx.py: "
            f"{training_root}"
        )
    sys.path.insert(0, str(training_dir))
    from export_onnx import load_model_for_export

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20260827)

    checkpoint = args.checkpoint.resolve()
    data_path = args.data.resolve()
    data = CalibrationData(
        data_path,
        args.board_size,
        args.calib_offset,
        args.calib_samples,
    )
    regular, swa, _, _ = load_model_for_export(
        str(checkpoint),
        use_swa=True,
        device=device,
        pos_len=args.board_size,
    )
    del regular
    if swa is None:
        raise ValueError("checkpoint has no SWA model")
    model = swa.to(device).eval()

    damp = args.damp
    if damp is None:
        damp = 0.05 if args.qmax == 127 else 0.001
    act_order = (
        args.qmax == 127
        if args.act_order == "auto"
        else args.act_order == "yes"
    )
    recipe = f"gptq{'-order' if act_order else ''}-d{damp:g}"
    if args.weight_scale_factor != 1.0:
        recipe += f"-w{args.weight_scale_factor:g}"
    if args.compensation_blend != 0.0:
        recipe += f"-comp{args.compensation_blend:g}"

    moments = collect_moments(
        model, data, device, args.batch_size, args.qmax
    )
    overrides: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_index, block in enumerate(model.blocks):
        for group, roles in GROUP_ROLES.items():
            item = moments[block_index][group]
            for role in roles:
                name = f"blocks.{block_index}.{role}"
                print(f"quantizing {name}", flush=True)
                overrides[name] = gptq_matrix(
                    getattr(block, role).weight,
                    item,
                    args.qmax,
                    damp,
                    act_order,
                    args.compensation_blend,
                    args.weight_scale_factor,
                )

    names = sorted(overrides)
    payload: dict[str, np.ndarray] = {
        "format": np.asarray("cpuptq-gptq-v1"),
        "qmax": np.asarray(args.qmax, dtype=np.int32),
        "activation_quantizer": np.asarray("row-sym"),
        "activation_scale_factor": np.asarray(1.0, dtype=np.float32),
        "recipe": np.asarray(recipe),
        "names": np.asarray(names),
    }
    projection_report = []
    for index, name in enumerate(names):
        codes, scales = overrides[name]
        _, block_text, role = name.split(".")
        source_weight = np.asarray(
            getattr(model.blocks[int(block_text)], role)
            .weight.detach().cpu(),
            dtype="<f4",
            order="C",
        )
        codes_np = np.asarray(codes.detach().cpu(), dtype=np.int8, order="C")
        scales_np = np.asarray(
            scales.detach().cpu().reshape(-1), dtype="<f4", order="C"
        )
        source_hash = hashlib.sha256(
            source_weight.tobytes(order="C")
        ).hexdigest()
        payload[f"codes_{index}"] = codes_np
        payload[f"scales_{index}"] = scales_np
        payload[f"source_sha256_{index}"] = np.asarray(source_hash)
        projection_report.append(
            {
                "name": name,
                "shape": list(codes_np.shape),
                "codeMin": int(codes_np.min()),
                "codeMax": int(codes_np.max()),
                "scaleMin": float(scales_np.min()),
                "scaleMax": float(scales_np.max()),
                "sourceSha256": source_hash,
            }
        )

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **payload)
    report_path = (
        args.report.resolve()
        if args.report is not None
        else destination.with_suffix(destination.suffix + ".json")
    )
    report = {
        "kind": "cpuptq-gptq-v1-calibration",
        "manifest": str(destination),
        "manifestSha256": sha256_file(destination),
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256_file(checkpoint),
        "data": str(data_path),
        "dataSha256": sha256_file(data_path),
        "boardSize": args.board_size,
        "calibrationOffset": args.calib_offset,
        "calibrationSamples": args.calib_samples,
        "qmax": args.qmax,
        "recipe": recipe,
        "projectionCount": len(names),
        "projections": projection_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
