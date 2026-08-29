#!/usr/bin/env python3
"""Dump deterministic raw inputs and FP32/PTQ logits for v206 validation.

The binary files are headerless little-endian FP32. Each input row contains
22*7*7 NCHW spatial values followed by 19 global values. Each output row
contains policy channel 0 (49 board logits plus pass), three value logits,
the first four misc-value logits, and the first two more-misc logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


BOARD_SIZE = 7
INPUT_FLOATS = 22 * BOARD_SIZE * BOARD_SIZE + 19
OUTPUT_FLOATS = 50 + 3 + 4 + 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_outputs(outputs_by_heads: tuple) -> np.ndarray:
    heads = outputs_by_heads[0]
    policy = heads[0][:, 0, :]
    value = heads[1]
    misc = heads[2][:, :4]
    more = heads[3][:, :2]
    combined = torch.cat((policy, value, misc, more), dim=1)
    if combined.shape[1] != OUTPUT_FLOATS:
        raise AssertionError(f"unexpected raw output shape {combined.shape}")
    return np.ascontiguousarray(combined.detach().cpu().numpy(), dtype="<f4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dump-traces",
        action="store_true",
        help="also dump the PTQ trunk immediately after the stem and each block",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    train_dir = (args.training_repo.resolve() / "train")
    if not train_dir.is_dir():
        raise ValueError(f"training repo has no train directory: {train_dir}")
    sys.path.insert(0, str(train_dir))

    from calibrate_cpu_ptq_v106 import ProjectionQuantController
    from evaluate_cpu_ptq import data_loader, load_manifest
    from export_onnx import load_model_for_export

    torch.set_num_threads(1)
    torch.manual_seed(20260829)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    data_path = args.data.resolve()
    for path in (checkpoint, manifest_path, data_path):
        if not path.is_file():
            raise ValueError(f"missing input file {path}")

    regular, swa, _, _ = load_model_for_export(
        str(checkpoint), use_swa=True, device=device, pos_len=BOARD_SIZE
    )
    del regular
    if swa is None:
        raise ValueError("checkpoint has no SWA model")
    model = swa.to(device).eval()
    model.configure_flex_attention(False)
    manifest = load_manifest(manifest_path, model)

    iterator = data_loader(
        [data_path], model, BOARD_SIZE, args.samples, device
    )
    batch = next(iterator)
    spatial = batch["binaryInputNCHW"].detach().float()
    global_input = batch["globalInputNC"].detach().float()
    if spatial.shape != (args.samples, 22, BOARD_SIZE, BOARD_SIZE):
        raise AssertionError(f"unexpected spatial shape {spatial.shape}")
    if global_input.shape != (args.samples, 19):
        raise AssertionError(f"unexpected global shape {global_input.shape}")
    inputs = np.ascontiguousarray(
        torch.cat((spatial.flatten(1), global_input), dim=1).cpu().numpy(),
        dtype="<f4",
    )
    if inputs.shape != (args.samples, INPUT_FLOATS):
        raise AssertionError(f"unexpected combined input shape {inputs.shape}")

    with torch.inference_mode():
        fp32 = raw_outputs(
            model(spatial, global_input, input_meta=None, disable_mask=True)
        )
        trace_tensors = []
        trace_handles = []
        if args.dump_traces:
            if not model.blocks:
                raise AssertionError("model has no trunk blocks")

            def capture_stem(_module, inputs):
                trace_tensors.append(inputs[0].detach().clone())

            def capture_block(_module, _inputs, output):
                trace_tensors.append(output.detach().clone())

            trace_handles.append(
                model.blocks[0].register_forward_pre_hook(capture_stem)
            )
            for block in model.blocks:
                trace_handles.append(block.register_forward_hook(capture_block))
        controller = ProjectionQuantController(
            model, manifest.qmax, manifest.overrides
        )
        try:
            ptq = raw_outputs(
                model(spatial, global_input, input_meta=None, disable_mask=True)
            )
        finally:
            controller.close()
            for handle in trace_handles:
                handle.remove()

    traces = None
    if args.dump_traces:
        if len(trace_tensors) != len(model.blocks) + 1:
            raise AssertionError(
                f"expected {len(model.blocks) + 1} trunk traces, "
                f"got {len(trace_tensors)}"
            )
        traces = np.ascontiguousarray(
            torch.cat(
                [tensor.permute(0, 2, 3, 1).flatten(1) for tensor in trace_tensors],
                dim=1,
            ).cpu().numpy(),
            dtype="<f4",
        )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "inputs": Path(str(prefix) + ".inputs.f32"),
        "fp32": Path(str(prefix) + ".fp32.f32"),
        "ptq": Path(str(prefix) + ".ptq.f32"),
    }
    if traces is not None:
        paths["ptqTraces"] = Path(str(prefix) + ".ptq-traces.f32")
    inputs.tofile(paths["inputs"])
    fp32.tofile(paths["fp32"])
    ptq.tofile(paths["ptq"])
    if traces is not None:
        traces.tofile(paths["ptqTraces"])

    report = {
        "kind": "cpu-ptq-v206-raw-reference-v1",
        "samples": args.samples,
        "inputFloatsPerRow": INPUT_FLOATS,
        "outputFloatsPerRow": OUTPUT_FLOATS,
        "traceFloatsPerRow": (
            int(traces.shape[1]) if traces is not None else None
        ),
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256_file(checkpoint),
        "manifest": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "data": str(data_path),
        "dataSha256": sha256_file(data_path),
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "ptqMinusFp32": {
            "maxAbs": float(np.max(np.abs(ptq - fp32))),
            "meanAbs": float(np.mean(np.abs(ptq - fp32))),
        },
    }
    report_path = Path(str(prefix) + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
