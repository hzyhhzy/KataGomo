#!/usr/bin/env python3
"""INT8 Post-Training Quantization for KataGo nano Transformer models.

Adapted for the nano/ directory module structure (configs, model, data).
Produces TensorRT-compatible INT8 ONNX models in QDQ format.

Pipeline:
    1. Load FP16-trained checkpoint → FP32 PyTorch model
    2. Export to FP32 ONNX (with RMSNorm decomposition for ONNX compat)
    3. (Optional) Simplify ONNX graph
    4. Calibrate activation ranges using real game data (.npz)
    5. Apply static INT8 quantization (QDQ, symmetric, per-channel weights)
    6. Selectively exclude sensitive layers (Softmax, norms, heads)
    7. Verify quantized model accuracy (KL divergence on policy head)

Usage:
    # Basic
    python quantize_int8.py \
        --checkpoint ../data/train/b40c512h8/checkpoint.ckpt \
        --export-dir ../data/models_int8 \
        --calib-data ../data/shuffleddata/current

    # Full options
    python quantize_int8.py \
        --checkpoint ../data/train/b40c512h8/checkpoint.ckpt \
        --export-dir ../data/models_int8 \
        --calib-data ../data/shuffleddata/current \
        --pos-len 19 --batch-size 64 --calib-num 512 \
        --calib-method Entropy --use-ema --verify
"""

import sys
import os
import argparse
import inspect
import logging
import datetime
import glob
import random
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ─── ONNX Runtime quantization imports ───────────────────────────────────────
try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        QuantType,
        QuantFormat,
        CalibrationMethod,
        CalibrationDataReader,
        quantize_static,
    )
    try:
        from onnxruntime.quantization import quant_pre_process
    except ImportError:
        quant_pre_process = None
    HAS_ORT_QUANT = True
except ImportError:
    HAS_ORT_QUANT = False

try:
    import onnx
    from onnx import TensorProto
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# ─── Local imports (nano directory modules) ───────────────────────────────────
import modelconfigs
from modelconfigs import get_num_bin_input_features, get_num_global_input_features
try:
    from configs import migrate_config
except ImportError:
    def migrate_config(c):
        return c  # passthrough if not available
from model_pytorch import Model
import data_processing_pytorch

CALIB_METHOD_ALIASES = {
    "minmax": "MinMax",
    "entropy": "Entropy",
    "percentile": "Percentile",
    "distribution": "Distribution",
    "batchavgminmax": "BatchAvgMinMax",
    "batch_avg_minmax": "BatchAvgMinMax",
}


def parse_calib_method(value: str) -> str:
    key = value.strip().lower()
    
    if key not in CALIB_METHOD_ALIASES:
        valid = ", ".join(CALIB_METHOD_ALIASES.values())
        raise argparse.ArgumentTypeError(f"Invalid calib method '{value}'. Choose from: {valid}")
    return CALIB_METHOD_ALIASES[key]


# ─── Patch RMSNorm for ONNX export compatibility ─────────────────────────────
_original_rms_norm_forward = None


def _patch_rmsnorm():
    """Replace nn.RMSNorm.forward with decomposed ops for ONNX export."""
    global _original_rms_norm_forward
    if hasattr(nn, "RMSNorm") and _original_rms_norm_forward is None:
        _original_rms_norm_forward = nn.RMSNorm.forward

        def _manual_rms_norm_forward(self, x):
            x_f32 = x.float()
            mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
            eps_t = torch.tensor([self.eps], dtype=x_f32.dtype, device=x_f32.device)
            inv_rms = torch.rsqrt(mean_sq + eps_t)
            return self.weight * (x_f32 * inv_rms).type_as(x)

        nn.RMSNorm.forward = _manual_rms_norm_forward


def _restore_rmsnorm():
    """Restore original nn.RMSNorm.forward."""
    global _original_rms_norm_forward
    if _original_rms_norm_forward is not None:
        nn.RMSNorm.forward = _original_rms_norm_forward


# Also patch the custom RMSNormFP32 in model.py if needed
def _patch_model_rmsnorms():
    """Patch model.RMSNormFP32 to decompose for ONNX export."""
    try:
        from model import RMSNormFP32
    except ImportError:
        logging.info("RMSNormFP32 not found in model.py, skipping patch")
        return None
    original = RMSNormFP32.forward

    def _manual_forward(self, x):
        eps = getattr(self, 'eps', None)
        if eps is None:
            eps = getattr(getattr(self, 'norm', None), 'eps', 1e-6)
        weight = getattr(self, 'weight', None)
        if weight is None:
            weight = getattr(getattr(self, 'norm', None), 'weight', None)
        x_f32 = x.float()
        mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
        eps_t = torch.tensor([eps], dtype=x_f32.dtype, device=x_f32.device)
        inv_rms = torch.rsqrt(mean_sq + eps_t)
        return (weight * (x_f32 * inv_rms)).type_as(x)

    RMSNormFP32.forward = _manual_forward
    return original


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model Loading (nano checkpoint format)
# ═══════════════════════════════════════════════════════════════════════════════

def _looks_like_te_checkpoint(model_state):
    return any(".layer.self_attention." in key for key in model_state)


def _detect_checkpoint_format_standalone(state_dict):
    for key in state_dict:
        if ".layer.self_attention." in key:
            return "te"
    return "pt"


def _convert_checkpoint_te_to_model_standalone(state_dict, zero_centered_norm=False):
    """Convert TE (TransformerEngine) state_dict back to model.py format."""
    norm_suffix = ".weight" if zero_centered_norm else ".norm.weight"
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        if ".layer.layernorm_mlp.fc1_weight" in key:
            block_prefix = key.rsplit(".layer.layernorm_mlp.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".layer.layernorm_mlp.fc2_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.fc2_weight", ".ffn_w2.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.layer_norm_weight",
                               ".norm1" + norm_suffix)] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_bias" in key:
            continue
        elif ".layer.layernorm_mlp.layer_norm_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.layer_norm_weight",
                               ".norm2" + norm_suffix)] = value
        elif ".layer.layernorm_mlp.layer_norm_bias" in key:
            continue
        elif ".layer.self_attention.layernorm_qkv.query_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.query_weight",
                               ".q_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.key_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.key_weight",
                               ".k_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.value_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.value_weight",
                               ".v_proj.weight")] = value
        elif ".layer.self_attention.proj.weight" in key:
            new_sd[key.replace(".layer.self_attention.proj.weight",
                               ".out_proj.weight")] = value
        else:
            new_sd[key] = value
    return new_sd


def load_model(checkpoint_path: str, pos_len: int, use_ema: bool = False,
               score_mode: str = "mixop") -> Tuple[Model, dict, dict]:
    """Load a nano KataGo Transformer checkpoint.

    Returns (model, config, train_meta)
    """
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "config" in state, "Checkpoint missing 'config' key"
    config = migrate_config(state["config"])

    # Extract training flags
    varlen = state.get("varlen", False)
    attn_res = state.get("attn_res", False)
    gated_attn = state.get("gated_attn", False)
    head_bias = state.get("head_bias", False)
    zero_centered_norm = state.get("zero_centered_norm", False)

    logging.info(f"Config: layers={len(config['block_kind'])}, hidden={config['trunk_num_channels']}, "
                 f"heads={config['transformer_heads']}, ffn={config['transformer_ffn_channels']}")
    logging.info(f"Flags: varlen={varlen}, attn_res={attn_res}, gated_attn={gated_attn}, "
                 f"head_bias={head_bias}, zero_centered_norm={zero_centered_norm}")

    # Create model — only pass kwargs that this version of Model.__init__ accepts
    all_kwargs = dict(
        score_mode=score_mode,
        varlen=varlen,
        attn_res=attn_res,
        gated_attn=gated_attn,
        head_bias=head_bias,
        zero_centered_norm=zero_centered_norm,
    )
    model_params = set(inspect.signature(Model.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in model_params}
    skipped = set(all_kwargs) - set(filtered_kwargs)
    if skipped:
        logging.info(f"Model.__init__ does not accept: {skipped}, skipping these flags")
    model = Model(config, pos_len, **filtered_kwargs)

    # Load state dict (handle EMA and TE format)
    model_state = dict(state["model"])
    if use_ema:
        ema_shadow = state.get("swa_model_0")
        if ema_shadow is not None:
            for name, tensor in ema_shadow.items():
                if name in model_state:
                    model_state[name] = tensor
            logging.info(f"Using EMA weights ({len(ema_shadow)} parameters)")
        else:
            logging.warning("--use-ema specified but no swa_model_0 in checkpoint")
            assert(False)

    # Convert TE checkpoint if needed
    if _looks_like_te_checkpoint(model_state):
        try:
            from model_te import detect_checkpoint_format, convert_checkpoint_te_to_model
        except ImportError:
            detect_checkpoint_format = _detect_checkpoint_format_standalone
            convert_checkpoint_te_to_model = _convert_checkpoint_te_to_model_standalone
        if detect_checkpoint_format(model_state) == "te":
            logging.info("Converting TE checkpoint to model.py format")
            model_state = convert_checkpoint_te_to_model(model_state,
                                                          zero_centered_norm=zero_centered_norm)

    result = model.load_state_dict(model_state, strict=False)
    if result.missing_keys:
        logging.warning(f"Missing keys in checkpoint: {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        logging.debug(f"Unexpected keys in checkpoint: {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")
    if zero_centered_norm and hasattr(model, 'fuse_zero_centered_norm'):
        model.fuse_zero_centered_norm()
    model.eval()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total:,} ({total / 1e6:.1f}M)")

    return model, config, state


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONNX Export
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_NAMES = [
    "out_policy",        # (N, 6, L+1)
    "out_value",         # (N, 3)
    "out_miscvalue",     # (N, 10)
    "out_moremiscvalue", # (N, 8)
    "out_ownership",     # (N, 1, H, W)
    "out_scoring",       # (N, 1, H, W)
    "out_futurepos",     # (N, 2, H, W)
    "out_seki",          # (N, 4, H, W)
    "out_scorebelief",   # (N, scorebelief_len)
]

INPUT_NAMES = ["input_spatial", "input_global"]


def export_fp32_onnx(model: Model, save_path: str, config: dict, pos_len: int,
                     batch_size: int = 8, opset: int = 17) -> None:
    """Export nano model to FP32 ONNX.
    
    Uses the legacy ONNX exporter (dynamo=False) to ensure weights are
    embedded inside the .onnx file, not stored as external data.
    This is required for ORT quantize_static to work correctly.
    """
    model.eval()

    n_spatial = get_num_bin_input_features(config)
    n_global = get_num_global_input_features(config)

    inp_spatial = torch.randn(batch_size, n_spatial, pos_len, pos_len)
    inp_spatial[:, 0, :, :] = 1.0  # mask channel
    inp_global = torch.randn(batch_size, n_global)

    dynamic_axes = {n: {0: "batch"} for n in INPUT_NAMES + OUTPUT_NAMES}

    logging.info(f"Exporting FP32 ONNX → {save_path} (opset={opset}, legacy exporter)")
    logging.info(f"  Inputs: spatial={list(inp_spatial.shape)}, global={list(inp_global.shape)}")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inp_spatial, inp_global),
            save_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=False,  # Force legacy exporter — keeps weights inside .onnx
        )

    # Sanity check: weights should be embedded, file should be > 100MB for large models
    file_mb = os.path.getsize(save_path) / (1024 * 1024)
    param_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    logging.info(f"FP32 ONNX export done. File: {file_mb:.1f} MB (expected ~{param_mb:.0f} MB)")
    if file_mb < param_mb * 0.5:
        logging.error(f"ONNX file is suspiciously small ({file_mb:.1f} MB vs expected ~{param_mb:.0f} MB). "
                      f"Weights may be stored externally — quantization will fail!")
        # Try to fix by re-loading and re-saving with internal data
        if HAS_ONNX:
            logging.info("Attempting to internalize external data...")
            m = onnx.load(save_path, load_external_data=True)
            onnx.save(m, save_path, save_as_external_data=False)
            file_mb = os.path.getsize(save_path) / (1024 * 1024)
            logging.info(f"Re-saved with internal data: {file_mb:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Calibration Data Reader
# ═══════════════════════════════════════════════════════════════════════════════

class KataGoCalibrationReader(CalibrationDataReader):
    """Reads real KataGo game data (.npz) for INT8 calibration."""

    def __init__(self, data_dir: str, config: dict, pos_len: int,
                 batch_size: int, num_samples: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.pos_len = pos_len
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.consumed = 0
        self._iter = self._make_iter()

    def _make_iter(self):
        npz_files = glob.glob(os.path.join(self.data_dir, "**/*.npz"), recursive=True)
        if not npz_files:
            logging.error(f"No .npz files in {self.data_dir}")
            return

        random.shuffle(npz_files)
        logging.info(f"Calibration: found {len(npz_files)} npz files, "
                     f"using {self.num_samples} samples (batch={self.batch_size})")

        for batch in data_processing_pytorch.read_npz_training_data(
            npz_files,
            self.batch_size,
            world_size=1,
            rank=0,
            pos_len=self.pos_len,
            device=torch.device("cpu"),
            symmetry_type="none",
            include_meta=False,
            enable_history_matrices=False,
            model_config=self.config,
        ):
            if self.consumed >= self.num_samples:
                break

            feed = {
                "input_spatial": batch["binaryInputNCHW"].numpy(),
                "input_global":  batch["globalInputNC"].numpy(),
            }
            self.consumed += self.batch_size
            yield feed

    def get_next(self):
        return next(self._iter, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Intelligent Node Exclusion (critical for Transformer accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

# Op types sensitive to quantization noise — keep in FP32
SENSITIVE_OP_TYPES = {
    "Softmax",               # Attention softmax — top Elo impact
    "LayerNormalization",
    "ReduceMean",            # RMSNorm decomposition
    "Sqrt", "Rsqrt",         # RMSNorm decomposition
    "Div",                   # RMSNorm decomposition
}

# Module patterns to never quantize (input stem + output heads + norms)
EXCLUDE_MODULE_PATTERNS = [
    # Input stem
    "conv_spatial", "conv_dw", "linear_global",
    # Output heads
    "policy_head", "value_head",
    # Final trunk norm
    "norm_final",
    # Per-block norms (within Transformer blocks)
    "norm1", "norm2",
]


def find_activation_matmuls(onnx_path: str) -> List[str]:
    """Find MatMul nodes where BOTH inputs are activations (not weights).
    
    In a Transformer, SDPA decomposes into:
      - Q @ K^T  (activation × activation)  ← MUST stay FP32
      - attn_weights @ V  (activation × activation)  ← MUST stay FP32
    
    Weight MatMuls (Linear layers) have one input that is a graph initializer.
    Attention MatMuls have ZERO initializer inputs — both come from computation.
    
    This function returns the names of activation-only MatMul nodes to exclude.
    """
    if not HAS_ONNX:
        logging.warning("onnx package not installed, cannot detect attention MatMuls")
        return []

    model = onnx.load(onnx_path)
    
    # Collect all initializer names (these are the stored weights)
    initializer_names = set(init.name for init in model.graph.initializer)
    
    # Also treat graph inputs as "external" — they are not weights
    graph_input_names = set(inp.name for inp in model.graph.input)
    
    attn_matmuls = []
    weight_matmuls = []
    
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        
        # Check if either input is a weight (initializer)
        has_weight = any(inp in initializer_names for inp in node.input)
        
        if has_weight:
            weight_matmuls.append(node.name)
        else:
            attn_matmuls.append(node.name)
    
    logging.info(f"MatMul analysis: {len(weight_matmuls)} weight (Linear), "
                 f"{len(attn_matmuls)} activation-only (attention Q@K^T, attn@V)")
    
    if attn_matmuls:
        logging.info(f"  Excluding {len(attn_matmuls)} attention MatMuls from INT8")
    
    del model
    return attn_matmuls


def find_nodes_with_name_patterns(onnx_path: str, patterns: List[str], op_types: Optional[List[str]] = None) -> List[str]:
    if not HAS_ONNX:
        return []
    model = onnx.load(onnx_path)
    lowered = [p.lower() for p in patterns]
    matched = []
    for node in model.graph.node:
        if op_types is not None and node.op_type not in op_types:
            continue
        name = node.name or ""
        lname = name.lower()
        if any(p in lname for p in lowered):
            matched.append(name)
    del model
    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INT8 Static Quantization (QDQ format for TensorRT)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_onnx(input_path: str, output_path: str) -> str:
    """Run ONNX Runtime shape inference + optimization before quantization."""
    if quant_pre_process is None:
        logging.info("quant_pre_process unavailable, skipping preprocessing")
        return input_path

    logging.info("Preprocessing ONNX model...")
    try:
        quant_pre_process(input_path, output_path, skip_symbolic_shape=False)
        logging.info("Preprocessing done (with symbolic shape inference)")
        return output_path
    except Exception as e:
        logging.warning(f"Symbolic shape inference failed ({e}), retrying without...")
        try:
            quant_pre_process(input_path, output_path, skip_symbolic_shape=True)
            logging.info("Preprocessing done (without symbolic shapes)")
            return output_path
        except Exception as e2:
            logging.warning(f"Preprocessing failed ({e2}), using original model")
            return input_path


def list_int8_layers(onnx_path: str, op_types_to_quantize: List[str]) -> List[str]:
    if not HAS_ONNX:
        return []
    model = onnx.load(onnx_path)
    producer_by_output = {}
    for node in model.graph.node:
        for out in node.output:
            producer_by_output[out] = node
    int8_layers = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type not in op_types_to_quantize:
            continue
        is_quantized = any(
            producer_by_output.get(inp) is not None and producer_by_output[inp].op_type == "DequantizeLinear"
            for inp in node.input
        )
        if is_quantized:
            int8_layers.append(node.name if node.name else f"{node.op_type}_{idx}")
    del model
    return int8_layers


def quantize_to_int8(
    model_input_path: str,
    int8_onnx_path: str,
    calib_reader: CalibrationDataReader,
    calib_method: str = "Entropy",
    calib_percentile: Optional[float] = None,
    per_channel: bool = True,
    op_types_to_quantize: Optional[List[str]] = None,
    nodes_to_exclude: Optional[List[str]] = None,
) -> None:
    """Apply static INT8 quantization in QDQ format (TensorRT-compatible).
    
    Uses a WHITELIST approach: only quantize specific op types.
    Caller should preprocess the model and detect nodes_to_exclude beforehand.
    """
    assert HAS_ORT_QUANT, "onnxruntime.quantization is required"

    method_map = {
        "MinMax": getattr(CalibrationMethod, "MinMax", None),
        "Entropy": getattr(CalibrationMethod, "Entropy", None),
        "Percentile": getattr(CalibrationMethod, "Percentile", None),
        "Distribution": getattr(CalibrationMethod, "Distribution", None),
        "BatchAvgMinMax": getattr(CalibrationMethod, "MinMax", None),
    }
    calib = method_map.get(calib_method)
    if calib is None:
        available = [name for name, enum_value in method_map.items() if enum_value is not None]
        raise RuntimeError(
            f"Calibration method '{calib_method}' is not supported by current onnxruntime. "
            f"Available: {', '.join(available)}"
        )

    if op_types_to_quantize is None:
        op_types_to_quantize = ["MatMul", "Conv"]

    # Count ops
    if HAS_ONNX:
        m = onnx.load(model_input_path)
        total = len(m.graph.node)
        quant_count = sum(1 for n in m.graph.node if n.op_type in op_types_to_quantize)
        excluded_count = len(nodes_to_exclude) if nodes_to_exclude else 0
        del m
        logging.info(f"Quantizing: {model_input_path} → {int8_onnx_path}")
        logging.info(f"  Total nodes:         {total}")
        logging.info(f"  Whitelisted ops:     {quant_count} ({', '.join(op_types_to_quantize)})")
        logging.info(f"  Excluded by name:    {excluded_count} (attention MatMuls)")
        logging.info(f"  Actual INT8 ops:     {quant_count - excluded_count}")
        logging.info(f"  FP32 ops:            {total - quant_count + excluded_count}")

    logging.info(f"  Method:              {calib_method}")
    if calib_percentile is not None:
        logging.info(f"  Percentile:          {calib_percentile}")
    logging.info(f"  Per-channel:         {per_channel}")
    logging.info(f"  Format:              QDQ symmetric INT8")

    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": False,
    }
    if calib_method == "Percentile":
        if calib_percentile is not None:
            if not (0.0 <= calib_percentile <= 100.0):
                raise ValueError(f"--calib-percentile must be in [0, 100], got {calib_percentile}")
            extra_options["percentile"] = calib_percentile
    elif calib_method == "BatchAvgMinMax":
        extra_options["moving_average"] = True
        logging.info("  Batch-avg MinMax:    enabled")
    elif calib_percentile is not None:
        logging.warning("--calib-percentile is set but calib method is not Percentile, ignoring it")

    kwargs = dict(
        model_input=model_input_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calib_reader,
        calibrate_method=calib,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types_to_quantize,
        extra_options=extra_options,
    )
    if nodes_to_exclude:
        kwargs["nodes_to_exclude"] = nodes_to_exclude

    quantize_static(**kwargs)
    logging.info("INT8 quantization complete.")
    int8_layers = list_int8_layers(int8_onnx_path, op_types_to_quantize)
    logging.info(f"INT8 layers ({len(int8_layers)}):")
    for name in int8_layers:
        logging.info(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_int8_model(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    config: dict,
    pos_len: int,
    batch_size: int,
    calib_data_dir: Optional[str] = None,
) -> dict:
    """Compare FP32 vs INT8 ONNX outputs. Returns per-output metrics."""
    logging.info("Verifying INT8 model against FP32 baseline...")

    # Prepare test inputs
    feed = None
    if calib_data_dir:
        reader = KataGoCalibrationReader(
            calib_data_dir, config, pos_len, batch_size, num_samples=batch_size,
        )
        feed = reader.get_next()

    if feed is None:
        n_sp = get_num_bin_input_features(config)
        n_gl = get_num_global_input_features(config)
        sp = np.random.randn(batch_size, n_sp, pos_len, pos_len).astype(np.float32)
        sp[:, 0, :, :] = 1.0
        gl = np.random.randn(batch_size, n_gl).astype(np.float32)
        feed = {"input_spatial": sp, "input_global": gl}

    def run_ort(path):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess.run(None, feed)

    fp32_outs = run_ort(fp32_onnx_path)
    int8_outs = run_ort(int8_onnx_path)

    results = {}
    for i, (f32, i8) in enumerate(zip(fp32_outs, int8_outs)):
        name = OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"output_{i}"
        max_diff = float(np.max(np.abs(f32 - i8)))
        mean_diff = float(np.mean(np.abs(f32 - i8)))
        results[name] = {"max_diff": max_diff, "mean_diff": mean_diff}

        # Policy KL divergence
        if i == 0:
            p_logits = torch.from_numpy(f32[:, 0, :]).float()
            q_logits = torch.from_numpy(i8[:, 0, :]).float()
            p = torch.softmax(p_logits, dim=-1)
            q_log = torch.log_softmax(q_logits, dim=-1)
            kl = torch.nn.functional.kl_div(q_log, p, reduction="batchmean").item()
            results[name]["kl_divergence"] = kl
            logging.info(f"  {name:22s}  max={max_diff:.6f}  mean={mean_diff:.6f}  "
                         f"policy_KL={kl:.6f}")
        else:
            logging.info(f"  {name:22s}  max={max_diff:.6f}  mean={mean_diff:.6f}")

    kl = results.get("out_policy", {}).get("kl_divergence", 0)
    if kl < 0.001:
        logging.info("✓ Excellent quantization quality (KL < 0.001)")
    elif kl < 0.01:
        logging.info("✓ Good quantization quality (KL < 0.01), expect ~5 Elo loss")
    elif kl < 0.05:
        logging.warning("⚠ Moderate quality (KL < 0.05), may lose ~20-30 Elo")
    else:
        logging.warning("✗ Poor quality (KL >= 0.05), consider excluding more layers")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Metadata
# ═══════════════════════════════════════════════════════════════════════════════

def add_onnx_metadata(onnx_path: str, meta_dict: dict) -> None:
    if not HAS_ONNX:
        return
    m = onnx.load(onnx_path)
    if hasattr(m, "metadata_props"):
        del m.metadata_props[:]
    for k, v in meta_dict.items():
        entry = m.metadata_props.add()
        entry.key = str(k)
        entry.value = str(v)
    onnx.save(m, onnx_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="INT8 quantization for KataGo nano Transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Source
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("--onnx-input", default=None,
                        help="Skip export; quantize this existing FP32 ONNX instead")

    # Output
    parser.add_argument("--export-dir", required=True, help="Output directory")
    parser.add_argument("--model-name", default=None,
                        help="Base name for output files (default: from checkpoint dir name)")

    # Model
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA weights if available")
    parser.add_argument("--pos-len", type=int, default=19,
                        help="Board size (default: 19)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for calibration (default: 64)")
    parser.add_argument("--score-mode", default="mixop",
                        help="Score mode (default: mixop)")

    # Calibration
    parser.add_argument("--calib-data", required=True,
                        help="Directory with .npz game data for calibration")
    parser.add_argument("--calib-num", type=int, default=256,
                        help="Number of calibration samples (default: 256)")
    parser.add_argument("--calib-method", default="Entropy",
                        type=parse_calib_method,
                        choices=["MinMax", "Entropy", "Percentile", "Distribution", "BatchAvgMinMax"],
                        help="Calibration method (default: Entropy)")
    parser.add_argument("--calib-percentile", type=float, default=None,
                        help="Percentile value for Percentile calibration, range [0, 100]")

    # Quantization tuning
    parser.add_argument("--per-channel", action="store_true", default=True,
                        help="Per-channel weight quantization (default: True)")
    parser.add_argument("--per-tensor", action="store_true",
                        help="Per-tensor weight quantization (overrides --per-channel)")
    parser.add_argument("--extra-skip", default=None,
                        help="Extra ONNX op types to exclude, comma-separated")
    parser.add_argument("--no-skip-sensitive-ops", action="store_true",
                        help="Quantize ALL ops including Softmax/norms (aggressive)")

    # Other
    parser.add_argument("--simplify", action="store_true",
                        help="Run onnx-simplifier before quantization")
    parser.add_argument("--verify", action="store_true",
                        help="Verify FP32 vs INT8 accuracy after quantization")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17, use legacy exporter)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.export_dir, exist_ok=True)
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.export_dir, "quantize_int8.log")),
        ],
    )

    logging.info("=" * 70)
    logging.info("KataGo nano Transformer — INT8 Quantization")
    logging.info(f"Date: {datetime.datetime.now().isoformat()}")
    logging.info("=" * 70)

    per_channel = args.per_channel and not args.per_tensor
    skip_sensitive = not args.no_skip_sensitive_ops

    # ── Step 1: Load model ────────────────────────────────────────────────
    _patch_rmsnorm()
    _patch_model_rmsnorms()

    model, config, ckpt_state = load_model(
        args.checkpoint, args.pos_len, use_ema=args.use_ema,
        score_mode=args.score_mode,
    )

    # Build save name
    if args.model_name:
        save_name = args.model_name
    else:
        save_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))

    ts = ckpt_state.get("train_state", {})
    step = ts.get("global_step_samples") or ts.get("total_num_data_rows")
    if step:
        save_name += f"-s{step}"

    # ── Step 2: Export FP32 ONNX ──────────────────────────────────────────
    if args.onnx_input:
        fp32_path = args.onnx_input
        logging.info(f"Using existing FP32 ONNX: {fp32_path}")
        # Validate the file has actual weights embedded
        file_mb = os.path.getsize(fp32_path) / (1024 * 1024)
        param_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        if file_mb < param_mb * 0.5:
            logging.error(
                f"ONNX file is only {file_mb:.1f} MB but model has ~{param_mb:.0f} MB of FP32 weights. "
                f"Weights are likely stored externally (dynamo exporter). "
                f"Re-exporting with legacy exporter..."
            )
            fp32_path = os.path.join(args.export_dir, f"{save_name}_fp32.onnx")
            export_fp32_onnx(
                model, fp32_path, config, args.pos_len,
                batch_size=args.batch_size, opset=args.opset,
            )
    else:
        fp32_path = os.path.join(args.export_dir, f"{save_name}_fp32.onnx")
        export_fp32_onnx(
            model, fp32_path, config, args.pos_len,
            batch_size=args.batch_size, opset=args.opset,
        )

    # ── Step 2.5: Simplify (optional) ────────────────────────────────────
    if args.simplify:
        try:
            import onnxsim
            logging.info("Simplifying ONNX graph...")
            sim_path = fp32_path.replace(".onnx", "_sim.onnx")
            m, ok = onnxsim.simplify(fp32_path)
            if ok:
                onnx.save(m, sim_path)
                fp32_path = sim_path
                logging.info("Simplification done.")
            else:
                logging.warning("Simplification check failed, using original graph")
        except ImportError:
            logging.warning("onnxsim not installed, skipping. Install: pip install onnxsim")

    # ── Step 3: Preprocess ONNX for quantization ─────────────────────────
    preprocessed_path = fp32_path.replace(".onnx", "_preproc.onnx")
    quant_input_path = preprocess_onnx(fp32_path, preprocessed_path)

    # ── Step 4: Detect attention MatMuls to exclude ───────────────────────
    # CRITICAL: SDPA decomposes into Q@K^T and attn@V MatMuls.
    # These are activation×activation — quantizing them destroys accuracy.
    # Only weight MatMuls (Linear layers) should be INT8.
    op_types = ["MatMul", "Conv"]
    attn_matmuls = find_activation_matmuls(quant_input_path)
    head_nodes = find_nodes_with_name_patterns(
        quant_input_path,
        ["policy_head", "value_head", "linear_global", "conv_spatial"],
        op_types=op_types,
    )
    nodes_to_exclude = sorted(set(attn_matmuls + head_nodes))
    
    logging.info(f"Quantization strategy: whitelist {op_types}")
    logging.info(f"  Weight MatMuls (Linear):      will be INT8")
    logging.info(f"  Attention MatMuls (Q@K, A@V):  {len(attn_matmuls)} excluded, stay FP32")
    logging.info(f"  Head nodes (policy/value):     {len(head_nodes)} excluded, stay FP32")
    logging.info(f"  All other ops:                 stay FP32")

    # ── Step 5: Calibrate + Quantize ──────────────────────────────────────
    int8_path = os.path.join(args.export_dir, f"{save_name}_int8.onnx")
    reader = KataGoCalibrationReader(
        args.calib_data, config, args.pos_len, args.batch_size,
        num_samples=args.calib_num,
    )

    quantize_to_int8(
        quant_input_path, int8_path, reader,
        calib_method=args.calib_method,
        calib_percentile=args.calib_percentile,
        per_channel=per_channel,
        op_types_to_quantize=op_types,
        nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
    )

    # Cleanup preprocessed temp
    if quant_input_path != fp32_path and os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    # ── Step 6: Metadata (KataGo engine requires specific fields) ────────
    n_spatial = get_num_bin_input_features(config)
    n_global = get_num_global_input_features(config)
    meta = {
        # KataGo engine REQUIRED fields
        "name": save_name,
        "modelVersion": str(config.get("version", 15)),
        "num_spatial_inputs": str(n_spatial),
        "num_global_inputs": str(n_global),
        "pos_len": str(args.pos_len),
        "pos_len_x": str(args.pos_len),
        "pos_len_y": str(args.pos_len),
        "has_mask": "false",
        "auto_fp16_already": "false",
        "exported_with_dynamo": "false",
        # Quantization info
        "is_int8": "true",
        "quantization_method": f"ORT-{args.calib_method}",
        "quantized_ops": ",".join(op_types),
        "attention_matmuls_excluded": str(len(attn_matmuls)),
        "head_nodes_excluded": str(len(head_nodes)),
        "per_channel": str(per_channel),
        "calib_samples": str(args.calib_num),
        "exported_at": datetime.datetime.now().isoformat(),
        "model_config": str(config),
        "author": "unknown",
        "comment": "",
    }
    add_onnx_metadata(int8_path, meta)

    # ── Step 7: Verify ────────────────────────────────────────────────────
    if args.verify:
        _restore_rmsnorm()
        verify_int8_model(
            fp32_path, int8_path, config, args.pos_len,
            args.batch_size, calib_data_dir=args.calib_data,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    logging.info("=" * 70)
    logging.info("Quantization pipeline complete!")
    logging.info(f"  FP32 ONNX:  {fp32_path}")
    logging.info(f"  INT8 ONNX:  {int8_path}")
    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
    logging.info(f"  Size: {fp32_mb:.1f} MB → {int8_mb:.1f} MB ({int8_mb / fp32_mb * 100:.0f}%)")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
