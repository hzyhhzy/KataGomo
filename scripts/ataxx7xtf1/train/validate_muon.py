#!/usr/bin/python3
import sys
import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel

import modelconfigs
from model_pytorch import Model
from metrics_pytorch import Metrics
import load_model
import data_processing_pytorch
from metrics_logging import accumulate_metrics, log_metrics
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)

torch.set_float32_matmul_precision('high')

def reset_nan_batchnorm(model, verbose=True):
    """
    Reset NaN/Inf in BatchNorm layers
    """
    has_nan = False
    
    for module in model.modules():
        for name, param in module.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                if verbose:
                    logging.info(f"Reset {name} in {module.__class__.__name__} (include NaN/Inf)")
                if "running_mean" in name:
                    nn.init.zeros_(param)
                elif "running_var" in name:
                    nn.init.ones_(param)
                elif "running_std" in name:
                    nn.init.ones_(param)
                else:
                    logging.info("Unrecoverable NAN")
                    assert(False)
                has_nan = True
        
        for name, buf in module.named_buffers():
            if torch.isnan(buf).any() or torch.isinf(buf).any():
                if verbose:
                    logging.info(f"Reset {name} in {module.__class__.__name__} (include NaN/Inf)")
                if "running_mean" in name:
                    nn.init.zeros_(buf)
                elif "running_var" in name:
                    nn.init.ones_(buf)
                elif "running_std" in name:
                    nn.init.ones_(buf)
                else:
                    logging.info("Unrecoverable NAN")
                    assert(False)
                has_nan = True
    
    if verbose and not has_nan:
        logging.info("No NaN/Inf in BatchNorm layers")
    
    return has_nan

def detensorify_metrics(metrics):
    ret = {}
    for key in metrics:
        if isinstance(metrics[key], torch.Tensor):
            ret[key] = metrics[key].detach().cpu().item()
        else:
            ret[key] = metrics[key]
    return ret

def main():
    description = """
    Validate neural net on Go positions from npz files.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-checkpoint', help='Path to checkpoint file', required=True)
    parser.add_argument('-data', help='Path to npz file or directory of npz files', required=True)
    parser.add_argument('-pos-len', help='Spatial edge length of expected training data, e.g. 19 for 19x19 Go', type=int, required=True)
    parser.add_argument('-batch-size', help='Batch size to use for validation', type=int, default=64)
    parser.add_argument('-model-kind', help='String name for what model config to use (if not in checkpoint)', required=False)
    parser.add_argument('-use-swa', help='Use SWA model if available', action='store_true')
    parser.add_argument('-swa-index', help='Index of SWA model to use (default 0)', type=int, default=0)
    parser.add_argument('-enable-history-matrices', help='Enable history matrices transformation', action='store_true')
    parser.add_argument('-symmetry-type', help='Data symmetry type', type=str, default="none")
    parser.add_argument('-max-batches', help='Max batches to validate', type=int, default=None)
    parser.add_argument('-no-compile', help='Do not torch.compile', action='store_true')
    
    # Loss scales
    parser.add_argument('-soft-policy-weight-scale', type=float, default=8.0, help='Soft policy loss coeff')
    parser.add_argument('-value-loss-scale', type=float, default=0.6, help='Additional value loss coeff')
    parser.add_argument('-td-value-loss-scales', type=str, default="0.6,0.6,0.6", help='Additional td value loss coeffs')
    parser.add_argument('-seki-loss-scale', type=float, default=1.0, help='Additional seki loss coeff')
    parser.add_argument('-variance-time-loss-scale', type=float, default=1.0, help='Additional variance time loss coeff')
    parser.add_argument('-main-loss-scale', type=float, default=1.0, help='Loss factor scale for main head')
    parser.add_argument('-intermediate-loss-scale', type=float, default=1.0, help='Loss factor scale for intermediate head')
    
    parser.add_argument('-disable-optimistic-policy', help='Disable optimistic policy', action='store_true')
    parser.add_argument('-meta-kata-only-soft-policy', help='Mask soft policy on non-kata rows using sgfmeta', action='store_true')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    data_path = args.data
    pos_len = args.pos_len
    batch_size = args.batch_size
    model_kind = args.model_kind
    use_swa = args.use_swa
    swa_index = args.swa_index
    enable_history_matrices = args.enable_history_matrices
    symmetry_type = args.symmetry_type
    max_batches = args.max_batches
    no_compile = args.no_compile
    
    soft_policy_weight_scale = args.soft_policy_weight_scale
    value_loss_scale = args.value_loss_scale
    td_value_loss_scales = [float(x) for x in args.td_value_loss_scales.split(",")]
    seki_loss_scale = args.seki_loss_scale
    variance_time_loss_scale = args.variance_time_loss_scale
    main_loss_scale = args.main_loss_scale
    intermediate_loss_scale = args.intermediate_loss_scale
    
    disable_optimistic_policy = args.disable_optimistic_policy
    meta_kata_only_soft_policy = args.meta_kata_only_soft_policy

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.warning("Using CPU")

    # Load Checkpoint
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine Config
    if "config" in state_dict:
        model_config = state_dict["config"]
        logging.info("Loaded config from checkpoint")
    elif model_kind is not None:
        model_config = modelconfigs.config_of_name[model_kind]
        logging.info(f"Using config for {model_kind}")
    else:
        raise ValueError("Model config not found in checkpoint and -model-kind not specified")
    
    # Initialize Model
    raw_model = Model(model_config, pos_len)
    raw_model.initialize()

    # Load State Dict
    if use_swa:
        logging.info(f"Loading SWA model (index {swa_index})...")
        swa_model_state_dict = load_model.load_swa_model_state_dict(state_dict, idx=swa_index)
        if swa_model_state_dict is None:
            raise ValueError(f"SWA model {swa_index} not found in checkpoint")
        
        swa_model_wrapper = AveragedModel(raw_model)
        swa_model_wrapper.load_state_dict(swa_model_state_dict)
        raw_model = swa_model_wrapper.module
        
    else:
        logging.info("Loading normal model...")
        model_state_dict = load_model.load_model_state_dict(state_dict)
        raw_model.load_state_dict(model_state_dict)

    reset_nan_batchnorm(raw_model)
    raw_model.to(device)
    raw_model.eval()
    
    if not no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(raw_model, mode="default")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}. Using raw model.")
            model = raw_model
    else:
        model = raw_model

    # Metrics
    metrics_obj = Metrics(batch_size, 1, raw_model) # world_size=1

    # Data Loading
    npz_files = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            if f.endswith(".npz"):
                npz_files.append(os.path.join(data_path, f))
    else:
        npz_files.append(data_path)
    
    npz_files.sort()
    logging.info(f"Found {len(npz_files)} npz files.")

    # Generator
    data_loader = data_processing_pytorch.read_npz_training_data(
        npz_files=npz_files,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        pos_len=pos_len,
        device=device,
        symmetry_type=symmetry_type,
        include_meta=True,
        enable_history_matrices=enable_history_matrices,
        model_config=model_config
    )
    
    val_metric_sums = defaultdict(float)
    val_metric_weights = defaultdict(float)

    batch_count = 0
    start_time = time.time()

    logging.info("Starting validation...")
    
    if data_loader is None:
        logging.error("No data found or data loader initialization failed.")
        sys.exit(1)

    with torch.no_grad():
        for batch in data_loader:
            if max_batches is not None and batch_count >= max_batches:
                break
                
            model_outputs = model(
                batch["binaryInputNCHW"],
                batch["globalInputNC"],
                batch.get("metadataInputNC") # Pass if exists
            )
            
            postprocessed = raw_model.postprocess_output(model_outputs)
            
            metrics = metrics_obj.metrics_dict_batchwise(
                raw_model,
                postprocessed,
                None, # extra_outputs
                batch,
                is_training=False,
                soft_policy_weight_scale=soft_policy_weight_scale,
                disable_optimistic_policy=disable_optimistic_policy,
                meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                value_loss_scale=value_loss_scale,
                td_value_loss_scales=td_value_loss_scales,
                seki_loss_scale=seki_loss_scale,
                variance_time_loss_scale=variance_time_loss_scale,
                main_loss_scale=main_loss_scale,
                intermediate_loss_scale=intermediate_loss_scale
            )
            
            metrics = detensorify_metrics(metrics)
            accumulate_metrics(val_metric_sums, val_metric_weights, metrics, batch_size, decay=1.0, new_weight=1.0)
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Processed {batch_count} batches...", end='\r')

    print(f"\nFinished processing {batch_count} batches.")
    print(f"Time: {time.time() - start_time:.2f}s")
    
    log_metrics(val_metric_sums, val_metric_weights, {}, None, "Validation")

if __name__ == "__main__":
    main()
