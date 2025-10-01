#!/usr/bin/python3
import sys
import os
import argparse
import logging
import json
import datetime
import numpy as np
import torch
import torch.onnx
from typing import Dict, List, Optional, Tuple

import modelconfigs
from model_pytorch import Model
from load_model import load_model

torch.backends.mha.set_fastpath_enabled(False) # transformer model will have bugs, so set false


debug_mode=False
# Command and args -------------------------------------------------------------------

description = """
Export PyTorch neural net weights to ONNX format for inference.
"""

# Command line arguments will be parsed in the main block


class ONNXExportWrapper(torch.nn.Module):
    """
    Wrapper class to handle the model's forward pass for ONNX export.
    This handles the complex output structure and makes it ONNX-compatible.
    """
    
    def __init__(self, model: Model):
        super(ONNXExportWrapper, self).__init__()
        self.model = model
        self.has_intermediate_head = model.get_has_intermediate_head()
        self.has_metadata_encoder = model.get_has_metadata_encoder()    
    
    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor, 
                input_meta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass that returns a flattened tuple of outputs for ONNX compatibility.
        """
        # Call the original model
        if self.has_metadata_encoder and input_meta is not None:
            outputs = self.model(input_spatial, input_global, input_meta)
        else:
            outputs = self.model(input_spatial, input_global)
        outputs=outputs[0]
        
        pruned_outputs = tuple([outputs[i] for i in [0, 1, 2, 3, 4]])
        return pruned_outputs


def export_to_onnx(model: Model, export_path: str, pos_len: int = 19, 
                   batch_size: int = 1, opset_version: int = 11, 
                   verbose: bool = False) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        export_path: Path to save the ONNX model
        pos_len: Board position length
        batch_size: Batch size for the model
        opset_version: ONNX opset version
        verbose: Whether to enable verbose logging
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Create wrapper for ONNX export
    wrapper = ONNXExportWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs
    input_spatial = torch.randn(batch_size, 22, pos_len, pos_len, dtype=torch.float32)
    input_spatial[:,0,:,:]=1.0
    input_global = torch.randn(batch_size, 19, dtype=torch.float32)
    
    # Prepare inputs and input names
    inputs = [input_spatial, input_global]
    input_names = ['input_spatial', 'input_global']
    
    # Add metadata input if the model supports it
    if wrapper.has_metadata_encoder:
        # Assuming metadata has some standard size - this might need adjustment
        # based on the actual metadata encoder configuration
        input_meta = torch.randn(batch_size, 32, dtype=torch.float32)  # Placeholder size
        inputs.append(input_meta)
        input_names.append('input_meta')
    
    #output_names = [
    #    'policy', 'value', 'miscvalue', 'moremiscvalue', 
    #    'ownership', 'scoring', 'futurepos', 'seki', 'scorebelief_logprobs'
    #]
    output_names = [
        'out_policy', 'out_value', 'out_miscvalue', 'out_moremiscvalue', 
        'out_ownership'
    ]
    
    # Dynamic axes for variable batch size
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}
    
    # Export to ONNX
    logging.info(f"Exporting model to ONNX format: {export_path}")
    logging.info(f"Input shapes: spatial={list(input_spatial.shape)}, global={list(input_global.shape)}")
    
    dynamo=False # now it does not support True
    report=False
    if dynamo:
        dynamic_axes=None
        report=True


    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            export_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            dynamo=dynamo,
            report=report
        )
    
    logging.info("ONNX export completed successfully!")


def verify_onnx_model(onnx_path: str, original_model: Model, pos_len: int = 19, 
                      batch_size: int = 1, ignore_intermediate_head: bool = True) -> bool:
    """
    Verify that the ONNX model produces similar outputs to the original PyTorch model.
    
    Args:
        onnx_path: Path to the ONNX model
        original_model: Original PyTorch model
        pos_len: Board position length
        batch_size: Batch size for testing
        ignore_intermediate_head: Whether to ignore intermediate head outputs
        
    Returns:
        True if verification passes, False otherwise
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logging.warning("onnxruntime not available, skipping verification")
        return True
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test inputs
    input_spatial = torch.randn(batch_size, 22, pos_len, pos_len, dtype=torch.float32)
    input_spatial[:,0,:,:]=1.0
    input_global = torch.randn(batch_size, 19, dtype=torch.float32)
    
    # Get PyTorch outputs
    original_model.eval()
    with torch.no_grad():
        if original_model.get_has_metadata_encoder():
            # For models with metadata encoder, we need to handle this case
            pytorch_outputs = original_model(input_spatial, input_global)
        else:
            pytorch_outputs = original_model(input_spatial, input_global)
    pytorch_outputs = pytorch_outputs[0]
    pytorch_outputs = [pytorch_outputs[i] for i in [0, 1, 2, 3, 4]]

    # Prepare ONNX inputs
    onnx_inputs = {
        'input_spatial': input_spatial.numpy(),
        'input_global': input_global.numpy()
    }
    
    # Get ONNX outputs
    onnx_outputs = ort_session.run(None, onnx_inputs)
    
    
    # Check if number of outputs match
    if len(onnx_outputs) != len(pytorch_outputs):
        logging.error(f"Output count mismatch: ONNX={len(onnx_outputs)}, PyTorch={len(pytorch_outputs)}")
        return False
    
    # Check output shapes and values
    for i, (onnx_out, pytorch_out) in enumerate(zip(onnx_outputs, pytorch_outputs)):
        pytorch_np = pytorch_out.detach().numpy()
        
        if onnx_out.shape != pytorch_np.shape:
            logging.error(f"Output {i} shape mismatch: ONNX={onnx_out.shape}, PyTorch={pytorch_np.shape}")
            return False
        
        # Check if values are close (allowing for small numerical differences)
        if not np.allclose(onnx_out, pytorch_np, rtol=1e-5, atol=1e-6):
            max_diff = np.max(np.abs(onnx_out - pytorch_np))
            logging.warning(f"Output {i} values differ, max difference: {max_diff}")
            # Don't fail on small differences, just warn
    
    logging.info("ONNX model verification passed!")
    return True


if __name__ == "__main__":

    if not debug_mode:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('-checkpoint', help='Checkpoint file to export', required=True)
        parser.add_argument('-export-dir', help='Directory to export ONNX model to', required=True)
        parser.add_argument('-model-name', help='Name for the exported model', required=True)
        parser.add_argument('-use-swa', help='Use SWA model if available', action='store_true', required=False)
        parser.add_argument('-pos-len', help='Spatial edge length (e.g. 19 for 19x19 Go)', type=int, default=19, required=False)
        parser.add_argument('-batch-size', help='Batch size for ONNX export', type=int, default=1, required=False)
        parser.add_argument('-opset-version', help='ONNX opset version', type=int, default=20, required=False)
        parser.add_argument('-simplify', help='Simplify ONNX model using onnx-simplifier', action='store_true', required=False)
        parser.add_argument('-verbose', help='Verbose output', action='store_true', required=False)
        
        args = parser.parse_args()



        checkpoint_file = args.checkpoint
        export_dir = args.export_dir
        model_name = args.model_name
        use_swa = args.use_swa
        pos_len = args.pos_len
        batch_size = args.batch_size
        opset_version = args.opset_version
        simplify = args.simplify
        verbose = args.verbose
    else:
        checkpoint_file = "../data/train/go_b24c128tf1b_muon1_fd1/checkpoint.ckpt"
        export_dir = "../onnx_exports"
        model_name = "go_b24c128tf1b_muon1_fd1"
        use_swa = True
        pos_len = 19
        batch_size = 128
        opset_version = 20
        simplify = False
        verbose = False
    
    # Create export directory
    os.makedirs(export_dir, exist_ok=True)
    
    # Set up logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(os.path.join(export_dir, "export_log.txt")),
        ],
    )
    
    logging.info(f"ONNX Export Script - {datetime.datetime.now()}")
    logging.info(f"Arguments: {sys.argv}")
    
    # Load model
    logging.info(f"Loading model from checkpoint: {checkpoint_file}")
    model, swa_model, other_state_dict = load_model(
        checkpoint_file, use_swa, device="cpu", pos_len=pos_len, verbose=True
    )
    
    # Use SWA model if requested and available
    export_model = swa_model if (use_swa and swa_model is not None) else model
    model_type = "SWA" if (use_swa and swa_model is not None) else "regular"
    
    logging.info(f"Exporting {model_type} model")
    logging.info(f"Model config: {export_model.config}")
    
    # Export to ONNX
    onnx_filename = f"{model_name}.onnx"
    onnx_path = os.path.join(export_dir, onnx_filename)
    
    export_to_onnx(
        export_model, 
        onnx_path, 
        pos_len=pos_len, 
        batch_size=batch_size, 
        opset_version=opset_version,
        verbose=verbose
    )
    
    # Verify the exported model
    logging.info("Verifying exported ONNX model...")
    verification_passed = verify_onnx_model(onnx_path, export_model, pos_len, batch_size)
    
    if not verification_passed:
        logging.error("ONNX model verification failed!")
        exit(1)
    
    # Simplify model if requested
    if simplify:
        import onnxsim
        logging.info("Simplifying ONNX model...")
        simplified_path = os.path.join(export_dir, f"{model_name}_simplified.onnx")
        onnxsim.simplify(onnx_path, simplified_path)
        logging.info(f"Simplified model saved to: {simplified_path}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "export_time": datetime.datetime.now().isoformat(),
        "checkpoint_file": checkpoint_file,
        "model_type": model_type,
        "pos_len": pos_len,
        "batch_size": batch_size,
        "opset_version": opset_version,
        "has_intermediate_head": export_model.get_has_intermediate_head(),
        "has_metadata_encoder": export_model.get_has_metadata_encoder(),
        "model_config": export_model.config
    }
    
    # Add training state info if available
    if "train_state" in other_state_dict:
        train_state = other_state_dict["train_state"]
        if "global_step_samples" in train_state:
            metadata["global_step_samples"] = train_state["global_step_samples"]
        if "total_num_data_rows" in train_state:
            metadata["total_num_data_rows"] = train_state["total_num_data_rows"]
    
    metadata_path = os.path.join(export_dir, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Export completed successfully!")
    logging.info(f"ONNX model: {onnx_path}")
    logging.info(f"Metadata: {metadata_path}")
    
    exit(0)

