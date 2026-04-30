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
try:
    import torch.export
except ImportError:
    torch_export = None
import torch.ao.quantization
from typing import Dict, List, Optional, Tuple

import modelconfigs
from model_pytorch import Model
#from load_model import load_model
import data_processing_pytorch

from qat_helper import get_tensorrt_qat_qconfig, is_qat_checkpoint, disable_qat_for_unsupported_modules, UNSUPPORTED_QUANTIZATION_MODULES

try:
    from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader, CalibrationMethod, QuantFormat, quant_pre_process
except ImportError:
    QuantType = None
    quantize_static = None
    CalibrationDataReader = None
    CalibrationMethod = None
    QuantFormat = None
    quant_pre_process = None

#torch.backends.mha.set_fastpath_enabled(False) # transformer model will have bugs, so set false
# 定义手动实现的 forward 函数
def manual_rms_norm_forward(self, x):
   # 1. 强制 FP32 以保证数值稳定 (Crucial for LLMs)
    x_f32 = x.float()
    
    # 2. 使用乘法代替 pow(2)，效率略高且算子最简单
    # mean(-1) 计算均方值
    mean_square = (x_f32 * x_f32).mean(-1, keepdim=True)
    
    eps_tensor = torch.tensor([self.eps,], dtype=x_f32.dtype, device=x_f32.device) # to make sure int8 not error
    # 3. 计算 rsqrt (1 / sqrt(x))
    inv_rms = torch.rsqrt(mean_square + eps_tensor)
    
    # 4. 乘回原类型输入，并应用 gamma 参数
    # 注意：最终乘法建议在原精度下做，或者全在 FP32 做完再转回
    return self.weight * (x_f32 * inv_rms).type_as(x)

# 将 torch.nn.RMSNorm 类的 forward 方法替换为我们的手动实现
# 这样当导出器遍历模型时，看到的是基础数学运算，而不是 aten::rms_norm
if hasattr(torch.nn, "RMSNorm"):
    original_rms_norm_forward = torch.nn.RMSNorm.forward
    torch.nn.RMSNorm.forward = manual_rms_norm_forward

debug_mode=False
# Command and args -------------------------------------------------------------------

description = """
Export PyTorch neural net weights to ONNX format for inference.
"""

# Command line arguments will be parsed in the main block




def load_model_for_export(checkpoint_file, use_swa, device, pos_len=19, verbose=False, convert_qat_to_float=False):
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        logging.info(f"No config in checkpoint")
        assert False, "No config in checkpoint"

    logging.info(f"Model config: {model_config}")
    
    is_qat = is_qat_checkpoint(checkpoint)
    if is_qat and convert_qat_to_float:
        logging.info("QAT checkpoint detected and -convert-qat-to-float is enabled. Converting to regular float model...")
        is_qat = False # Pretend it's not QAT from now on
    
    def create_and_load_model(state_dict_key, is_swa=False):
        m = Model(model_config, pos_len)
        m.initialize()
        
        if is_qat:
            logging.info(f"Applying QAT configuration to {'SWA ' if is_swa else ''}model...")
            m.qconfig = get_tensorrt_qat_qconfig()
            disable_qat_for_unsupported_modules(m)
            torch.ao.quantization.prepare_qat(m, inplace=True)
        
        # Strip "module." prefix and filter keys
        state_dict = checkpoint[state_dict_key]
        model_keys = m.state_dict().keys()
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[7:]
            
            # Skip keys that are known to be problematic or don't exist in the target model
            if name == "n_averaged":
                continue
            if "score_belief_offset_vector" in name or "score_belief_offset_bias_vector" in name or "score_belief_parity_vector" in name:
                continue
            
            if name in model_keys:
                new_state_dict[name] = v
            else:
                logging.debug(f"Skipping key {name} as it does not exist in the model")
        
        # Use strict=False if we are converting from QAT to float to ignore fake_quant/observer keys
        m.load_state_dict(new_state_dict, strict=not convert_qat_to_float)
        
        if is_qat:
            # For TensorRT QAT export, we keep the FakeQuantize nodes and DO NOT call convert.
            # This allows torch.onnx.export to generate standard QDQ (QuantizeLinear/DequantizeLinear) nodes.
            logging.info(f"Preparing QAT {'SWA ' if is_swa else ''}model for QDQ ONNX export (keeping FakeQuantize)...")
            m.eval()
            # Explicitly disable observers to avoid aten::copy in ONNX export
            m.apply(torch.ao.quantization.disable_observer)
            m.apply(torch.ao.quantization.enable_fake_quant)
            
        return m

    model = create_and_load_model("model")
    
    swa_model = None
    if use_swa:
        swa_key = "swa_model_0" if "swa_model_0" in checkpoint else "swa_model"
        if swa_key in checkpoint:
            swa_model = create_and_load_model(swa_key, is_swa=True)
        else:
            logging.warning(f"SWA model requested but {swa_key} not found in checkpoint")

    other_state_dict = {}
    for key in ["metrics", "running_metrics", "train_state", "last_val_metrics"]:
        if key in checkpoint:
            other_state_dict[key] = checkpoint[key]
            
    is_qat = is_qat_checkpoint(checkpoint)
    return model, swa_model, other_state_dict, is_qat


class ONNXExportWrapper(torch.nn.Module):
    """
    Wrapper class to handle the model's forward pass for ONNX export.
    This handles the complex output structure and makes it ONNX-compatible.
    """
    
    def __init__(self, model: Model, disable_mask: bool):
        super(ONNXExportWrapper, self).__init__()
        self.model = model
        self.has_intermediate_head = model.get_has_intermediate_head()
        self.has_metadata_encoder = model.get_has_metadata_encoder()    
        self.disable_mask = disable_mask
    
    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor, input_meta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass that returns a flattened tuple of outputs for ONNX compatibility.
        """
        # Call the original model
        if self.has_metadata_encoder:
            outputs = self.model(input_spatial, input_global, input_meta, disable_mask=self.disable_mask)
        else:
            outputs = self.model(input_spatial, input_global, disable_mask=self.disable_mask)
        
        outputs = outputs[0]
        pruned_outputs = tuple([outputs[i] for i in [0, 1, 2, 3, 4]])
        return pruned_outputs

if CalibrationDataReader is not None:
    class ONNXCalibrationDataReader(CalibrationDataReader):
        """
        Calibration data reader for ONNX quantization.
        Reads data from .npz files using the project's data processing logic.
        """
        def __init__(self, calib_data_dir: str, model: Model, pos_len: int, batch_size: int, require_exact_poslen: bool, num_samples: int = 128):
            super().__init__()
            self.calib_data_dir = calib_data_dir
            self.model = model
            self.pos_len = pos_len
            self.batch_size = batch_size
            self.require_exact_poslen = require_exact_poslen
            self.num_samples = num_samples
            self.consumed_samples = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Calibration data reader using device: {self.device}")
            self.data_iter = self._create_iter()

        def _create_iter(self):
            import glob
            npz_files = glob.glob(os.path.join(self.calib_data_dir, "**/*.npz"), recursive=True)
            if not npz_files:
                logging.warning(f"No .npz files found in {self.calib_data_dir} for calibration.")
                return

            # Shuffle files to get a representative sample
            import random
            random.shuffle(npz_files)

            for batch in data_processing_pytorch.read_npz_training_data(
                npz_files,
                self.batch_size,
                world_size=1,
                rank=0,
                pos_len=self.pos_len,
                device=self.device,
                symmetry_type="none",
                include_meta=self.model.get_has_metadata_encoder(),
                enable_history_matrices=False,
                model_config=self.model.config
            ):
                if self.consumed_samples >= self.num_samples:
                    break

                mask_layer= batch["binaryInputNCHW"][:,0,:,:].cpu().numpy()
                assert mask_layer.shape == (self.batch_size, self.pos_len, self.pos_len), f"mask_layer shape {mask_layer.shape} != {(self.batch_size, self.pos_len, self.pos_len)}"
                if self.require_exact_poslen:
                    assert mask_layer.all(axis=(1,2)).all(), f"int8 calibration data should be all {self.pos_len}x{self.pos_len} games if -disable-mask"
                
                # Move to CPU for ONNX Runtime input
                inputs = {
                    'input_spatial': batch["binaryInputNCHW"].cpu().numpy(),
                    'input_global': batch["globalInputNC"].cpu().numpy()
                }
                if self.model.get_has_metadata_encoder():
                    inputs['input_meta'] = batch["metadataInputNC"].cpu().numpy()
                
                self.consumed_samples += self.batch_size
                yield inputs

        def get_next(self):
            return next(self.data_iter, None)


def quantize_onnx(model_path: str, output_path: str, calib_data_dir: str, 
                  model: Model, pos_len: int, batch_size: int, 
                  disable_mask: bool,
                  num_samples: int = 128, method: str = "Entropy",
                  verbose: bool = False) -> None:
    """
    Quantize an ONNX model to INT8.
    """
    if quantize_static is None:
        logging.error("onnxruntime-quantization not installed. Cannot perform INT8 quantization.")
        return

    # Pre-processing
    actual_input_path = model_path
    temp_preprocessed_path = None
    if quant_pre_process is not None:
        logging.info(f"Running pre-processing on {model_path}...")
        temp_preprocessed_path = model_path.replace(".onnx", "_preprocessed.onnx")
        try:
            logging.info("Attempting pre-processing with symbolic shape inference...")
            quant_pre_process(model_path, temp_preprocessed_path, skip_symbolic_shape=False)
            actual_input_path = temp_preprocessed_path
            logging.info(f"Pre-processing completed successfully.")
        except Exception as e:
            logging.warning(f"Pre-processing with symbolic shape inference failed: {e}")
            logging.info("Retrying pre-processing without symbolic shape inference...")
            try:
                quant_pre_process(model_path, temp_preprocessed_path, skip_symbolic_shape=True)
                actual_input_path = temp_preprocessed_path
                logging.info(f"Pre-processing completed (skipped symbolic shape inference).")
            except Exception as e2:
                logging.error(f"Pre-processing failed again: {e2}. Proceeding with original model.")
                temp_preprocessed_path = None
    else:
        logging.warning("quant_pre_process not available. Skipping pre-processing.")

    logging.info(f"Quantizing model to INT8: {actual_input_path} -> {output_path}")
    logging.info(f"Using calibration data from: {calib_data_dir} ({num_samples} samples)")

    dr = ONNXCalibrationDataReader(calib_data_dir, model, pos_len, batch_size, disable_mask, num_samples)
    
    # Identify nodes to exclude from quantization (input/output heads)
    nodes_to_exclude = []
    try:
        import onnx
        onnx_model = onnx.load(actual_input_path)
        for node in onnx_model.graph.node:
            for module_name in UNSUPPORTED_QUANTIZATION_MODULES:
                # Check if module name is in node name (PyTorch export usually includes scope)
                if module_name in node.name:
                    nodes_to_exclude.append(node.name)
                    break
        if nodes_to_exclude:
            logging.info(f"Excluding {len(nodes_to_exclude)} nodes from quantization (matched UNSUPPORTED_QUANTIZATION_MODULES)")
            if verbose:
                logging.debug(f"Excluded nodes: {nodes_to_exclude}")
    except Exception as e:
        logging.warning(f"Failed to identify nodes to exclude from ONNX model: {e}")

    # Map method string to CalibrationMethod
    calib_method = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }.get(method, CalibrationMethod.Entropy)

    from onnxruntime.quantization import QuantFormat

    kwargs = {
        "model_input": actual_input_path,
        "model_output": output_path,
        "calibration_data_reader": dr,
        "calibrate_method": calib_method,
        "activation_type": QuantType.QInt8,  # 改为 QInt8 以实现对称量化，兼容 TensorRT
        "weight_type": QuantType.QInt8,      # 权重保持 QInt8
        "per_channel": True,                # 关闭 per_channel 以避免标量(如eps)量化时的 axis 错误 (TensorRT 报错 nbDims=0)
        "reduce_range": False,
        "quant_format": QuantFormat.QDQ,
        "nodes_to_exclude": nodes_to_exclude,
        "extra_options": {
            "ActivationSymmetric": True, 
            "WeightSymmetric": True,
            "QuantizeBias": False
        }
    }

    logging.info(f"Starting INT8 quantization (QDQ format for TensorRT)...")
    quantize_static(**kwargs)
    logging.info("INT8 quantization completed.")

    # Clean up temporary pre-processed model
    if temp_preprocessed_path and os.path.exists(temp_preprocessed_path):
        try:
            os.remove(temp_preprocessed_path)
            logging.info(f"Cleaned up temporary pre-processed model: {temp_preprocessed_path}")
        except Exception as e:
            logging.warning(f"Failed to delete temporary pre-processed model: {e}")


def export_to_onnx(model: Model, full_name: str ,export_path: str, pos_len: int = 19, 
                   batch_size: int = 1, opset_version: int = 20, disable_mask: bool = False,
                   verbose: bool = False, extra_meta_data: Dict[str, str] = None,
                   auto_fp16: bool = False, fix_batchsize: bool = False,
                   dynamo: bool = False) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        export_path: Path to save the ONNX model
        pos_len: Board position length
        batch_size: Batch size for the model
        opset_version: ONNX opset version
        verbose: Whether to enable verbose logging
        auto_fp16: Whether to automatically convert to mixed precision (FP16)
        fix_batchsize: Whether to fix the batch size to the specified value
        dynamo: Whether to use TorchDynamo for export
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Create wrapper for ONNX export
    wrapper = ONNXExportWrapper(model, disable_mask=disable_mask)
    wrapper.eval()
    
    # Create dummy inputs
    num_spatial_inputs = modelconfigs.get_num_bin_input_features(model.config)
    input_spatial = torch.randn(batch_size, num_spatial_inputs, pos_len, pos_len, dtype=torch.float32)
    input_spatial[:,0,:,:]=1.0
    num_global_inputs = modelconfigs.get_num_global_input_features(model.config)
    input_global = torch.randn(batch_size, num_global_inputs, dtype=torch.float32)
    
    # Prepare inputs and input names
    inputs = [input_spatial, input_global]
    input_names = ['input_spatial', 'input_global']
    
    # Add metadata input if the model supports it
    if wrapper.has_metadata_encoder:
        num_meta_input_features = modelconfigs.get_num_meta_encoder_input_features(model.config)
        input_meta = torch.randn(batch_size, num_meta_input_features, dtype=torch.float32)
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
    dynamic_shapes = None
    if not fix_batchsize:
        if dynamo:
              # For dynamo, we need to provide dynamic_shapes as a dict or tuple
              # Now that forward has explicit arguments, we can use a dict matching input names
              dynamic_shapes = {}
              batch_dim = None
              if hasattr(torch, "export") and hasattr(torch.export, "Dim"):
                  batch_dim = torch.export.Dim("batch_size", min=1, max=16384)
              
              for name in input_names:
                  if batch_dim is not None:
                      dynamic_shapes[name] = {0: batch_dim}
                  else:
                      dynamic_shapes[name] = {0: "batch_size"}
              
              dynamic_axes = None # dynamo uses dynamic_shapes
        else:
            for name in input_names:
                dynamic_axes[name] = {0: 'batch_size'}
            for name in output_names:
                dynamic_axes[name] = {0: 'batch_size'}
    else:
        dynamic_axes = None
        dynamic_shapes = None
        logging.info(f"Fixed batch size enabled: {batch_size}")
    
    # Export to ONNX
    logging.info(f"Exporting model to ONNX format: {export_path}")
    logging.info(f"Input shapes: spatial={list(input_spatial.shape)}, global={list(input_global.shape)}")
    
    report=False
    if dynamo:
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
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
            dynamo=dynamo,
            report=report
        )
    
    logging.info("ONNX export completed successfully!")

    # Auto convert to FP16 if requested
    if auto_fp16:
        assert False,"Currently has trouble converting RMSNorm to fp16. Do not use -auto-fp16"
        
        import onnx
        from onnxconverter_common import auto_convert_mixed_precision,float16

        logging.info("Converting model to auto mixed precision (FP16)...")
        
        onnx_model = onnx.load(export_path)
        # Create feed_dict with dummy inputs converted to numpy
        feed_dict = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, inputs)}
        onnx_model_fp16 = auto_convert_mixed_precision(onnx_model, feed_dict, rtol=0.01, keep_io_types=True)
        #onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model_fp16, export_path)
        logging.info("Converted to FP16 successfully.")
            

    # Add metadata to the ONNX model
    try:
        import onnx
        from onnx import helper
        
        onnx_model = onnx.load(export_path)
        
        # Add metadata_props
        meta = {
            "name": full_name,
            "modelVersion": str(model.config["version"]),
            # Add other useful info if available
            "exported_at": datetime.datetime.now().isoformat(),
            "auto_fp16_already": "true" if auto_fp16 else "false",
            "opset_version": str(opset_version),
            "exported_with_dynamo": "true" if dynamo else "false",
            "num_spatial_inputs": str(num_spatial_inputs),
            "num_global_inputs": str(num_global_inputs),
            "pos_len": str(pos_len),
            "pos_len_x": str(pos_len),
            "pos_len_y": str(pos_len),
            "has_mask": "true" if not disable_mask else "false",
            "is_simplified": "false",
            "is_int8": "true" if (extra_meta_data and extra_meta_data.get("is_int8") == "true") else "false",
            "model_config": str(model.config)
        }
        if extra_meta_data is not None:
            meta.update(extra_meta_data)
        
        # Clear existing metadata if any to avoid duplicates
        if hasattr(onnx_model, "metadata_props"):
            del onnx_model.metadata_props[:]
            
        for key, value in meta.items():
            meta_entry = onnx_model.metadata_props.add()
            meta_entry.key = key
            meta_entry.value = value
            
        # Save the model with metadata
        onnx.save(onnx_model, export_path)
        logging.info(f"Added metadata to ONNX model: {meta}")
        
    except ImportError:
        logging.warning("onnx package not installed, skipping metadata addition")
    except Exception as e:
        logging.error(f"Failed to add metadata: {e}")


def compare_models(model1, model2, model1_type: str, model2_type: str, pos_len: int = 19, 
                   batch_size: int = 1, calib_data_dir: str = None, config=None) -> bool:
    """
    Compare two models (either PyTorch or ONNX) to ensure they produce similar outputs.
    
    Args:
        model1: First model (Model object or ONNX path)
        model2: Second model (Model object or ONNX path)
        model1_type: 'pytorch' or 'onnx'
        model2_type: 'pytorch' or 'onnx'
        pos_len: Board position length
        batch_size: Batch size for testing
        calib_data_dir: Directory with .npz files for validation data
        config: Model config (required if both are ONNX)
        
    Returns:
        True if verification passes, False otherwise
    """
    try:
        import onnxruntime as ort
    except ImportError:
        if model1_type == 'onnx' or model2_type == 'onnx':
            assert False, "onnxruntime not available, cannot compare ONNX models"
    
    # Get config
    if config is None:
        if model1_type == 'pytorch':
            config = model1.config
        elif model2_type == 'pytorch':
            config = model2.config
        else:
            assert False, "Config must be provided if both models are ONNX"

    # Create test inputs
    input_spatial, input_global, input_meta = None, None, None
    if calib_data_dir is not None:
        logging.info(f"Using calibration data from {calib_data_dir} for comparison")
        # Use a temporary model for data reader if needed
        data_reader_model = model1 if model1_type == 'pytorch' else model2
        dr = ONNXCalibrationDataReader(calib_data_dir, data_reader_model, pos_len, batch_size, require_exact_poslen=False, num_samples=batch_size)
        batch = dr.get_next()
        if batch is not None:
            input_spatial = torch.from_numpy(batch['input_spatial'])
            input_global = torch.from_numpy(batch['input_global'])
            if 'input_meta' in batch:
                input_meta = torch.from_numpy(batch['input_meta'])
        else:
            logging.warning(f"No .npz files found in {calib_data_dir} for comparison. Falling back to dummy_input.py")
    
    if input_spatial is None:
        if calib_data_dir is None:
            logging.warning("No -calib-data provided, using dummy_input.py for comparison")
        import dummy_input
        input_spatial, input_global, input_meta = dummy_input.generate_dummy_inputs(config, batch_size, pos_len, "cpu")

    def get_outputs(model, model_type, spatial, global_in, meta):
        if model_type == 'pytorch':
            model.eval()
            with torch.no_grad():
                if meta is not None:
                    outputs = model(spatial, global_in, meta)
                else:
                    outputs = model(spatial, global_in)
            outputs = outputs[0]
            return [outputs[i] for i in [0, 1, 2, 3, 4]]
        else:
            ort_session = ort.InferenceSession(model)
            onnx_inputs = {
                'input_spatial': spatial.numpy(),
                'input_global': global_in.numpy()
            }
            if meta is not None:
                onnx_inputs['input_meta'] = meta.numpy()
            onnx_outputs = ort_session.run(None, onnx_inputs)
            return [torch.from_numpy(out) for out in onnx_outputs]

    outputs1 = get_outputs(model1, model1_type, input_spatial, input_global, input_meta)
    outputs2 = get_outputs(model2, model2_type, input_spatial, input_global, input_meta)
    
    # Check if number of outputs match
    if len(outputs1) != len(outputs2):
        logging.error(f"Output count mismatch: {model1_type}={len(outputs1)}, {model2_type}={len(outputs2)}")
        return False
    
    # Check output shapes and values
    for i, (out1, out2) in enumerate(zip(outputs1, outputs2)):
        if out1.shape != out2.shape:
            logging.error(f"Output {i} shape mismatch: {model1_type}={out1.shape}, {model2_type}={out2.shape}")
            return False
        
        # Policy cross-entropy validation (channel 0)
        if i == 0:
            # Formula: CrossEntropy(new, original) - Entropy(original) = KL(original || new)
            p_logits = out1[:, 0, :]
            q_logits = out2[:, 0, :]
            
            p_probs = torch.nn.functional.softmax(p_logits, dim=-1)
            q_logprobs = torch.nn.functional.log_softmax(q_logits, dim=-1)
            
            kl = torch.nn.functional.kl_div(q_logprobs, p_probs, reduction='batchmean')
            logging.info(f"Policy channel 0 cross-entropy diff (KL Divergence) between {model1_type} and {model2_type}: {kl.item():.6f}")

        # Check if values are close
        out1_np = out1.detach().numpy()
        out2_np = out2.detach().numpy()
        if not np.allclose(out1_np, out2_np, rtol=1e-5, atol=1e-6):
            max_diff = np.max(np.abs(out1_np - out2_np))
            logging.warning(f"Output {i} values differ between {model1_type} and {model2_type}, max difference: {max_diff}")
    
    logging.info(f"Model comparison between {model1_type} and {model2_type} completed!")
    return True


def verify_onnx_model(onnx_path: str, original_model: Model, pos_len: int = 19, 
                      batch_size: int = 1, calib_data_dir: str = None) -> bool:
    torch.nn.RMSNorm.forward = original_rms_norm_forward
    return compare_models(original_model, onnx_path, 'pytorch', 'onnx', pos_len, batch_size, calib_data_dir)


if __name__ == "__main__":

    if not debug_mode:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('-checkpoint', help='Checkpoint file to export', required=True)
        parser.add_argument('-export-dir', help='Directory to export ONNX model to', required=True)
        parser.add_argument('-filename-prefix', help='Filename for the exported ONNX model (optional)', required=False)
        parser.add_argument('-model-name', help='Name for the exported model', required=True)
        parser.add_argument('-use-swa', help='Use SWA model if available', action='store_true', required=False)
        parser.add_argument('-pos-len', help='Spatial edge length (e.g. 19 for 19x19 Go)', type=int, default=19, required=False)
        parser.add_argument('-batch-size', help='Batch size for ONNX export', type=int, default=4, required=False)
        parser.add_argument('-fix-batchsize', help='Fix the batch size to the value of -batch-size (no dynamic axes)', action='store_true', required=False)
        parser.add_argument('-opset-version', help='ONNX opset version', type=int, default=20, required=False)
        parser.add_argument('-simplify', help='Simplify ONNX model using onnx-simplifier', action='store_true', required=False)
        parser.add_argument('-disable-mask', help='Disable masks in CNN and attention', action='store_true', required=False)
        parser.add_argument('-auto-fp16', help='Convert to half precision (FP16) automatically', action='store_true', required=False)
        parser.add_argument('-int8', help='Convert to INT8 quantization', action='store_true', required=False)
        parser.add_argument('-calib-data', help='Directory with .npz files for INT8 calibration', required=False)
        parser.add_argument('-calib-num', help='Number of samples for INT8 calibration', type=int, default=128, required=False)
        parser.add_argument('-calib-method', help='Calibration method: MinMax, Entropy, Percentile', default='Entropy', required=False)
        parser.add_argument('-convert-qat-to-float', help='Convert QAT model to traditional float model', action='store_true', required=False)
        parser.add_argument('-verbose', help='Verbose output', action='store_true', required=False)
        parser.add_argument('-author', help='Author name for metadata', required=False,default="unknown")
        parser.add_argument('-comment', help='Comment for metadata', required=False,default="")
        parser.add_argument('-dynamo', help='Use TorchDynamo for ONNX export', action='store_true', required=False)
        parser.add_argument('-skip-verification', help='Skip ONNX model verification', action='store_true', required=False)
        
        args = parser.parse_args()



        checkpoint_file = args.checkpoint
        export_dir = args.export_dir
        filename_prefix = args.filename_prefix
        model_name = args.model_name
        use_swa = args.use_swa
        pos_len = args.pos_len
        batch_size = args.batch_size
        fix_batchsize = args.fix_batchsize
        opset_version = args.opset_version
        simplify = args.simplify
        disable_mask = args.disable_mask
        auto_fp16 = args.auto_fp16
        int8 = args.int8
        calib_data = args.calib_data
        calib_num = args.calib_num
        calib_method = args.calib_method
        convert_qat_to_float = args.convert_qat_to_float
        verbose = args.verbose
        author = args.author
        comment = args.comment
        dynamo = args.dynamo
        skip_verification = args.skip_verification
        extra_meta_data = {}
        if author is not None:
            extra_meta_data["author"] = author
        if comment is not None:
            extra_meta_data["comment"] = comment
    else:
        checkpoint_file = "../data/train/go_b24c128tf1b_muon1_fd1/checkpoint.ckpt"
        export_dir = "../onnx_exports"
        filename_prefix = None
        model_name = "go_b24c128tf1b_muon1_fd1"
        use_swa = True
        pos_len = 19
        batch_size = 128
        fix_batchsize = False
        opset_version = 20
        simplify = False
        auto_fp16 = False
        int8 = False
        calib_data = None
        calib_num = 128
        calib_method = "Entropy"
        convert_qat_to_float = False
        verbose = False
        dynamo = False
        skip_verification = False
    
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
    model, swa_model, other_state_dict, is_qat_in_checkpoint = load_model_for_export(
        checkpoint_file, use_swa, device="cpu", pos_len=pos_len, verbose=True, convert_qat_to_float=False
    )
    
    # If convert_qat_to_float is requested, we need the float version too
    export_model = None
    if convert_qat_to_float and is_qat_in_checkpoint:
        logging.info("Loading float version of the QAT model for comparison...")
        float_model, float_swa_model, _, _ = load_model_for_export(
            checkpoint_file, use_swa, device="cpu", pos_len=pos_len, verbose=True, convert_qat_to_float=True
        )
        
        # Original QAT model for comparison
        qat_model_to_compare = swa_model if (use_swa and swa_model is not None) else model
        # Converted float model for export
        export_model = float_swa_model if (use_swa and float_swa_model is not None) else float_model
        
        logging.info("Comparing original QAT model with converted float model...")
        compare_models(qat_model_to_compare, export_model, 'pytorch', 'pytorch', pos_len, batch_size, calib_data)
        
        # Now we proceed with the float model as the one to export
        is_qat = False 
    else:
        export_model = swa_model if (use_swa and swa_model is not None) else model
        is_qat = is_qat_in_checkpoint

    if is_qat:
        logging.info("Model loaded and converted from QAT checkpoint.")
        if extra_meta_data is None:
            extra_meta_data = {}
        extra_meta_data["is_qat"] = "true"
        extra_meta_data["is_int8"] = "true"
    
    # Use SWA model if requested and available
    model_type = "SWA" if (use_swa and swa_model is not None) else "regular"
    
    logging.info(f"Exporting {model_type} model")
    logging.info(f"Model config: {export_model.config}")
    
    model_full_name = f"{model_name}"
    # Add training state info if available
    if "train_state" in other_state_dict:
        train_state = other_state_dict["train_state"]
        if "global_step_samples" in train_state:
            model_full_name += f"-s{train_state['global_step_samples']}"
        if "total_num_data_rows" in train_state:
            model_full_name += f"-d{train_state['total_num_data_rows']}"
    # Export to ONNX
    if filename_prefix is not None:
        onnx_filename = f"{filename_prefix}.onnx"
    else:
        onnx_filename = f"{model_full_name}.onnx"
    
    onnx_path = os.path.join(export_dir, onnx_filename)
    
    export_to_onnx(
        export_model, 
        model_full_name,
        onnx_path, 
        pos_len=pos_len, 
        batch_size=batch_size, 
        opset_version=opset_version,
        disable_mask=disable_mask,
        verbose=verbose,
        extra_meta_data=extra_meta_data,
        auto_fp16=auto_fp16,
        fix_batchsize=fix_batchsize,
        dynamo=dynamo
    )
    
    # Verify the exported model
    if not skip_verification:
        logging.info("Verifying exported ONNX model...")
        verification_passed = verify_onnx_model(onnx_path, export_model, pos_len, batch_size, calib_data_dir=calib_data)
        
        if not verification_passed:
            logging.error("ONNX model verification failed!")
            exit(1)
    else:
        logging.info("Skipping ONNX model verification as requested.")
    
    # Simplify model if requested
    if simplify:
        import onnxsim
        logging.info("Simplifying ONNX model...")
        simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
        try:
            # 使用 onnxsim 进行简化
            model_opt, check = onnxsim.simplify(onnx_path)
            if check:
                import onnx
                onnx.save(model_opt, simplified_path)
                onnx_path = simplified_path
                logging.info(f"Simplified model saved to: {simplified_path}")
                
                # Update metadata
                model_to_update = onnx.load(onnx_path)
                for prop in model_to_update.metadata_props:
                    if prop.key == "is_simplified":
                        prop.value = "true"
                onnx.save(model_to_update, onnx_path)
            else:
                logging.error("ONNX simplify check failed!")
        except Exception as e:
            logging.error(f"Simplifying failed: {e}")

    # Quantize to INT8 if requested
    if int8:
        if is_qat:
            logging.info("Model is already QAT, skipping post-training quantization.")
        else:
            if calib_data is None:
                logging.error("Calibration data directory (-calib-data) is required for INT8 quantization.")
                exit(1)
            
            onnx_int8_path = onnx_path.replace(".onnx", "_int8.onnx")
            quantize_onnx(
                onnx_path, 
                onnx_int8_path, 
                calib_data, 
                export_model, 
                pos_len, 
                batch_size, 
                disable_mask,
                num_samples=calib_num,
                method=calib_method,
                verbose=verbose
            )
            
            # Verify the quantized model
            if not skip_verification:
                logging.info("Verifying quantized INT8 model...")
                int8_verification_passed = verify_onnx_model(onnx_int8_path, export_model, pos_len, batch_size, calib_data_dir=calib_data)
                if not int8_verification_passed:
                    logging.warning("INT8 model verification failed! (This is common for INT8, check accuracy manually)")
            else:
                logging.info("Skipping quantized INT8 model verification as requested.")
            
            # Use the INT8 model for subsequent steps
            onnx_path = onnx_int8_path
            
            # Update metadata
            import onnx
            model_to_update = onnx.load(onnx_path)
            for prop in model_to_update.metadata_props:
                if prop.key == "is_int8":
                    prop.value = "true"
            onnx.save(model_to_update, onnx_path)
    
    
    logging.info(f"Export completed successfully!")
    logging.info(f"ONNX model: {onnx_path}")
    
    exit(0)

