
# Modules that should not be quantized (input layers and heads/final trunk layers)
UNSUPPORTED_QUANTIZATION_MODULES = [
    "conv_spatial", "linear_global", "metadata_encoder",
    "norm_trunkfinal", "act_trunkfinal", "policy_head", "value_head",
    "norm_intermediate_trunkfinal", "act_intermediate_trunkfinal",
    "intermediate_policy_head", "intermediate_value_head"
]

def disable_qat_for_unsupported_modules(model):
    # Disable QAT for input layers and heads/final trunk layers
    # (the parts inside autocast(enabled=False) and the very beginning)
    for module_name in UNSUPPORTED_QUANTIZATION_MODULES:
        module = getattr(model, module_name, None)
        if module is not None:
            module.qconfig = None
            
def is_qat_checkpoint(state_dict):
    # Check for the existence of common QAT-specific keys
    # Usually QAT models contain activation_post_process or fake_quant modules
    # and these modules have scale and zero_point buffers
    qat_keys = [k for k in state_dict.keys() if "activation_post_process" in k or "fake_quant" in k]
    if not qat_keys:
        # Check inside 'model' key if it exists
        if "model" in state_dict:
            qat_keys = [k for k in state_dict["model"].keys() if "activation_post_process" in k or "fake_quant" in k]
    has_scale = any("scale" in k for k in qat_keys)
    has_zp = any("zero_point" in k for k in qat_keys)
    return has_scale and has_zp
    

def get_tensorrt_qat_qconfig():
    """
    针对 TensorRT 优化的 QAT 配置 (兼容 PyTorch 2.x+)
    """
    
    import torch
    # 从顶层引入基础类，而不是依赖不稳定的预设实例
    from torch.ao.quantization import (
        FakeQuantize, 
        MovingAverageMinMaxObserver, 
        PerChannelMinMaxObserver, 
        QConfig
    )
    # -------------------------------------------------------------------------
    # 1. 权重配置 (Weights): Per-Channel Symmetric
    # -------------------------------------------------------------------------
    # 使用 FakeQuantize 类直接构建

    usePerChannel = True # slow when using TensorRT for inference

    if usePerChannel:
        weight_qconfig = FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            quant_min=-128, 
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0  # 明确指定通道轴 (通常 Conv2d 的输出通道是第 0 维)
        )
    else:
        weight_qconfig = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128, 
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False  # TensorRT 8.x+ 不需要 reduce_range
        )
    
    # -------------------------------------------------------------------------
    # 2. 激活值配置 (Activations): Per-Tensor Symmetric
    # -------------------------------------------------------------------------
    act_qconfig = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128, 
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False  # TensorRT 8.x+ 不需要 reduce_range
    )
    
    return QConfig(activation=act_qconfig, weight=weight_qconfig)