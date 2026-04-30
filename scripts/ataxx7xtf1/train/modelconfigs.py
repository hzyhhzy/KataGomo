#!/usr/bin/python3
"""
This file contains a bunch of configs for models of different sizes.
See the bottom of this file "base_config_of_name" for a dictionary of all the different
base model architectures, and which ones are recommended of each different model size.

For each base model, additional configs are also pregenerated with different suffixes.

For example, for b10c384nbt, we also have models like:
b10c384nbt-mish  (use mish instead of relu)
b10c384nbt-bn-mish-rvgl (use batchnorm, mish, and repvgg-linear-style convolutions).

KataGo's main models for the distributed training run currently find the following to work
well or best: "-fson-mish-rvgl-bnh"
* Use fixed activation scale initialization + one batch norm for the whole net
* Mish activation
* Repvgg-linear-style convolutions
* Batch norm output head + non-batch-norm output head, where the former drives optimization
  but the latter is used for inference.

Version=15 by default, "-v11" or "-v102" to set the version to 11 or 102. 

For transformers, the recommended settings are "-bng-silu"
for example: b14c192h6tfrs-bng-silu
Transformer models should be exported by export_onnx.py
  
"""

from typing import Dict, Any, Union

ModelConfig = Dict[str,Any]

# version = 0 # V1 features, with old head architecture using crelus (no longer supported)
# version = 1 # V1 features, with new head architecture, no crelus
# version = 2 # V2 features, no internal architecture change.
# version = 3 # V3 features, selfplay-planned features with lots of aux targets
# version = 4 # V3 features, but supporting belief stdev and dynamic scorevalue
# version = 5 # V4 features, slightly different pass-alive stones feature
# version = 6 # V5 features, most higher-level go features removed
# version = 7 # V6 features, more rules support
# version = 8 # V7 features, asym, lead, variance time
# version = 9 # V7 features, shortterm value error prediction, inference actually uses variance time, unsupported now
# version = 10 # V7 features, shortterm value error prediction done properly
# version = 11 # V7 features, New architectures!
# version = 12 # V7 features, Optimistic policy head
# version = 13 # V7 features, Adjusted scaling on shortterm score variance, and made C++ side read in scalings.
# version = 14 # V7 features, Squared softplus for error variance predictions
# version = 15 # V7 features, Extra nonlinearity for pass output

def get_version(config: ModelConfig):
    return config["version"]

def get_num_bin_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15:
        return 22
    elif version == 101 or version == 102:
        return 22
    else:
        assert(False)

def get_num_global_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15:
        return 19
    elif version == 101 or version == 102:
        return 39
    else:
        assert(False)

def get_num_meta_encoder_input_features(config_or_meta_encoder_version: Union[ModelConfig,int]):
    if isinstance(config_or_meta_encoder_version,int):
        version = config_or_meta_encoder_version
    else:
        if "metadata_encoder" not in config:
            version = 0
        elif "meta_encoder_version" not in config["metadata_encoder"]:
            version = 1
        else:
            version = config["metadata_encoder"]["meta_encoder_version"]
    assert version == 1
    return 192

b1c6nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":6,
    "mid_num_channels":4,
    "gpool_num_channels":4,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","bottlenest2"],
    ],
    "p1_num_channels":4,
    "g1_num_channels":4,
    "v1_num_channels":4,
    "sbv2_num_channels":4,
    "num_scorebeliefs":2,
    "v2_size":6,
}

b2c16 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":16,
    "mid_num_channels":16,
    "gpool_num_channels":8,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regulargpool"],
    ],
    "p1_num_channels":8,
    "g1_num_channels":8,
    "v1_num_channels":8,
    "sbv2_num_channels":12,
    "num_scorebeliefs":2,
    "v2_size":12,
}

b2c16r = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":16,
    "mid_num_channels":16,
    "gpool_num_channels":8,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
    ],
    "p1_num_channels":8,
    "g1_num_channels":8,
    "v1_num_channels":8,
    "sbv2_num_channels":12,
    "num_scorebeliefs":2,
    "v2_size":16,
}

b4c32 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":32,
    "mid_num_channels":32,
    "gpool_num_channels":16,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regulargpool"],
        ["rconv4","regular"],
    ],
    "p1_num_channels":12,
    "g1_num_channels":12,
    "v1_num_channels":12,
    "sbv2_num_channels":24,
    "num_scorebeliefs":4,
    "v2_size":24,
}

b6c96 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":96,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regulargpool"],
        ["rconv4","regular"],
        ["rconv5","regulargpool"],
        ["rconv6","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}
b6c64 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":64,
    "mid_num_channels":64,
    "gpool_num_channels":16,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regulargpool"],
        ["rconv4","regular"],
        ["rconv5","regulargpool"],
        ["rconv6","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}
b12c64 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":64,
    "mid_num_channels":64,
    "gpool_num_channels":16,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regulargpool"],
        ["rconv4","regular"],
        ["rconv5","regulargpool"],
        ["rconv6","regular"],
        ["rconv7","regular"],
        ["rconv8","regulargpool"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}
b12c240nb1t = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":240,
    "mid_num_channels":64,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest1"],
        ["rconv2","bottlenest1gpool"],
        ["rconv3","bottlenest1"],
        ["rconv4","bottlenest1"],
        ["rconv5","bottlenest1gpool"],
        ["rconv6","bottlenest1"],
        ["rconv7","bottlenest1"],
        ["rconv8","bottlenest1gpool"],
        ["rconv9","bottlenest1"],
        ["rconv10","bottlenest1"],
        ["rconv11","bottlenest1gpool"],
        ["rconv12","bottlenest1"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}

b10c128 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":128,
    "mid_num_channels":128,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regulargpool"],
        ["rconv6","regular"],
        ["rconv7","regular"],
        ["rconv8","regulargpool"],
        ["rconv9","regular"],
        ["rconv10","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":64,
    "num_scorebeliefs":6,
    "v2_size":80,
}

b5c192nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2gpool"],
        ["rconv3","bottlenest2"],
        ["rconv4","bottlenest2gpool"],
        ["rconv5","bottlenest2"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":64,
    "num_scorebeliefs":6,
    "v2_size":80,
}

b15c192 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regular"],
        ["rconv7","regulargpool"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","regulargpool"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}

b20c256 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regular"],
        ["rconv7","regulargpool"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","regulargpool"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regular"],
        ["rconv17","regulargpool"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}




b10c384nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b10c256nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":128,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}



b30c320 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":320,
    "mid_num_channels":320,
    "gpool_num_channels":96,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b40c256 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
        ["rconv31","regulargpool"],
        ["rconv32","regular"],
        ["rconv33","regular"],
        ["rconv34","regular"],
        ["rconv35","regular"],
        ["rconv36","regulargpool"],
        ["rconv37","regular"],
        ["rconv38","regular"],
        ["rconv39","regular"],
        ["rconv40","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b18c384nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}


b40c384 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":384,
    "gpool_num_channels":128,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
        ["rconv31","regulargpool"],
        ["rconv32","regular"],
        ["rconv33","regular"],
        ["rconv34","regular"],
        ["rconv35","regular"],
        ["rconv36","regulargpool"],
        ["rconv37","regular"],
        ["rconv38","regular"],
        ["rconv39","regular"],
        ["rconv40","regular"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}


b60c320 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":320,
    "mid_num_channels":320,
    "gpool_num_channels":96,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
        ["rconv31","regulargpool"],
        ["rconv32","regular"],
        ["rconv33","regular"],
        ["rconv34","regular"],
        ["rconv35","regular"],
        ["rconv36","regulargpool"],
        ["rconv37","regular"],
        ["rconv38","regular"],
        ["rconv39","regular"],
        ["rconv40","regular"],
        ["rconv41","regulargpool"],
        ["rconv42","regular"],
        ["rconv43","regular"],
        ["rconv44","regular"],
        ["rconv45","regular"],
        ["rconv46","regulargpool"],
        ["rconv47","regular"],
        ["rconv48","regular"],
        ["rconv49","regular"],
        ["rconv50","regular"],
        ["rconv51","regulargpool"],
        ["rconv52","regular"],
        ["rconv53","regular"],
        ["rconv54","regular"],
        ["rconv55","regular"],
        ["rconv56","regulargpool"],
        ["rconv57","regular"],
        ["rconv58","regular"],
        ["rconv59","regular"],
        ["rconv60","regular"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}



b28c512nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":512,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2gpool"],
        ["rconv19","bottlenest2"],
        ["rconv20","bottlenest2"],
        ["rconv21","bottlenest2gpool"],
        ["rconv22","bottlenest2"],
        ["rconv23","bottlenest2"],
        ["rconv24","bottlenest2gpool"],
        ["rconv25","bottlenest2"],
        ["rconv26","bottlenest2"],
        ["rconv27","bottlenest2gpool"],
        ["rconv28","bottlenest2"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

b24c128tf1b = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":128,
    "mid_num_channels":128,
    "gpool_num_channels":32,
    "transformer_ffn_channels":512,
    "transformer_heads":4,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","transformer"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","transformer"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","transformer"],
        ["rconv13","regular"],
        ["rconv14","transformer"],
        ["rconv15","regular"],
        ["rconv16","transformer"],
        ["rconv17","regular"],
        ["rconv18","transformer"],
        ["rconv19","transformer"],
        ["rconv20","transformer"],
        ["rconv21","transformer"],
        ["rconv22","transformer"],
        ["rconv23","transformer"],
        ["rconv24","transformer"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}

b30c256bt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":128,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle"],
        ["rconv2","bottle"],
        ["rconv3","bottlegpool"],
        ["rconv4","bottle"],
        ["rconv5","bottle"],
        ["rconv6","bottle"],
        ["rconv7","bottlegpool"],
        ["rconv8","bottle"],
        ["rconv9","bottle"],
        ["rconv10","bottle"],
        ["rconv11","bottlegpool"],
        ["rconv12","bottle"],
        ["rconv13","bottle"],
        ["rconv14","bottle"],
        ["rconv15","bottlegpool"],
        ["rconv16","bottle"],
        ["rconv17","bottle"],
        ["rconv18","bottle"],
        ["rconv19","bottlegpool"],
        ["rconv20","bottle"],
        ["rconv21","bottle"],
        ["rconv22","bottle"],
        ["rconv23","bottlegpool"],
        ["rconv24","bottle"],
        ["rconv25","bottle"],
        ["rconv26","bottle"],
        ["rconv27","bottlegpool"],
        ["rconv28","bottle"],
        ["rconv29","bottle"],
        ["rconv30","bottle"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}



b11c96h4tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":96,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "transformer_ffn_channels":256,
    "transformer_heads":4,
    "transformer_kv_heads":4,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}




b11c96h3tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":96,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "transformer_ffn_channels":256,
    "transformer_heads":3,
    "transformer_kv_heads":3,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}

b11c96h3tfr = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":96,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "transformer_ffn_channels":384,
    "transformer_heads":3,
    "transformer_kv_heads":3,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropeg"],
        ["rconv2","transformerropeg"],
        ["rconv3","transformerropeg"],
        ["rconv4","transformerropeg"],
        ["rconv5","transformerropeg"],
        ["rconv6","transformerropeg"],
        ["rconv7","transformerropeg"],
        ["rconv8","transformerropeg"],
        ["rconv9","transformerropeg"],
        ["rconv10","transformerropeg"],
        ["rconv11","transformerropeg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}



b30c128h4tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":128,
    "mid_num_channels":128,
    "gpool_num_channels":32,
    "transformer_ffn_channels":320,
    "transformer_heads":4,
    "transformer_kv_heads":4,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
        ["rconv25","transformerropesg"],
        ["rconv26","transformerropesg"],
        ["rconv27","transformerropesg"],
        ["rconv28","transformerropesg"],
        ["rconv29","transformerropesg"],
        ["rconv30","transformerropesg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}



b14c192h6tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":192,
    "gpool_num_channels":32,
    "transformer_ffn_channels":512,
    "transformer_heads":6,
    "transformer_kv_heads":6,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}

b14c192h6tfr2s = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":192,
    "gpool_num_channels":32,
    "transformer_ffn_channels":512,
    "transformer_heads":6,
    "transformer_kv_heads":6,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformersg"],
        ["rconv8","transformersg"],
        ["rconv9","transformersg"],
        ["rconv10","transformersg"],
        ["rconv11","transformersg"],
        ["rconv12","transformersg"],
        ["rconv13","transformersg"],
        ["rconv14","transformersg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}

b7c256h8tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":32,
    "transformer_ffn_channels":768,
    "transformer_heads":8,
    "transformer_kv_heads":8,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}


b30c128h4tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":128,
    "mid_num_channels":128,
    "gpool_num_channels":32,
    "transformer_ffn_channels":320,
    "transformer_heads":4,
    "transformer_kv_heads":4,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
        ["rconv25","transformerropesg"],
        ["rconv26","transformerropesg"],
        ["rconv27","transformerropesg"],
        ["rconv28","transformerropesg"],
        ["rconv29","transformerropesg"],
        ["rconv30","transformerropesg"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}

b12c384h12tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":384,
    "gpool_num_channels":64,
    "transformer_ffn_channels":1024,
    "transformer_heads":12,
    "transformer_kv_heads":12,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b24c256h8tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "transformer_ffn_channels":768,
    "transformer_heads":8,
    "transformer_kv_heads":8,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b46c192h6tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":192,
    "gpool_num_channels":32,
    "transformer_ffn_channels":512,
    "transformer_heads":6,
    "transformer_kv_heads":6,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
        ["rconv25","transformerropesg"],
        ["rconv26","transformerropesg"],
        ["rconv27","transformerropesg"],
        ["rconv28","transformerropesg"],
        ["rconv29","transformerropesg"],
        ["rconv30","transformerropesg"],
        ["rconv31","transformerropesg"],
        ["rconv32","transformerropesg"],
        ["rconv33","transformerropesg"],
        ["rconv34","transformerropesg"],
        ["rconv35","transformerropesg"],
        ["rconv36","transformerropesg"],
        ["rconv37","transformerropesg"],
        ["rconv38","transformerropesg"],
        ["rconv39","transformerropesg"],
        ["rconv40","transformerropesg"],
        ["rconv41","transformerropesg"],
        ["rconv42","transformerropesg"],
        ["rconv43","transformerropesg"],
        ["rconv44","transformerropesg"],
        ["rconv45","transformerropesg"],
        ["rconv46","transformerropesg"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b18c384h12tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":384,
    "gpool_num_channels":64,
    "transformer_ffn_channels":1024,
    "transformer_heads":12,
    "transformer_kv_heads":12,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b10c768h24tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":768,
    "mid_num_channels":768,
    "gpool_num_channels":128,
    "transformer_ffn_channels":2048,
    "transformer_heads":24,
    "transformer_kv_heads":24,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

b21c512h16tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":512,
    "mid_num_channels":512,
    "gpool_num_channels":64,
    "transformer_ffn_channels":1536,
    "transformer_heads":16,
    "transformer_kv_heads":16,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

b40c512h16tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":512,
    "mid_num_channels":512,
    "gpool_num_channels":64,
    "transformer_ffn_channels":1536,
    "transformer_heads":16,
    "transformer_kv_heads":16,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
        ["rconv25","transformerropesg"],
        ["rconv26","transformerropesg"],
        ["rconv27","transformerropesg"],
        ["rconv28","transformerropesg"],
        ["rconv29","transformerropesg"],
        ["rconv30","transformerropesg"],
        ["rconv31","transformerropesg"],
        ["rconv32","transformerropesg"],
        ["rconv33","transformerropesg"],
        ["rconv34","transformerropesg"],
        ["rconv35","transformerropesg"],
        ["rconv36","transformerropesg"],
        ["rconv37","transformerropesg"],
        ["rconv38","transformerropesg"],
        ["rconv39","transformerropesg"],
        ["rconv40","transformerropesg"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}
b40c384h12tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":384,
    "gpool_num_channels":64,
    "transformer_ffn_channels":1024,
    "transformer_heads":12,
    "transformer_kv_heads":12,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
        ["rconv25","transformerropesg"],
        ["rconv26","transformerropesg"],
        ["rconv27","transformerropesg"],
        ["rconv28","transformerropesg"],
        ["rconv29","transformerropesg"],
        ["rconv30","transformerropesg"],
        ["rconv31","transformerropesg"],
        ["rconv32","transformerropesg"],
        ["rconv33","transformerropesg"],
        ["rconv34","transformerropesg"],
        ["rconv35","transformerropesg"],
        ["rconv36","transformerropesg"],
        ["rconv37","transformerropesg"],
        ["rconv38","transformerropesg"],
        ["rconv39","transformerropesg"],
        ["rconv40","transformerropesg"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

b40c384h12tfr = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":384,
    "gpool_num_channels":64,
    "transformer_ffn_channels":1536,
    "transformer_heads":12,
    "transformer_kv_heads":12,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropeg"],
        ["rconv2","transformerropeg"],
        ["rconv3","transformerropeg"],
        ["rconv4","transformerropeg"],
        ["rconv5","transformerropeg"],
        ["rconv6","transformerropeg"],
        ["rconv7","transformerropeg"],
        ["rconv8","transformerropeg"],
        ["rconv9","transformerropeg"],
        ["rconv10","transformerropeg"],
        ["rconv11","transformerropeg"],
        ["rconv12","transformerropeg"],
        ["rconv13","transformerropeg"],
        ["rconv14","transformerropeg"],
        ["rconv15","transformerropeg"],
        ["rconv16","transformerropeg"],
        ["rconv17","transformerropeg"],
        ["rconv18","transformerropeg"],
        ["rconv19","transformerropeg"],
        ["rconv20","transformerropeg"],
        ["rconv21","transformerropeg"],
        ["rconv22","transformerropeg"],
        ["rconv23","transformerropeg"],
        ["rconv24","transformerropeg"],
        ["rconv25","transformerropeg"],
        ["rconv26","transformerropeg"],
        ["rconv27","transformerropeg"],
        ["rconv28","transformerropeg"],
        ["rconv29","transformerropeg"],
        ["rconv30","transformerropeg"],
        ["rconv31","transformerropeg"],
        ["rconv32","transformerropeg"],
        ["rconv33","transformerropeg"],
        ["rconv34","transformerropeg"],
        ["rconv35","transformerropeg"],
        ["rconv36","transformerropeg"],
        ["rconv37","transformerropeg"],
        ["rconv38","transformerropeg"],
        ["rconv39","transformerropeg"],
        ["rconv40","transformerropeg"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}
b80c256h8tfrs = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "transformer_ffn_channels":768,
    "transformer_heads":8,
    "transformer_kv_heads":8,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","transformerropesg"],
        ["rconv2","transformerropesg"],
        ["rconv3","transformerropesg"],
        ["rconv4","transformerropesg"],
        ["rconv5","transformerropesg"],
        ["rconv6","transformerropesg"],
        ["rconv7","transformerropesg"],
        ["rconv8","transformerropesg"],
        ["rconv9","transformerropesg"],
        ["rconv10","transformerropesg"],
        ["rconv11","transformerropesg"],
        ["rconv12","transformerropesg"],
        ["rconv13","transformerropesg"],
        ["rconv14","transformerropesg"],
        ["rconv15","transformerropesg"],
        ["rconv16","transformerropesg"],
        ["rconv17","transformerropesg"],
        ["rconv18","transformerropesg"],
        ["rconv19","transformerropesg"],
        ["rconv20","transformerropesg"],
        ["rconv21","transformerropesg"],
        ["rconv22","transformerropesg"],
        ["rconv23","transformerropesg"],
        ["rconv24","transformerropesg"],
        ["rconv25","transformerropesg"],
        ["rconv26","transformerropesg"],
        ["rconv27","transformerropesg"],
        ["rconv28","transformerropesg"],
        ["rconv29","transformerropesg"],
        ["rconv30","transformerropesg"],
        ["rconv31","transformerropesg"],
        ["rconv32","transformerropesg"],
        ["rconv33","transformerropesg"],
        ["rconv34","transformerropesg"],
        ["rconv35","transformerropesg"],
        ["rconv36","transformerropesg"],
        ["rconv37","transformerropesg"],
        ["rconv38","transformerropesg"],
        ["rconv39","transformerropesg"],
        ["rconv40","transformerropesg"],
        ["rconv41","transformerropesg"],
        ["rconv42","transformerropesg"],
        ["rconv43","transformerropesg"],
        ["rconv44","transformerropesg"],
        ["rconv45","transformerropesg"],
        ["rconv46","transformerropesg"],
        ["rconv47","transformerropesg"],
        ["rconv48","transformerropesg"],
        ["rconv49","transformerropesg"],
        ["rconv50","transformerropesg"],
        ["rconv51","transformerropesg"],
        ["rconv52","transformerropesg"],
        ["rconv53","transformerropesg"],
        ["rconv54","transformerropesg"],
        ["rconv55","transformerropesg"],
        ["rconv56","transformerropesg"],
        ["rconv57","transformerropesg"],
        ["rconv58","transformerropesg"],
        ["rconv59","transformerropesg"],
        ["rconv60","transformerropesg"],
        ["rconv61","transformerropesg"],
        ["rconv62","transformerropesg"],
        ["rconv63","transformerropesg"],
        ["rconv64","transformerropesg"],
        ["rconv65","transformerropesg"],
        ["rconv66","transformerropesg"],
        ["rconv67","transformerropesg"],
        ["rconv68","transformerropesg"],
        ["rconv69","transformerropesg"],
        ["rconv70","transformerropesg"],
        ["rconv71","transformerropesg"],
        ["rconv72","transformerropesg"],
        ["rconv73","transformerropesg"],
        ["rconv74","transformerropesg"],
        ["rconv75","transformerropesg"],
        ["rconv76","transformerropesg"],
        ["rconv77","transformerropesg"],
        ["rconv78","transformerropesg"],
        ["rconv79","transformerropesg"],
        ["rconv80","transformerropesg"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}
sandbox = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regular"],
        ["rconv7","regulargpool"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","regulargpool"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regular"],
        ["rconv17","regulargpool"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


base_config_of_name = {
    # Micro-sized model configs
    "b1c6nbt": b1c6nbt,
    "b2c16": b2c16,
    "b2c16r": b2c16r,
    "b4c32": b4c32,
    "b6c64": b6c64,
    "b6c96": b6c96,
    "b12c64": b12c64,

    # Small model configs, not too different in inference cost from b10c128
    "b10c128": b10c128,
    "b5c192nbt": b5c192nbt,

    # Medium model configs, not too different in inference cost from b15c192
    "b15c192": b15c192,
    "b10c256nbt": b10c256nbt,

    # Roughly AlphaZero-sized, not too different in inference cost from b20c256
    "b20c256": b20c256,
    "b10c384nbt": b10c384nbt,  # Recommended best config for this cost


    # Roughly AlphaGoZero-sized, not too different in inference cost from b40c256
    "b30c320": b30c320,
    "b40c256": b40c256,
    "b18c384nbt": b18c384nbt,  # Recommended best config for this cost

    # Large model configs, not too different in inference cost from b60c320
    "b40c384": b40c384,
    "b60c320": b60c320,
    "b28c512nbt": b28c512nbt,  # Recommended best config for this cost


# Transformer models ------------------------------------------

# "b14c192h6tfrs" as an example:
# "b14" = 14 layers 
# "c192" = 192 hidden dims
# "h6" = 6 heads (traditional MHA, Q,K,V have same number of heads)
# "tfrs" = transformer with RoPE and SwiGLU
# ffn_width ~ 8/3 * hidden_dims if SwiGLU, otherwise = 4*hidden_dims
# param num ~ 12 * b * c * c
# RoPE is obviously better than no positional encoding (or using CNN as positional encoding like "b24c128tf1b")


# 1.3M parameter:
    "b12c240nb1t": b12c240nb1t, # 1.3M param CNN for comparition
    "b11c96h4tfrs": b11c96h4tfrs,
    "b11c96h3tfrs": b11c96h3tfrs, # Recommended
    "b11c96h3tfr": b11c96h3tfr, 


# 6M parameter:

    "b30c256bt": b30c256bt,  # 6M param CNN for comparition
    "b24c128tf1b": b24c128tf1b,  # old, CNN+transformer mixed, no RoPE/SwiGLU
    "b30c128h4tfrs":b30c128h4tfrs, # strong but slow
    "b14c192h6tfrs":b14c192h6tfrs, # good trade-off, Recommended
    "b7c256h8tfrs":b7c256h8tfrs,   # fast but weak

    "b14c192h6tfr2s": b14c192h6tfr2s, # partly rope

# 20M parameter:
    "b12c384h12tfrs":b12c384h12tfrs, 
    "b24c256h8tfrs":b24c256h8tfrs, 
    "b46c192h6tfrs":b46c192h6tfrs, 

# 32M parameter
    "b18c384h12tfrs":b18c384h12tfrs, 

# 70M parameter:
    "b10c768h24tfrs":b10c768h24tfrs, 
    "b21c512h16tfrs":b21c512h16tfrs, 
    "b40c384h12tfrs":b40c384h12tfrs, 
    "b40c384h12tfr":b40c384h12tfr, 
    "b80c256h8tfrs":b80c256h8tfrs, 
    
    "b40c512h16tfrs":b40c512h16tfrs, 
    "sandbox": sandbox,
}

config_of_name = {}
for name, base_config in base_config_of_name.items():
    config = base_config.copy()
    config_of_name[name] = config


for name, base_config in list(config_of_name.items()):
    # Fixup initialization
    config = base_config.copy()
    config["norm_kind"] = "fixup"
    config_of_name[name+""] = config

    # Fixed scaling normalization
#    config = base_config.copy()
#    config["norm_kind"] = "fixscale"
#    config_of_name[name+"-fs"] = config

    # Batchnorm without gamma terms
    config = base_config.copy()
    config["norm_kind"] = "bnorm"
    config_of_name[name+"-bn"] = config

    # Batchrenorm without gamma terms
    config = base_config.copy()
    config["norm_kind"] = "brenorm"
    config_of_name[name+"-brn"] = config

    # Fixed scaling normalization + Batchrenorm without gamma terms
#    config = base_config.copy()
#    config["norm_kind"] = "fixbrenorm"
#    config_of_name[name+"-fbrn"] = config

    # Batchnorm with gamma terms
    config = base_config.copy()
    config["norm_kind"] = "bnorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-bng"] = config

    # Batchrenorm with gamma terms
    config = base_config.copy()
    config["norm_kind"] = "brenorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-brng"] = config

    # Fixed scaling normalization + Batchrenorm with gamma terms
#    config = base_config.copy()
#    config["norm_kind"] = "fixbrenorm"
#    config["bnorm_use_gamma"] = True
#    config_of_name[name+"-fbrng"] = config

    # Fixed scaling normalization + ONE batch norm layer in the entire net.
    config = base_config.copy()
    config["norm_kind"] = "fixscaleonenorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-fson"] = config

for name, base_config in list(config_of_name.items()):
#    config = base_config.copy()
#    config["activation"] = "elu"
#    config_of_name[name+"-elu"] = config

#    config = base_config.copy()
#    config["activation"] = "gelu"
#    config_of_name[name+"-gelu"] = config

    config = base_config.copy()
    config["activation"] = "mish"
    config_of_name[name+"-mish"] = config

    config = base_config.copy()
    config["activation"] = "silu"
    config_of_name[name+"-silu"] = config

for name, base_config in list(config_of_name.items()):
    config = base_config.copy()
    config["use_attention_pool"] = True
    config_of_name[name+"-ap"] = config

for name, base_config in list(config_of_name.items()):
#    config = base_config.copy()
#    config["use_repvgg_init"] = True
#    config_of_name[name+"-rvgi"] = config

#    config = base_config.copy()
#    config["use_repvgg_linear"] = True
#    config_of_name[name+"-rvgl"] = config

    config = base_config.copy()
    config["use_repvgg_init"] = True
    config["use_repvgg_learning_rate"] = True
    config_of_name[name+"-rvglr"] = config

for name, base_config in list(config_of_name.items()):
    # Add intermediate heads, for use with self-distillation or embedding small net in big one.
#    config = base_config.copy()
#    config["has_intermediate_head"] = True
#    if("intermediate_head_blocks" not in config):
#        config["intermediate_head_blocks"] = len(config["block_kind"]) // 2
#    config_of_name[name+"-ih"] = config

    # Add parallel heads that uses the final trunk batchnorm.
    # The original normal heads disables the final trunk batchnorm
    # This only makes sense for configs that use some form of batchnorm.
    if "norm" in base_config["norm_kind"]:
        config = base_config.copy()
        config["has_intermediate_head"] = True
        if("intermediate_head_blocks" not in config):
            config["intermediate_head_blocks"] = len(config["block_kind"])
        config["trunk_normless"] = True
        config_of_name[name+"-bnh"] = config

#for name, base_config in list(config_of_name.items()):
#    config = base_config.copy()
#    config["metadata_encoder"] = {
#        "meta_encoder_version": 1,
#        "internal_num_channels": config["trunk_num_channels"],
#    }
#    config_of_name[name+"-meta"] = config

for name, base_config in list(config_of_name.items()):
    # Other games: v11
    config = base_config.copy()
    config["version"] = 11
    config_of_name[name+"-v11"] = config
    
    # Gomoku: v102
    config = base_config.copy()
    config["version"] = 102 
    config_of_name[name+"-v102"] = config
    
# print("Len of config = ",len(config_of_name))  # Len of config = 222000 !, so some functions are removed