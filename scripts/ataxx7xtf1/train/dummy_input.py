import torch
import modelconfigs

def generate_dummy_inputs(config, batch_size, pos_len, device):
    num_bin_input_features = modelconfigs.get_num_bin_input_features(config)
    num_global_input_features = modelconfigs.get_num_global_input_features(config)
    
    # binaryInputNCHW: (batch_size, num_bin_input_features, pos_len, pos_len)
    binary_input = torch.randn(batch_size, num_bin_input_features, pos_len, pos_len).to(device)
    binary_input[:,0,:,:] = 1.0 # mask channel
    
    # globalInputNC: (batch_size, num_global_input_features)
    global_input = torch.randn(batch_size, num_global_input_features).to(device)
    
    # metadataInputNC: (batch_size, num_meta_encoder_input_features) if model has metadata encoder
    input_meta = None
    if "metadata_encoder" in config and config["metadata_encoder"] is not None:
        num_meta_input_features = modelconfigs.get_num_meta_encoder_input_features(config)
        input_meta = torch.randn(batch_size, num_meta_input_features).to(device)
        
    return binary_input, global_input, input_meta
