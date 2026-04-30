#!/usr/bin/python3
import sys
import os
import argparse
import logging
import torch
from collections import OrderedDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)

def main():
    parser = argparse.ArgumentParser(description="Create Super SWA checkpoint via Median (or other methods)")
    parser.add_argument('-dir', help='Directory containing checkpoints', required=True)
    parser.add_argument('-out', help='Output checkpoint filename', default='checkpoint_superswa.ckpt')
    parser.add_argument('-pattern', help='File prefix pattern', default='checkpoint_prev')
    parser.add_argument('-count', help='Number of checkpoints', type=int, default=32)
    parser.add_argument('-method', help='Aggregation method: median, mean', default='median')
    
    args = parser.parse_args()
    
    directory = args.dir
    output_filename = args.out
    pattern = args.pattern
    count = args.count
    method = args.method
    
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist")
        sys.exit(1)
        
    output_path = os.path.join(directory, output_filename)
    
    # List of files
    files = [os.path.join(directory, f"{pattern}{i}.ckpt") for i in range(count)]
    
    # Check if files exist
    for f in files:
        if not os.path.exists(f):
            logging.error(f"File {f} not found")
            sys.exit(1)
            
    logging.info(f"Found {len(files)} checkpoints in {directory}. Method: {method}")
    
    # Load base checkpoint (index 0) to keep metadata
    logging.info(f"Loading base checkpoint: {files[0]}")
    base_checkpoint = torch.load(files[0], map_location='cpu')
    
    # We will modify base_checkpoint['model'] in place or replace it
    
    # Collect all model state dicts
    model_state_dicts = []
    
    # Add the base one
    model_state_dicts.append(base_checkpoint['model'])
    
    # Load the rest
    for i in range(1, count):
        logging.info(f"Loading {files[i]}...")
        ckpt = torch.load(files[i], map_location='cpu')
        model_state_dicts.append(ckpt['model'])
        del ckpt # Free memory of optimizer states etc
        
    logging.info("All checkpoints loaded. Processing parameters...")
    
    new_model_state_dict = OrderedDict()
    
    # Get keys from the first model
    keys = list(model_state_dicts[0].keys())
    
    total_keys = len(keys)
    for idx, key in enumerate(keys):
        # Collect tensors for this key from all models
        tensors = []
        for m_idx in range(count):
            tensors.append(model_state_dicts[m_idx][key])
            
        # Check if it's a tensor
        if torch.is_tensor(tensors[0]):
            # Stack them
            try:
                stacked = torch.stack(tensors)
                if method == 'median':
                    # torch.median returns (values, indices)
                    aggregated = torch.median(stacked, dim=0).values
                elif method == 'mean':
                    aggregated = torch.mean(stacked.float(), dim=0).to(tensors[0].dtype)
                elif method == 'max':
                    aggregated = torch.max(stacked, dim=0).values
                elif method == 'min':
                    aggregated = torch.min(stacked, dim=0).values
                else:
                    raise ValueError(f"Unknown method {method}")
                
                new_model_state_dict[key] = aggregated
            except Exception as e:
                logging.warning(f"Error processing key {key}: {e}. Keeping value from prev0.")
                new_model_state_dict[key] = tensors[0]
        else:
            # Not a tensor (e.g. metadata inside state dict?), just keep first one
            new_model_state_dict[key] = tensors[0]
            
        if idx % 100 == 0:
            print(f"Processed {idx}/{total_keys} keys...", end='\r')
            
    print(f"Processed {total_keys}/{total_keys} keys.")
    
    # Update base checkpoint
    base_checkpoint['model'] = new_model_state_dict
    
    # Save
    logging.info(f"Saving to {output_path}...")
    torch.save(base_checkpoint, output_path)
    logging.info("Done.")

if __name__ == "__main__":
    main()
