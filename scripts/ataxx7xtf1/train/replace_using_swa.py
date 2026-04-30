#!/usr/bin/python3
import sys
import os
import argparse
import logging
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)

def main():
    parser = argparse.ArgumentParser(description="Replace the original model using SWA model")
    parser.add_argument('-input', help='Input checkpoint filename', required=True)
    parser.add_argument('-output', help='Output checkpoint filename', required=True)
    
    args = parser.parse_args()
    
    input_filename = args.input
    output_filename = args.output
    
        
    
    
    # Check if files exist
    if not os.path.exists(input_filename):
        logging.error(f"File {f} not found")
        sys.exit(1)
            
    
    # Load base checkpoint (index 0) to keep metadata
    logging.info(f"Loading base checkpoint: {input_filename}")
    base_checkpoint = torch.load(input_filename, map_location='cpu',weights_only=False)
    
    print(base_checkpoint.keys())
    # Get keys from the first model
    keys = list(base_checkpoint["model"].keys())

    
    total_keys = len(keys)
    for idx, key in enumerate(keys):
        base_checkpoint["model"][key] = base_checkpoint["swa_model_0"]["module."+key] 
    
    
    # Save
    logging.info(f"Saving to {output_filename}...")
    torch.save(base_checkpoint, output_filename)
    logging.info("Done.")

if __name__ == "__main__":
    main()
