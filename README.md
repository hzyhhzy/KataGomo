# HZY's repo for KataGo Training
See `python` directory for training scripts.
## Muon and Soap optimizer
`train_muon_ki.sh` for muon optimizer,    
`train_soap.sh` for soap optimizer.   
Adapted by @LK
  
### Conclusion
**Muon and Soap are significantly better than SGD.** ~0.1x steps for same loss.   
Soap is a bit better than Muon with same steps, but much slower.    
So muon is recommended.   

## Transformer
See `modelconfigs.py` and `model_pytorch.py` for transformer configs.
`transformer` and `transformer2` have been tested and works well. 
`transformer3` has QK-norm and have not been tested.

**No positional encoding**. So all structures in `modelconfigs.py` are CNN+Transformer mixed
### Conclusion
CNN+Transformer is stronger than CNN with same params. Roughly equal to a CNN with 1.5x~2x params.   
But much slower. So whether it worths is still unknown.

## Other
1. `torch.compile` 1.6x speed and reduce memory usage.
2. `x*(gamma+1)` instead of `x*gamma`. To avoid `gamma` becoming close to 0.
3. disable autocast and higher weightdecay for output head to avoid NAN.
4. many details