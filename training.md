# This folder include scripts and configs used for training and some other scripts
# Usage for selfplay
1. First find a computer with a good GPU/many good GPUs. Both linux and windows are OK. Then install pytorch. Recommend installing `Git-Bash` if windows.     
2. Put the compiled engine into `./engine/katago` . Put required `.so` files (for Linux) or `.dll` files (for Windows) to the correct place to make the engine run.      
3. Then modify **some parameters** written below. The scripts in this folder is what I actually used, which may be not suitable for your computer.    
4. Finally run `run_train.sh` or something name like this.   
## Before running, these things should probably be modified: 
Below are the parameters most frequently be changed
Writing everything in detail is a massive work. So the below does not include everything. You should better also view the code itself.
### Model size
In `./run_train.sh `   

Example: `bash train.sh ../data b18c384n b18c384nbt-fson-mish-rvglr-bnh 128 main`   

`b18c384n` Name of your model   
 
`b18c384nbt-fson-mish-rvglr-bnh` Size and type of your model.   
 Defined in `./train/modelconfigs.py` .You can add new model sizes if needed.   
Recommend sizes:   
`b6c96` or `b10c128` for fast debug.    
`b10c256nbt` for short runs (1 ~ 5 RTX4090 Days, ~ 1M selfplay games).   
`b10c384nbt` for medium runs (5 ~ 20 RTX4090 Days, ~ 5M selfplay games).   
`b18c384nbt` for long runs (20 ~ 180 RTX4090 Days)  
`b28c512nbt` for very long runs (>180 RTX4090 Days)   
Recommend postfix: `-fson-mish-rvglr-bnh`   

`128` Batchsize. Better to make it match your GPU memory size. Larger for small nets and smaller for large nets

### Other training settings
In `./train/train.sh `  

`-pos-len 19`  Max board size in data. Should match `dataBoardLen` in `./selfplay.cfg`   
`-samples-per-epoch 1000000` Train how many samples (`sample=batchsize*step`) at once. I will discribe how to change it in the below.      
`-lr-scale 2.0` Learning rate scale. Recommend `2.0` at the beginning. Drop to `1.0` after trained ~1M selfplay games   
`-multi-gpus 0,1` GPU indexs used in this training. `-multi-gpus 0` or remove this if you have only one GPU.

In `./train/shuffle.sh `   

`-keep-target-rows 1100000`  Keep how many rows when shuffling. Should better slightly larger than `-samples-per-epoch`   



### Selfplay settings
In `./run_train.sh `   
`-max-games-total 20000` Run how many selfplay games at once.

In `./selfplay.cfg`   
`numNNServerThreadsPerModel = 2` How many GPUs are used   
`gpuToUseThread0 = 0` `gpuToUseThread1 = 1` `gpuToUseThread2 = 2` `...` Which GPUs are used   
For other things, the template provided here probably works well.   
Find out what they mean and change them if needed

### Other things in ./run_train.sh 
`export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"`   
Change `/root/lib/` to the place where you place `.so` files   
Not required if windows    
   
`CUDA_VISIBLE_DEVICES="..."`  Select the GPUs you decide to use   
Remove this or `CUDA_VISIBLE_DEVICES="0"` if only one GPU   

### Training(-samples-per-epoch) - Selfplay(-max-games-total) Rate
If you are training a new game, recommended: `-samples-per-epoch 1000000` `-max-games-total 10000`.    
Play 10000 selfplay games and train 1000000 samples(step\*batchsize) every generation.   

`samples-per-epoch / max-games-total` is proportional to how many times each sample will be trained.   
Too high will cost overfitting.   
Too low will waste selfplay games which is the major cost.   
Remember to change `-keep-target-rows` when you change `-samples-per-epoch`   

Value loss `vloss` is the mostly like to be overfitted because one full game rather than one move is just one data sample.    
Make sure `vloss_val - vloss_train < 0.05`, or you should increase `-max-games-total` or reduce `-samples-per-epoch`
