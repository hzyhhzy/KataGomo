mkdir data/
mkdir data/selfplay/
mkdir data/models/
conda activate t
while true
do
    
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 40000
    cd train
    bash shuffle.sh ../data C:/ktmp 16 1024 
    bash train.sh ../data b10c256n b10c256nbt-mish 512 main -lr-scale 1 -multi-gpus 0 -samples-per-epoch 2000000
    
    bash export.sh connectfour3d ../data 0
    python view_loss.py
    cd ..
done
