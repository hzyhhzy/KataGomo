mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    CUDA_VISIBLE_DEVICES="0" ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 5000
    cd train
    bash shuffle.sh ../data C:/ktmp 16 1024 
    CUDA_VISIBLE_DEVICES="0" bash train.sh ../data b10c256n b10c256nbt-mish 512 main -lr-scale 2 -multi-gpus 0 -samples-per-epoch 1000000
    CUDA_VISIBLE_DEVICES="0" bash train.sh ../data b10c128 b10c128-mish 512 extra -lr-scale 2 -multi-gpus 0 -samples-per-epoch 1000000
    
    CUDA_VISIBLE_DEVICES="0" bash export.sh scoresix ../data 0
    python view_loss.py
    cd ..
done
