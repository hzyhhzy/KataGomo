mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    CUDA_VISIBLE_DEVICES="0,1" ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 30000
    cd train
    bash shuffle.sh ../data ./ktmp 16 256
    CUDA_VISIBLE_DEVICES="0,1" bash train.sh ../data b18trans b18c384nbt-mish 256 main
    CUDA_VISIBLE_DEVICES="0,1" bash train.sh ../data b18c384n b18c384nbt-mish 256 trainonly -samples-per-epoch 1200000
    
    CUDA_VISIBLE_DEVICES="0" bash export.sh connectsix19x2 ../data 0
    python view_loss.py
    cd ..
done
