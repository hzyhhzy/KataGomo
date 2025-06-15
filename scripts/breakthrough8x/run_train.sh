mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 20000
    cd train
    bash shuffle.sh ../data ../data ./ktmp 12 128
    bash train.sh ../data b18c384n b18c384nbt-fson-mish-rvgl-bnh 512 main -multi-gpus 0,1
    CUDA_VISIBLE_DEVICES="0" bash export.sh breakthrough8x ../data 0
    python view_loss.py
    cd ..
done
