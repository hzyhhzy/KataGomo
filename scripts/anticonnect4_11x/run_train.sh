mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 30000
    cd train
    bash shuffle.sh ../data ./ktmp 32 128
    bash train.sh ../data b10c384n b10c384nbt-fson-mish-rvglr-bnh 1024 main -multi-gpus 0,1
    CUDA_VISIBLE_DEVICES="0" bash export.sh anticon4_11x ../data 0
    python view_loss.py
    cd ..
done
