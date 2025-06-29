mkdir data/
mkdir data/selfplay/
mkdir data/models/

while true
do
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 20000 
    cd train
    bash shuffle.sh ../data ./ktmp 16 512
    CUDA_VISIBLE_DEVICES="0" bash train.sh ../data b10c128 b10c128-fson-mish-rvglr-bnh 512 main -lr-scale 2

    CUDA_VISIBLE_DEVICES="0" bash export.sh dab6x ../data 0
    python view_loss.py
    cd ..
done
