mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 50000
    cd train
    bash shuffle.sh ../data ../data ./ktmp 64 128
    #bash train.sh ../data ../data/shuffleddata/current b40c256 b40c256-fson-mish-rvgl-bnh 512 main -multi-gpus 0,1,2,3 -lr-scale 2.0
    #bash train.sh ../data ../data/shuffleddata/current b18trans b18c384nbt-fson-mish-rvgl-bnh 512 main -multi-gpus 0,1,2,3
    bash train.sh ../data ../data/shuffleddata/current b28c512n b28c512nbt-fson-mish-rvglr-bnh 512 main -multi-gpus 0,1,2,3 -lr-scale 0.5
    #bash train.sh ../data ../data/shuffleddata/current b18c384n b18c384nbt-fson-mish-rvglr-bnh 512 trainonly -multi-gpus 0,1,2,3 -lr-scale 2.0
    #bash train.sh ../data ../data/shuffleddata/current b18c384nbt b18c384nbt-fson-mish-rvgl-bnh 512 trainonly -multi-gpus 0,1,2,3 -lr-scale 2.0
    CUDA_VISIBLE_DEVICES="0" bash export.sh nogo9x2025 ../data 0
    python view_loss.py
    cd ..
done
