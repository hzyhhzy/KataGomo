mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 20000
    cd train
    bash shuffle.sh ../data ../data ./ktmp 32 128
    bash train.sh ../data b10c256n b10c256nbt-fson-mish-rvglr-bnh 512 main -multi-gpus 0,1
    #bash train.sh ../data b10c128 b10c128-fson-mish-rvglr-bnh 1024 trainonly -multi-gpus 0,1
    CUDA_VISIBLE_DEVICES="0" bash export.sh pente ../data 0

    python view_loss.py
    #cd ..
    #cd train_muon
    #bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c128muon b10c128-bng-mish-rvglr-bnh 1024 trainonly -muon-momentum 0.99 -lr-scale 0.3 -wd-scale 10.0
    #bash train.sh ../data b18c384n b18c384nbt-fson-mish-rvglr-bnh 256 trainonly -multi-gpus 0,1
    #cd ..
done
