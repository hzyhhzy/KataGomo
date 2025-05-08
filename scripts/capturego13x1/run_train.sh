mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    CUDA_VISIBLE_DEVICES="0,1" ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 17000
    cd train
    bash shuffle.sh ../data ./ktmp 16 256
    CUDA_VISIBLE_DEVICES="0,1" bash train.sh ../data b18trans b18c384nbt-fson-mish-rvgl-bnh 256 main -multi-gpus 0,1 -lr-scale 1.0
    CUDA_VISIBLE_DEVICES="0,1" bash train.sh ../data b28trans b28c512nbt-fson-mish-rvgl-bnh 256 extra -multi-gpus 0,1 -lr-scale 1.0
    CUDA_VISIBLE_DEVICES="0,1" bash train.sh ../data b10c384n b10c384nbt-fson-mish-rvgl-bnh 256 extra -multi-gpus 0,1
    
    CUDA_VISIBLE_DEVICES="0" bash export.sh capturego13x ../data 0
    python view_loss.py
    cd ..
done
