mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    chmod +x ./engine/katago
    ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 20000
    
    cd trainsgd
    bash train.sh ../data b10c384n_sgd b10c384nbt-bng-mish-bnh 512 main -multi-gpus 0,1
    CUDA_VISIBLE_DEVICES="0" bash export.sh animalchessV3 ../data 0
    cd ..
done
