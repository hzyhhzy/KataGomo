mkdir data/
mkdir data/selfplay/
mkdir data/models/
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/lib/"
while true
do
    

    chmod +x ./engine/katago
    for i in {1..10}; do
        ./engine/katago selfplay -models-dir  data/models -config selfplay.cfg -output-dir data/selfplay  -max-games-total 60000
        
        cd train
        
        bash shuffle.sh ../data ../data ./ktmp 16 2048
        bash train_muon_ki.sh ../data ../data/shuffleddata/current b12c384h6tfrs2 b12c384h6tfrs2-bng-silu 2048 trainonly -multi-gpus 0,1 -lr-scale 0.25 -lr-base 6e-6 -master-port 23467 -samples-per-epoch 10000000 -max-epochs-this-instance 1

        cd ..
    done

    # fast drop and export
    cd train
    bash shuffle.sh ../data ../data ./ktmp 16 2048
    bash train_muon_ki.sh ../data ../data/shuffleddata/current b12c384h6tfrs2 b12c384h6tfrs2-bng-silu 2048 trainonly -multi-gpus 0,1 -lr-scale 0.177 -lr-base 6e-6 -master-port 23467 -samples-per-epoch 10000000 -max-epochs-this-instance 1
    bash shuffle.sh ../data ../data ./ktmp 16 2048
    bash train_muon_ki.sh ../data ../data/shuffleddata/current b12c384h6tfrs2 b12c384h6tfrs2-bng-silu 2048 trainonly -multi-gpus 0,1 -lr-scale 0.125 -lr-base 6e-6 -master-port 23467 -samples-per-epoch 10000000 -max-epochs-this-instance 1
    bash shuffle.sh ../data ../data ./ktmp 16 2048
    bash train_muon_ki.sh ../data ../data/shuffleddata/current b12c384h6tfrs2 b12c384h6tfrs2-bng-silu 2048 trainonly -multi-gpus 0,1 -lr-scale 0.088 -lr-base 6e-6 -master-port 23467 -samples-per-epoch 10000000 -max-epochs-this-instance 1
    bash shuffle.sh ../data ../data ./ktmp 16 2048
    bash train_muon_ki.sh ../data ../data/shuffleddata/current b12c384h6tfrs2 b12c384h6tfrs2-bng-silu 2048 trainonly -multi-gpus 0,1 -lr-scale 0.0625 -lr-base 6e-6 -master-port 23467 -samples-per-epoch 10000000 -max-epochs-this-instance 1
    bash shuffle.sh ../data ../data ./ktmp 16 2048
    bash train_muon_ki.sh ../data ../data/shuffleddata/current b12c384h6tfrs2 b12c384h6tfrs2-bng-silu 2048 main -multi-gpus 0,1 -lr-scale 0.03 -lr-base 6e-6 -master-port 23467 -samples-per-epoch 10000000 -max-epochs-this-instance 1
    CUDA_VISIBLE_DEVICES="0" bash export_onnx_for_selfplay.sh ataxx ../data 0
    cd ..



    
done
