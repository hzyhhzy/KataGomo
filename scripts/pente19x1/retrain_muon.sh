mkdir data/
mkdir data/selfplay/
mkdir data/models/
cd train
#bash shuffleall.sh ../data ../data ./ktmp 32 128
cd ..
cd train_muon
#bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 4.0 -wd-scale 10.0 -max-epochs-this-instance 20 -samples-per-epoch 5000000
#bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 2.0 -wd-scale 10.0 -max-epochs-this-instance 30 -samples-per-epoch 5000000
#bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 1.0 -wd-scale 10.0 -max-epochs-this-instance 10 -samples-per-epoch 5000000
#bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 0.3 -wd-scale 10.0 -max-epochs-this-instance 5 -samples-per-epoch 5000000
#bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 0.1 -wd-scale 10.0 -max-epochs-this-instance 2 -samples-per-epoch 5000000

bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 0.3 -wd-scale 10.0 -max-epochs-this-instance 3 -samples-per-epoch 5000000
bash train_muon_ki.sh ../data ../data/shuffleddata/current b10c256n_muon b10c256nbt-bng-mish-rvglr-bnh 512 trainonly -muon-momentum 0.99 -lr-scale 0.1 -wd-scale 10.0 -max-epochs-this-instance 2 -samples-per-epoch 5000000
#bash train.sh ../data b18c384n b18c384nbt-fson-mish-rvglr-bnh 256 trainonly -multi-gpus 0,1
cd ..
cd ../connect6_19x3
bash run_train.sh
