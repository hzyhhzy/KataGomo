
source C:/ProgramData/Anaconda3/etc/profile.d/conda.sh
conda activate t4

cd train
CUDA_VISIBLE_DEVICES="1" python save_model_for_export_manual.py -traindir ../data/train/b18renju -exportdir ../data/torchmodels_toexport_extra -exportprefix b18renju
CUDA_VISIBLE_DEVICES="1" bash export.sh connectsix19x ../data 0
