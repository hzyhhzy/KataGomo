
source C:/ProgramData/Anaconda3/etc/profile.d/conda.sh
conda activate t

cd train
CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b18c384n -exportdir ../data/torchmodels_toexport_extra -exportprefix b18notrans
CUDA_VISIBLE_DEVICES="0" bash export.sh connectsix19x ../data 0
