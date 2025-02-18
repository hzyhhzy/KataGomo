
cd train
CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b28c512n -exportdir ../data/torchmodels_toexport -exportprefix b28c512n
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b40c128nbt -exportdir ../data/torchmodels_toexport -exportprefix b40c128nbt
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b10c256nbt -exportdir ../data/torchmodels_toexport -exportprefix b10c256nbt
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b8c192nbt -exportdir ../data/torchmodels_toexport -exportprefix b8c192nbt
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b5c192nbt -exportdir ../data/torchmodels_toexport -exportprefix b5c192nbt
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b4c128nbt -exportdir ../data/torchmodels_toexport -exportprefix b4c128nbt
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b3c96nbt -exportdir ../data/torchmodels_toexport -exportprefix b3c96nbt
#CUDA_VISIBLE_DEVICES="0" python save_model_for_export_manual.py -traindir ../data/train/b6c64 -exportdir ../data/torchmodels_toexport -exportprefix b6c64

CUDA_VISIBLE_DEVICES="0" bash export.sh nogo13x2025 ../data 0