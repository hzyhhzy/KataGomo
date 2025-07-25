cd train_muon
#python save_model_for_export_manual.py -traindir ../data/train/b10c128muon -exportdir ../data/torchmodels_toexport_extra -exportprefix b10c128muon
python save_model_for_export_manual.py -traindir ../data/train/b10c256n_muon -exportdir ../data/torchmodels_toexport_extra -exportprefix b10c256n_muon
CUDA_VISIBLE_DEVICES="0" bash export.sh pente ../data 0