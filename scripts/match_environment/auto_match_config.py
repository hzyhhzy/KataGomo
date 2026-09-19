"""Edit here, then run: python3 script_generator.py (Linux runtime, portable folder)."""

# CUDA_VISIBLE_DEVICES: physical GPU IDs. Engine IDs below are remapped to 0,1,...
VISIBLE_GPUS = [0]
# Relative to this folder, absolute Linux paths, and ~/... are supported.
# System CUDA/cuDNN paths normally need no entry; add non-system .so directories.
LIBRARY_PATHS = ["lib"]
PYTHON_COMMAND = "python3"

# Assets below are relative to this folder. Native CUDA and ONNX use different builds.
ENGINE_NATIVE = "engine/katago"
ENGINE_ONNX = "engine/katago_onnx"
MODEL_FILE_PREFERENCE = ["model.bin.gz", "model.bin", "model.onnx"]
GROUP_A = "modelsGroupA"
GROUP_B = "modelsGroupB"
# True: skip same-name pairs, retain only one orientation of each name pair.
# Identical N-entry groups: True -> N*(N-1)/2 matches; False -> all N*N matches.
DEDUPLICATE_MATCHES = True
BASE_CONFIG = "match.cfg"
OPENING_FILE = "openings/renju5_982.txt"  # "", '""', or None disables fixed openings.

# Each A x B pair plays this many games. 1000 = first 500 openings, both colors.
GAMES_PER_PAIR = 1000
DEFAULT_PLAYOUTS = 100  # Bare folders use this budget; recorded name becomes <folder>_<N>po.
GAME_THREADS = 200
SEARCH_THREADS = 1
BATCH_SIZE = 36
NN_SERVER_THREADS_PER_MODEL = 4
NN_CACHE_POWER = 24
GRAPH_SEARCH = True

# Additional scalar engine overrides, e.g. {"basicRules": "FREESTYLE"}.
# Pair names/files, playouts, scheduling, GPU mapping and values above are reserved.
EXTRA_OVERRIDES = {}

# Same experiment settings reuse verified completed pairs; change this for a fresh run.
EXPERIMENT_TAG = "renju-first500"
CALCULATE_ELO_AT_END = True
# None => mean Elo 0; otherwise a rating name, e.g. b14c192h6tflrs_100po.
ELO_REFERENCE = None
ELO_REFERENCE_RATING = 0.0
# Add this many virtual wins to EACH side of each played pair (avoids infinite Elo).
# 0 disables smoothing but all-win/all-loss data may have no finite solution.
ELO_PSEUDO_WINS = 0.5
