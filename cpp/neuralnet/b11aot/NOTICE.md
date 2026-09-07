# Attribution and third-party code

The standalone export setup, inert DLPack metadata pattern and local FP16
softmax-rescale cast are derived from the publicly available
[doomoooo/KataGomo_fork final-migration](https://github.com/doomoooo/KataGomo_fork/tree/final-migration)
reference (inspected commit `5dfd8cb16`), particularly its
`python/sm120_generate_cute_qkv_aot.py` and
`cpp/neuralnet/fa4_aot/build_aot.py`. They are covered by that repository's
KataGo MIT license, reproduced in `licenses/KATAGO-MIT.txt`. This implementation
does not require or read that repository at generation time.

The vendored small SM120 adapter and source-level accumulator patch operate on
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).
Its source retains the notice:

Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
Pradeep Ramani, Tri Dao.

The upstream package's BSD 3-Clause license is reproduced verbatim in
`licenses/FLASH-ATTENTION-BSD-3-Clause.txt`. The full FA4 package is copied into
an explicitly requested external dependency overlay, not vendored here. Its
source notices must remain with that copy. Generated artifacts copy these
license notices for redistribution alongside the binary.

[NVIDIA CUTLASS/CuTe DSL](https://github.com/NVIDIA/cutlass) and its Python
compiler/runtime dependencies remain external build-time dependencies under
their own package licenses. No CUTLASS implementation source is copied here.
The final executable links CUDA runtime/driver libraries but does not need
Python, PyTorch, flash-attn, or the CuTe DSL runtime shared library.
