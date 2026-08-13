# SM120 CUDA specializations

This directory contains shape-qualified CUDA implementations, not general
backend orchestration. The `c256` and `c384` profiles are peers. Kernels that
serve both channel widths live in `shared`; generic selectors, cuBLAS/cuDNN
fallbacks, and execution sequencing remain in `cpp/neuralnet`.

Directory and model naming follow two separate axes:

- `b24` / `b36` describe model depth.
- `bs28` / `bs36` describe runtime batch size.

Model depth must not be embedded in a reusable per-layer kernel ABI. A
depth-specific choice belongs in the whole-model plan or exact transaction,
which may reference the same low-level descriptor for compatible b24 and b36
models.

Legacy headers remain at their former `cpp/neuralnet` paths as forwarding
headers for generated packages and external ABI consumers. Historical C
symbols, tactic IDs, markers, structure layouts, and ABI versions are kept
unchanged.

The CUDA tactics and validation gates under `c256/experimental_int8` are
explicitly opt-in and are excluded from the standard production build unless
their CMake option is enabled. The CPU quantization helpers remain available
to the model loader, as before the directory move.
