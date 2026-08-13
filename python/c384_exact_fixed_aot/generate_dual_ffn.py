#!/usr/bin/env python3
"""Generate one exact B24/B28 C384/F1024 paired-projection SwiGLU object."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")

from bridge_codegen import render_dual_ffn_bridge
from contract import DEFAULT_SPACE, find_task, load_space, require
import generator_common as common


SEQUENCE = 225
INPUT_CHANNELS = 384
OUTPUT_CHANNELS = 1024
WIDE_CHANNELS = 2048


def patch_dense_source(source: str) -> str:
    replacements = (
        (
            "            cute.arch.setmaxregister_increase(self.mma_register_requirement)",
            "            pass  # SM120 rejects setmaxregister increase",
        ),
        (
            "            cute.arch.setmaxregister_decrease(self.load_register_requirement)",
            "            pass  # SM120 rejects setmaxregister decrease",
        ),
        (
            '''        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )''',
            '''        gC_mnl = cute.local_tile(
            mC_mnl,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1] // 2),
            (None, None, None),
        )''',
        ),
        (
            '''        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)''',
            '''        acc_shape = tiled_mma.partition_shape_C(self.tile_shape_mnk[:2])
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)''',
        ),
        (
            '''                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]''',
            '''                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        linear = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                        gate = tRS_rAcc[
                            (epi_idx + epi_tile_num) * size_tRS_rD + epi_v
                        ]
                        tRS_rD[epi_v] = (
                            linear / (1.0 + cute.math.exp(-linear)) * gate
                        ).to(self.acc_dtype)''',
        ),
    )
    for before, after in replacements:
        source = common.replace_exactly_once(source, before, after)
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--batch", type=int, required=True, choices=(24, 28))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--describe", action="store_true",
                        help="print the CPU-only materialized task and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    space = load_space(args.space.resolve())
    task = find_task(space, "dual_ffn", args.batch, args.candidate_id)
    if args.describe:
        print(json.dumps(task.__dict__, indent=2, default=list))
        return 0

    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    dense_path, cutlass_commit = common.validate_cutlass(args.cutlass_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / task.artifact_stem
    bridge_path = output_dir / f"{task.artifact_stem}_bridge.cu"
    metadata_path = base.with_suffix(".json")
    for path in (base.with_suffix(".h"), base.with_suffix(".o"), bridge_path,
                 metadata_path):
        require(not path.exists(), f"refusing to overwrite generated file: {path}")
    patched_path = output_dir / f"{task.artifact_stem}_dense_patch.py"
    require(not patched_path.exists(), f"refusing to overwrite: {patched_path}")
    patched_path.write_text(
        patch_dense_source(dense_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    module = common.load_module_from_source(
        patched_path, f"katago_c384_dual_ffn_{task.symbol_token}",
    )

    class Kernel(module.Sm120GemmKernel):
        def __init__(self) -> None:
            super().__init__(cutlass.Float16, task.tile)
            self.atom_layout = task.atom_layout
            self.num_mma_warps = (
                task.atom_layout[0] * task.atom_layout[1] * task.atom_layout[2]
            )
            self.threads_per_cta = (
                self.num_mma_warps + 1
            ) * self.num_threads_per_warp
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=self.num_mma_warps * self.num_threads_per_warp,
            )

        def _compute_stages(self, *unused):
            return task.ab_stages, task.epilogue_stages

        @staticmethod
        def _compute_grid(c, tile_shape_mnk, max_active_clusters):
            output_tile = (
                tile_shape_mnk[0], tile_shape_mnk[1] // 2, tile_shape_mnk[2],
            )
            return module.Sm120GemmKernel._compute_grid(
                c, output_tile, max_active_clusters,
            )

    gemm = Kernel()
    rows = task.token_rows
    a = common.make_tensor(
        from_dlpack, (rows, INPUT_CHANNELS, 1),
        (INPUT_CHANNELS, 1, rows * INPUT_CHANNELS), 0x10000,
    )
    # Logical B is [2F,K], backed by row-major paired [K,2F].
    b = common.make_tensor(
        from_dlpack, (WIDE_CHANNELS, INPUT_CHANNELS, 1),
        (1, WIDE_CHANNELS, WIDE_CHANNELS * INPUT_CHANNELS), 0x20000,
    )
    c = common.make_tensor(
        from_dlpack, (rows, OUTPUT_CHANNELS, 1),
        (OUTPUT_CHANNELS, 1, rows * OUTPUT_CHANNELS), 0x30000,
    )

    @cute.jit
    def launch(a_arg, b_arg, c_arg, stream: cuda.CUstream):
        gemm(a_arg, b_arg, c_arg, task.max_active_clusters, stream)

    compiled = cute.compile(
        launch, a, b, c,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    compiled.export_to_c(str(output_dir), file_name=task.artifact_stem)
    bridge_path.write_text(render_dual_ffn_bridge(task), encoding="utf-8")
    metadata = common.artifact_metadata(
        task, space, Path(__file__), dense_path, patched_path, output_dir,
        bridge_path, cutlass_commit,
        {
            "tile": list(task.tile),
            "effective_output_tile": list(task.effective_output_tile or ()),
            "atom_layout": list(task.atom_layout),
            "ab_stages": task.ab_stages,
            "epilogue_stages": task.epilogue_stages,
            "input": [rows, INPUT_CHANNELS],
            "paired_weights": [INPUT_CHANNELS, WIDE_CHANNELS],
            "output": [rows, OUTPUT_CHANNELS],
            "weight_pair_columns": 64,
            "epilogue": "silu-linear-times-gate",
        },
    )
    common.write_metadata(metadata_path, metadata)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
