#!/usr/bin/env python3
"""Generate one B24/B28 C384 packed-QKV + learned-RoPE CuTe object.

This is a source/object generator, not a benchmark. It specializes the pinned
CUTLASS SM120 dense GEMM for one exact M and fuses Q/K rotation into the FP16
register-fragment epilogue. It never launches the generated kernel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUTE_DSL_ARCH"] = "sm_120"

from bridge_codegen import render_qkv_rope_bridge
from contract import DEFAULT_SPACE, find_task, load_space, require
import generator_common as common


SEQUENCE = 225
INPUT_CHANNELS = 384
OUTPUT_CHANNELS = 1152
ROPE_PAIRS = 192


def patch_dense_source(source: str, rows: int) -> str:
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
            """        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,""",
            """        c: cute.Tensor,
        rope_table: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,""",
        ),
        (
            """            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(""",
            """            self.epi_smem_layout_staged,
            rope_table,
            tile_sched_params,
        ).launch(""",
        ),
        (
            """        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,""",
            """        epi_smem_layout_staged: cute.ComposedLayout,
        rope_table: cute.Tensor,
        tile_sched_params: utils.PersistentTileSchedulerParams,""",
        ),
        (
            """                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                # Allocate D registers.""",
            f"""                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                coord_mnl = cute.make_identity_tensor(({rows}, 1152, 1))
                coord_tile = cute.local_tile(
                    coord_mnl,
                    cute.slice_(self.tile_shape_mnk, (None, None, 0)),
                    (tile_coord_mnl[0], tile_coord_mnl[1], tile_coord_mnl[2]),
                )
                tCgCoord = thr_mma.partition_C(coord_tile)
                tRS_rCoord = tiled_copy_r2s.retile(tCgCoord)

                # Allocate D registers.""",
        ),
        (
            """                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Type conversion""",
            """                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # N128 tiles 0..2 are Q, 3..5 are K, and 6..8 are V.
                    # Only Q/K are rotated; tail M rows remain predicate-masked
                    # by the pinned dense epilogue.
                    if tile_coord_mnl[1] < 6:
                        base_m = tRS_rCoord[0][0]
                        base_n = tRS_rCoord[0][1]
                        qk_n_offset = 384 if tile_coord_mnl[1] >= 3 else 0
                        epi_m = (epi_idx % 2) * 64
                        epi_n = (epi_idx // 2) * 32
                        for rope_pair in cutlass.range_constexpr(size_tRS_rD // 2):
                            idx0 = rope_pair * 2
                            idx1 = idx0 + 1
                            global_m = base_m + epi_m + (rope_pair % 2) * 8
                            global_n = base_n + epi_n + (rope_pair // 2) * 16
                            hp = (global_n - qk_n_offset) // 2
                            xy = global_m % 225
                            cos_v = rope_table[(xy, hp, 0)]
                            sin_v = rope_table[(xy, hp, 1)]
                            q0 = tRS_rD[idx0]
                            q1 = tRS_rD[idx1]
                            tRS_rD[idx0] = q0 * cos_v - q1 * sin_v
                            tRS_rD[idx1] = q0 * sin_v + q1 * cos_v

                    # Type conversion""",
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
    space_path = args.space.resolve()
    space = load_space(space_path)
    task = find_task(space, "qkv_rope", args.batch, args.candidate_id)
    if args.describe:
        print(json.dumps(task.__dict__, indent=2, default=list))
        return 0

    # Lazy imports keep --describe and py_compile CPU/toolchain-only.
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cuda.bindings import _version as cuda_bindings_version
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
        patch_dense_source(dense_path.read_text(encoding="utf-8"),
                           task.token_rows),
        encoding="utf-8",
    )
    module = common.load_module_from_source(
        patched_path, f"katago_c384_qkv_rope_{task.symbol_token}",
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

    gemm = Kernel()
    rows = task.token_rows
    a = common.make_tensor(
        from_dlpack, (rows, INPUT_CHANNELS, 1),
        (INPUT_CHANNELS, 1, rows * INPUT_CHANNELS), 0x10000,
    )
    # Logical B is [N,K], backed by row-major [K,N].
    b = common.make_tensor(
        from_dlpack, (OUTPUT_CHANNELS, INPUT_CHANNELS, 1),
        (1, OUTPUT_CHANNELS, OUTPUT_CHANNELS * INPUT_CHANNELS), 0x20000,
    )
    c = common.make_tensor(
        from_dlpack, (rows, OUTPUT_CHANNELS, 1),
        (OUTPUT_CHANNELS, 1, rows * OUTPUT_CHANNELS), 0x30000,
    )
    table = common.make_tensor(
        from_dlpack, (SEQUENCE, ROPE_PAIRS, 2),
        (ROPE_PAIRS * 2, 2, 1), 0x40000,
    )

    @cute.jit
    def launch(a_arg, b_arg, c_arg, table_arg, stream: cuda.CUstream):
        gemm(
            a_arg, b_arg, c_arg, table_arg,
            task.max_active_clusters, stream,
        )

    common.bind_local_stream_annotation(launch,cuda.CUstream)

    compiled = cute.compile(
        launch, a, b, c, table,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    compiled.export_to_c(str(output_dir), file_name=task.artifact_stem)
    bridge_path.write_text(render_qkv_rope_bridge(task), encoding="utf-8")
    metadata = common.artifact_metadata(
        task, space, Path(__file__), dense_path, patched_path, output_dir,
        bridge_path,cutlass_commit,cutlass,cuda,cuda_bindings_version,
        {
            "tile": list(task.tile),
            "atom_layout": list(task.atom_layout),
            "stage_policy": task.stage_policy,
            "input": [rows, INPUT_CHANNELS],
            "packed_weights": [INPUT_CHANNELS, OUTPUT_CHANNELS],
            "packed_output": [rows, OUTPUT_CHANNELS],
            "rope_table_half2": [SEQUENCE, ROPE_PAIRS],
            "qk_tiles": [0, 1, 2, 3, 4, 5],
            "v_tiles": [6, 7, 8],
        },
    )
    common.write_metadata(metadata_path, metadata)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
