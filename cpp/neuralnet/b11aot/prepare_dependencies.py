#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy and patch FA4 into a NEW explicit directory; never edit site-packages."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patch_both16(data, lock):
    """Small audited math patch, not a transformation of another generator."""
    spec = lock["sources"]["flash_attn/cute/flash_fwd.py"]
    normalized = data.replace(b"\r\n", b"\n")
    if sha(normalized) == spec["canonical_lf_sha256"]:
        return normalized
    if sha(data) != spec["unpatched_sha256"]:
        raise ValueError("flash_fwd.py is neither the pinned base nor validated both16 source")
    text = normalized.decode("utf-8")
    substitutions = (
        ("        q_subtile_factor: int = 1,\n", "        q_subtile_factor: int = 1,\n"
         "        qk_acc_dtype: Type[cutlass.Numeric] = Float32,\n"
         "        pv_acc_dtype: Type[cutlass.Numeric] = Float32,\n"),
        ("        self.qk_acc_dtype = Float32\n", "        self.qk_acc_dtype = qk_acc_dtype\n"
         "        self.pv_acc_dtype = pv_acc_dtype\n"),
        ("        tiled_mma_qk = cute.make_tiled_mma(\n"
         "            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),",
         "        tiled_mma_qk = cute.make_tiled_mma(\n"
         "            warp.MmaF16BF16Op(self.dtype, self.qk_acc_dtype, (16, 8, 16)),"),
        ("        tiled_mma_pv = cute.make_tiled_mma(\n"
         "            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),",
         "        tiled_mma_pv = cute.make_tiled_mma(\n"
         "            warp.MmaF16BF16Op(self.dtype, self.pv_acc_dtype, (16, 8, 16)),"),
        ("        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)",
         "        acc_O = cute.make_rmem_tensor(acc_shape_O, self.pv_acc_dtype)"),
        ("        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)",
         "        acc_S = cute.make_rmem_tensor(acc_shape_S, self.qk_acc_dtype)"),
    )
    for before, after in substitutions:
        if text.count(before) != 1:
            raise ValueError("unexpected pinned source around: " + before[:80])
        text = text.replace(before, after)
    patched = text.encode("utf-8")
    if sha(patched) != spec["canonical_lf_sha256"]:
        raise ValueError("both16 patch did not reproduce the validated canonical source")
    return patched


def patch_auxiliary(data, lock, name):
    """Preserve dtype on -inf mask literals and softmax exp writeback."""
    spec = lock["sources"]["flash_attn/cute/" + name]
    normalized = data.replace(b"\r\n", b"\n")
    if sha(normalized) == spec["canonical_lf_sha256"]:
        return normalized
    if sha(data) != spec["unpatched_sha256"]:
        raise ValueError("unexpected source before FP16 compatibility patch: " + name)
    text = normalized.decode("utf-8")
    if name == "softmax.py":
        substitutions = [("acc_S_mn[r, None].store(acc_S_row_exp)",
                          "acc_S_mn[r, None].store(acc_S_row_exp.to(acc_S_mn.element_type))", 1)]
    else:
        substitutions = [
            ("from cutlass import Float32, Int32, Uint32, const_expr\n",
             'from cutlass import Float16, Float32, Int32, Uint32, const_expr\n\n\n'
             'def _neg_inf(t):\n'
             '    """-inf literal matching the tensor\'s element dtype (fp16 accumulators)."""\n'
             '    if t.element_type == Float16:\n'
             '        return Float16(float("-inf"))\n'
             '    return -Float32.inf\n', 1),
            ("X[c] = X[c] if in_bound else -Float32.inf", "X[c] = X[c] if in_bound else _neg_inf(X)", 2),
            ("X[r, c] = X[r, c] if in_bound else -Float32.inf", "X[r, c] = X[r, c] if in_bound else _neg_inf(X)", 1),
            ("acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]", "acc_S_mn[r, c] = _neg_inf(acc_S_mn) if oob else acc_S_mn[r, c]", 1),
            ("acc_S_mn[r, col] = -cutlass.Float32.inf", "acc_S_mn[r, col] = _neg_inf(acc_S_mn)", 1),
            ("acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf", "acc_S_mn[r, col] = acc_S_mn[r, col] if cond else _neg_inf(acc_S_mn)", 2),
            ("                                    -Float32.inf\n", "                                    _neg_inf(acc_S_mn)\n", 1),
            ("                                -Float32.inf\n", "                                _neg_inf(acc_S_mn)\n", 2),
            ("acc_S_mn[r, c] = -Float32.inf", "acc_S_mn[r, c] = _neg_inf(acc_S_mn)", 1),
            ("acc_S[i] = acc_S[i] if cond else -Float32.inf", "acc_S[i] = acc_S[i] if cond else _neg_inf(acc_S)", 1),
            ("acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]", "acc_S[i] = _neg_inf(acc_S) if global_col >= self.seqlen_k else acc_S[i]", 1),
            ("acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]", "acc_S[i] = _neg_inf(acc_S) if mask_row >= self.seqlen_q else acc_S[i]", 1),
        ]
    for before, after, count in substitutions:
        if text.count(before) != count:
            raise ValueError("unexpected patch count in " + name + ": " + before[:80])
        text = text.replace(before, after)
    patched = text.encode("utf-8")
    if sha(patched) != spec["canonical_lf_sha256"]:
        raise ValueError("patch failed to reproduce validated canonical source: " + name)
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path,
                        help="Path to installed flash_attn package; default from distribution metadata")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New/empty dependency overlay directory, outside this source tree")
    args = parser.parse_args()
    source = args.source_package
    if source is None:
        source = Path(importlib.metadata.distribution("flash-attn-4").locate_file("flash_attn"))
    source, output = source.resolve(), args.output_dir.resolve()
    if source.name != "flash_attn" or not (source / "cute/flash_fwd.py").is_file():
        parser.error("--source-package must be a flash_attn package directory")
    if output == source or source in output.parents or HERE == output or HERE in output.parents:
        parser.error("output must be separate from the source package and b11aot source tree")
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty; existing files are never overwritten")
    lock = json.loads((HERE / "dependencies.json").read_text())
    patched = {"flash_fwd.py": patch_both16((source / "cute/flash_fwd.py").read_bytes(), lock)}
    for name in ("mask.py", "softmax.py"):
        patched[name] = patch_auxiliary((source / "cute" / name).read_bytes(), lock, name)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output / "flash_attn", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, data in patched.items():
        (output / "flash_attn/cute" / name).write_bytes(data)
    shutil.copyfile(HERE / "patches/flash_fwd_sm120.py", output / "flash_attn/cute/flash_fwd_sm120.py")
    shutil.copyfile(HERE / "licenses/FLASH-ATTENTION-BSD-3-Clause.txt", output / "LICENSE.flash-attention")
    records = {}
    for relative, expected in lock["sources"].items():
        data = (output / relative).read_bytes()
        canonical = sha(data.replace(b"\r\n", b"\n"))
        if canonical != expected["canonical_lf_sha256"]:
            raise RuntimeError("prepared source mismatch: " + relative)
        records[relative] = {"sha256": sha(data), "canonical_lf_sha256": canonical}
    (output / "preparation.json").write_text(json.dumps({
        "source_package": str(source), "output": str(output), "GPU_used": False,
        "recipe_sha256": sha(Path(__file__).read_bytes()), "sources": records,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
