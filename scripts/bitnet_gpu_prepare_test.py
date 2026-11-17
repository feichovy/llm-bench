#!/usr/bin/env python3
"""Apply two audited BitNet GPU fixes, or check the rebuilt CUDA library.

Run with the GPU virtual environment on the Linux VM:
  python bitnet_gpu_prepare_test.py patch --repo /path/to/BitNet --log-dir /path/to/logs
  python bitnet_gpu_prepare_test.py check --repo /path/to/BitNet --log-dir /path/to/logs

Only ``patch`` changes source files. Neither command changes model weights.
"""

from __future__ import annotations

import argparse
import ctypes
import difflib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from uuid import uuid4


PINNED_COMMIT = "0b341e582afbf9e1011f24744b554c96a3477eb5"
MODEL = "gpu/model.py"
HEADER = "gpu/bitnet_kernels/bitnet_kernels.h"
UNCHANGED_SOURCES = (
    "gpu/pack_weight.py",
    "gpu/bitnet_kernels/bitnet_kernels.cu",
    "gpu/bitnet_kernels/compile.sh",
)
REPLACEMENTS = {
    MODEL: (
        ("RMSNorm(dim, norm_eps)", "RMSNorm(dim, eps=norm_eps)"),
        ("RMSNorm(hidden_dim, norm_eps)", "RMSNorm(hidden_dim, eps=norm_eps)"),
    ),
    HEADER: (
        (
            "  int ws_idx = out_idx / (N / ws_num);",
            "  int ws_idx;\n"
            "  if constexpr (M == 1 && N == 3840 && K == 2560 && ws_num == 3) {\n"
            "    // BitNet 2B Q/K/V have 2560/640/640 rows and separate scales.\n"
            "    ws_idx = out_idx < 2560 ? 0 : (out_idx < 3200 ? 1 : 2);\n"
            "  } else {\n"
            "    ws_idx = out_idx / (N / ws_num);\n"
            "  }",
        ),
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def require_repo(repo: Path) -> None:
    actual_root = Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if actual_root != repo:
        raise RuntimeError(f"--repo must name the BitNet repository root: {actual_root}")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(f"Expected BitNet {PINNED_COMMIT}, found {head}")


def patch_plan(relative: str, original: bytes, current: bytes) -> dict:
    """Accept only the pinned original or exactly this helper's finished patch."""
    canonical = normalized(original)
    changed = canonical
    for before, after in REPLACEMENTS[relative]:
        old, new = before.encode(), after.encode()
        if changed.count(old) != 1:
            raise RuntimeError(f"Pinned source does not have exactly one expected match: {relative}: {before}")
        changed = changed.replace(old, new, 1)

    current_lf = normalized(current)
    if current_lf == changed:
        state, result = "already_patched", current
    elif current_lf == canonical:
        state, result = "original", current
        # Preserve the working tree's existing line endings and all other bytes.
        for before, after in REPLACEMENTS[relative]:
            old = before.encode()
            newline = b"\r\n" if b"\r\n" in current else b"\n"
            new = after.encode().replace(b"\n", newline)
            if result.count(old) != 1:
                raise RuntimeError(f"Cannot preserve source bytes safely: {relative}")
            result = result.replace(old, new, 1)
        if normalized(result) != changed:
            raise RuntimeError(f"Patched source validation failed: {relative}")
    else:
        raise RuntimeError(f"Unrecognized changes in {relative}; refusing to overwrite them")

    return {
        "path": relative, "state": state, "before": current, "after": result,
        "before_sha256": sha256(current), "after_sha256": sha256(result),
        "pinned_blob_sha256": sha256(original),
    }


def inspect_sources(repo: Path) -> tuple[list[dict], list[dict]]:
    require_repo(repo)
    plans, support = [], []
    for relative in REPLACEMENTS:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Expected a regular source file: {path}")
        plans.append(patch_plan(relative, git(repo, "show", f"HEAD:{relative}"), path.read_bytes()))
    for relative in UNCHANGED_SOURCES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Expected a regular source file: {path}")
        current = path.read_bytes()
        original = git(repo, "show", f"HEAD:{relative}")
        if normalized(current) != normalized(original):
            raise RuntimeError(f"Unrecognized changes in supporting source: {relative}")
        support.append({"path": relative, "sha256": sha256(current)})
    return plans, support


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def diagnostic_float(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def atomic_replace(path: Path, expected: bytes, replacement: bytes) -> None:
    if path.is_symlink() or path.read_bytes() != expected:
        raise RuntimeError(f"File changed since inspection; refusing to overwrite: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        if path.is_symlink() or path.read_bytes() != expected:
            raise RuntimeError(f"File changed while preparing its patch: {path}")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_patches(repo: Path, artifacts: Path, report: dict) -> None:
    plans, support = inspect_sources(repo)
    report["supporting_sources"] = support
    report["files"] = [{k: v for k, v in plan.items() if k not in ("before", "after")} for plan in plans]
    differences = []
    # Prepare every backup and the audit report before changing either source file.
    for plan in plans:
        relative = plan["path"]
        for folder, key in (("before", "before"), ("after", "after")):
            destination = artifacts / folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(plan[key])
        differences.extend(difflib.unified_diff(
            normalized(plan["before"]).decode().splitlines(keepends=True),
            normalized(plan["after"]).decode().splitlines(keepends=True),
            fromfile=f"before/{relative}", tofile=f"after/{relative}",
        ))
    (artifacts / "changes.diff").write_text("".join(differences), encoding="utf-8")
    report["status"] = "prepared"
    save_json(artifacts / "report.json", report)
    for plan, item in zip(plans, report["files"]):
        if plan["state"] == "original":
            atomic_replace(repo / plan["path"], plan["before"], plan["after"])
            item["action"] = "patched"
        else:
            item["action"] = "unchanged"
        save_json(artifacts / "report.json", report)
    after, _ = inspect_sources(repo)
    if any(plan["state"] != "already_patched" for plan in after):
        raise RuntimeError("Source did not retain the expected patches")


def check_gpu(repo: Path, artifacts: Path, report: dict) -> None:
    plans, support = inspect_sources(repo)
    report["source_states"] = {plan["path"]: plan["state"] for plan in plans}
    report["supporting_sources"] = support
    report["source_sha256"] = {plan["path"]: plan["before_sha256"] for plan in plans}
    library_path = repo / "gpu/bitnet_kernels/libbitnet.so"
    if not library_path.is_file():
        raise RuntimeError(f"CUDA library has not been built: {library_path}")
    report["library_sha256"] = sha256(library_path.read_bytes())

    # Imports are deliberately confined to check: patch requires only Python's standard library.
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA device")
    torch.cuda.set_device(0)
    report["torch_version"] = torch.__version__
    report["torch_cuda_version"] = torch.version.cuda
    report["device"] = torch.cuda.get_device_name(0)
    report["compute_capability"] = list(torch.cuda.get_device_capability(0))

    gpu_dir = repo / "gpu"
    os.chdir(gpu_dir)  # model.py loads the CUDA library by a path relative to this directory.
    sys.path.insert(0, str(gpu_dir))
    model = importlib.import_module("model")
    packing = importlib.import_module("pack_weight")
    for module, filename in ((model, "model.py"), (packing, "pack_weight.py")):
        if Path(module.__file__).resolve() != gpu_dir / filename:
            raise RuntimeError(f"Imported an unexpected module: {module.__file__}")

    config = model.ModelArgs()
    # Inspect the actual constructors without allocating an entire model or changing weights.
    with torch.device("meta"):
        attention = model.Attention(
            dim=config.dim, head_dim=config.dim // config.n_heads,
            n_heads=config.n_heads, n_kv_heads=config.n_kv_heads,
            rope_theta=config.rope_theta, norm_eps=config.norm_eps, use_kernel=True,
        )
        feed_forward = model.FeedForward(
            dim=config.dim, hidden_dim=config.ffn_dim,
            norm_eps=config.norm_eps, use_kernel=True,
        )
    observed_eps = {
        "attention.attn_sub_norm": attention.attn_sub_norm.eps,
        "feed_forward.ffn_sub_norm": feed_forward.ffn_sub_norm.eps,
    }
    norm_ok = all(value == config.norm_eps for value in observed_eps.values())
    report["normalization"] = {"expected_eps": config.norm_eps, "observed_eps": observed_eps, "passed": norm_ok}

    library = ctypes.CDLL(str(library_path))
    kernel = library.bitlinear_int8xint2
    kernel.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    kernel.restype = None

    n_rows, width = 3840, 2560
    generator = torch.Generator(device="cpu").manual_seed(42)
    inputs = torch.randint(-16, 17, (1, width), generator=generator, dtype=torch.int8)
    weights = torch.randint(-1, 2, (n_rows, width), generator=generator, dtype=torch.int8)
    # These rows guarantee a nonzero dot product in both regions affected by the old scale lookup.
    for row in (1280, 2559, 2560, 3199):
        weights[row] = inputs[0].sign()
    packed = packing.convert_weight_int8_to_int2(weights)
    if packed.shape != (n_rows, width // 4) or packed.dtype != torch.int8:
        raise RuntimeError("Official packing function returned an unexpected shape or dtype")

    input_scale = 4.0
    qkv_scales = (0.5, 2.0, 8.0)
    integer_reference = np.matmul(inputs.numpy().astype(np.int32), weights.numpy().astype(np.int32).T)
    row_scales = torch.empty(n_rows, dtype=torch.float32)
    row_scales[:2560], row_scales[2560:3200], row_scales[3200:] = qkv_scales
    reference_fp32 = torch.from_numpy(integer_reference).to(torch.float32) / input_scale
    reference = (reference_fp32 * row_scales).to(torch.bfloat16)
    legacy_scales = torch.tensor(qkv_scales, dtype=torch.float32).repeat_interleave(n_rows // 3)
    legacy_reference = (reference_fp32 * legacy_scales).to(torch.bfloat16)
    discriminating_rows = int(torch.count_nonzero(reference != legacy_reference).item())
    if discriminating_rows == 0:
        raise RuntimeError("Test inputs do not distinguish the old and corrected scale lookup")

    with torch.inference_mode():
        inputs_gpu = inputs.to("cuda").contiguous()
        packed_gpu = packed.to("cuda").contiguous()
        input_scale_gpu = torch.tensor([input_scale], dtype=torch.bfloat16, device="cuda")
        weight_scales_gpu = torch.tensor([*qkv_scales, 0.0], dtype=torch.bfloat16, device="cuda")
        output_gpu = torch.full((1, n_rows), float("nan"), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.current_stream()
        kernel(
            inputs_gpu.data_ptr(), packed_gpu.data_ptr(), output_gpu.data_ptr(),
            input_scale_gpu.data_ptr(), weight_scales_gpu.data_ptr(),
            1, n_rows, width, stream.cuda_stream,
        )
        torch.cuda.synchronize()
        actual = output_gpu.cpu()
    mismatch = actual != reference
    bad_rows = torch.nonzero(mismatch[0], as_tuple=False).flatten().tolist()
    kernel_ok = bool(torch.isfinite(actual).all().item()) and not bad_rows
    report["qkv_kernel"] = {
        "shape_m_n_k": [1, n_rows, width], "input_scale": input_scale,
        "qkv_scales": list(qkv_scales), "qkv_row_counts": [2560, 640, 640],
        "reference": "CPU int32 dot product, FP32 scaling, then BF16 rounding",
        "comparison": "exact BF16 equality; scale factors are powers of two",
        "old_scale_formula_mismatching_rows": discriminating_rows,
        "mismatching_rows": len(bad_rows), "passed": kernel_ok,
        "first_mismatches": [
            {"row": row, "actual": diagnostic_float(float(actual[0, row])), "expected": float(reference[0, row])}
            for row in bad_rows[:12]
        ],
    }
    save_json(artifacts / "report.json", report)
    source_ok = all(plan["state"] == "already_patched" for plan in plans)
    if not (source_ok and norm_ok and kernel_ok):
        raise RuntimeError(
            f"Check failed: patched_sources={source_ok}, norm_eps={norm_ok}, "
            f"qkv_kernel={kernel_ok}; rebuild the library after applying the patch"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("patch", "check"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = log_dir / f"bitnet-{args.command}-{stamp}-{uuid4().hex[:8]}"
    artifacts.mkdir(parents=True, exist_ok=False)
    report = {
        "command": args.command, "repo": str(repo), "required_commit": PINNED_COMMIT,
        "started_utc": stamp, "artifacts": str(artifacts), "status": "running",
    }
    try:
        if args.command == "patch":
            apply_patches(repo, artifacts, report)
        else:
            check_gpu(repo, artifacts, report)
        report["status"] = "PASS"
        result = 0
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            report["git_stderr"] = error.stderr.decode(errors="replace") if error.stderr else ""
        traceback.print_exc()
        result = 1
    save_json(artifacts / "report.json", report)
    print(f"{args.command.upper()} {report['status']}: {artifacts / 'report.json'}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
