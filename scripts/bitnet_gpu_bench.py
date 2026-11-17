#!/usr/bin/env python3
"""Benchmark the pinned BitNet GPU model with an explicitly separate protocol.

Run one fresh process per CPU thread count. This script loads the already
converted checkpoints; it neither converts weights nor patches upstream code.
It requires the RMSNorm and 2B QKV scale fixes to have been applied and tested.
GPU execution must be validated on the target machine, not by a syntax check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time


BITNET_COMMIT = "0b341e582afbf9e1011f24744b554c96a3477eb5"
SCHEMA_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitnet-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--method-json", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("repetitions", "prompt_tokens", "decode_tokens", "warmup"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if max(args.prompt_tokens, args.decode_tokens) > 4096:
        parser.error("token counts must not exceed the model's 4096-token limit")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def write_json_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def summarize(samples, tokens):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timing samples must be positive and finite")
    speeds = [tokens / value for value in samples]
    return {
        "tokens_per_sample": tokens,
        "sample_seconds": samples,
        "sample_tokens_per_second": speeds,
        "mean_seconds": statistics.mean(samples),
        "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "mean_tokens_per_second": statistics.mean(speeds),
        "stdev_tokens_per_second": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "aggregate_tokens_per_second": tokens * len(samples) / sum(samples),
    }


def method_description(args):
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": "bitnet_gpu_cuda_graph_last_logits_empty_decode_v1",
        "batch_size": 1,
        "prompt_tokens": args.prompt_tokens,
        "decode_steps": args.decode_tokens,
        "decode_initial_context_tokens": 0,
        "decode_kv_lengths": [1, args.decode_tokens],
        "kv_capacity_tokens": max(args.prompt_tokens, args.decode_tokens),
        "kv_dtype": "bfloat16",
        "attention_operator": "xformers.ops.fmha.flash.FwOp",
        "execution": "Separate captured prefill and single-step decode CUDA graphs",
        "prefill": {
            "label": f"prefill{args.prompt_tokens}_last_logits",
            "weights": "Expanded BF16 ternary weights; activation quantize/dequantize",
            "computation": "All transformer blocks on every input token; final RMSNorm and vocabulary head only on the last position",
            "timed": "One CUDA-graph replay followed by device synchronization",
        },
        "decode": {
            "label": f"decode{args.decode_tokens}_from_empty",
            "weights": "Custom packed W2A8 BitNet GEMV kernels; BF16 embedding/norm/head",
            "computation": "Exactly one model forward per fixed input token; no sampling or EOS handling",
            "timed": "Python loop, GPU-to-GPU input-token copies, GPU KV-length fills, CUDA-graph replays and device synchronization after every token",
        },
        "timer": "time.perf_counter_ns; torch.cuda.synchronize immediately before each timed interval, after prefill, and after every decode token",
        "untimed": [
            "checkpoint hashing and loading",
            "torch.compile and CUDA-graph capture",
            "all warmup passes",
            "initial CPU-to-GPU token transfer",
            "KV-cache clearing and initial input setup before every sample",
            "finite-output checks and JSON/log writes",
        ],
        "graph_capture_warmup_forwards_per_shape": 3,
        "untimed_graph_validation_replays_per_shape": 1,
        "complete_warmup_passes_per_phase": args.warmup,
        "repetitions_per_phase": args.repetitions,
        "input": "Seeded CPU torch.Generator produces fixed ordinary token IDs in [0,128000); tensors transferred to GPU before timing",
        "seed": args.seed,
        "sampling": False,
        "stop_at_eos": False,
        "comparison_limits": [
            "This is a separate BitNet GPU runner, not llama-bench output or a claim of identical execution paths.",
            "KV cache is BF16; the earlier llama-bench runs requested F16 KV.",
            "CPU thread count controls host PyTorch work, not the number of CUDA threads.",
            "Decode begins with an empty KV cache and processes fixed inputs; it is not a 128-token-prompt response benchmark.",
            "Prefill uses BF16 matrix multiplication, while decode uses the custom W2A8 GEMV kernel.",
            "The llama.cpp Qwen3 graph may omit the final-layer FFN for prompt positions whose logits are not requested. This wrapper still runs all 30 transformer layers at every prompt position; last-position-only logits do not make the workloads identical.",
        ],
    }


def collect_provenance(args, root, checkpoint_dir):
    commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    if commit != BITNET_COMMIT:
        raise RuntimeError(f"Unexpected BitNet commit: {commit}")
    paths = {
        "benchmark_script": Path(__file__).resolve(),
        "model_source": root / "gpu/model.py",
        "kernel_source": root / "gpu/bitnet_kernels/bitnet_kernels.cu",
        "kernel_header": root / "gpu/bitnet_kernels/bitnet_kernels.h",
        "kernel_library": root / "gpu/bitnet_kernels/libbitnet.so",
        "expanded_checkpoint": checkpoint_dir / "model_state_fp16.pt",
        "packed_checkpoint": checkpoint_dir / "model_state_int2.pt",
    }
    files = {}
    for name, path in paths.items():
        log(f"Hashing {name}")
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "bitnet_commit": commit,
        "submodules": command_output(["git", "-C", str(root), "submodule", "status", "--recursive"]),
        "working_tree_status": command_output(["git", "-C", str(root), "status", "--short"]),
        "files": files,
    }


def run(args):
    root = args.bitnet_root.expanduser().resolve(strict=True)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    output = args.output_json.expanduser().resolve()
    method_output = args.method_json.expanduser().resolve()
    if output == method_output:
        raise ValueError("Result and method JSON paths must differ")
    for path in (output, method_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")

    # Set these before importing torch or its numerical-library dependencies.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    method = method_description(args)
    write_json_exclusive(method_output, method)
    provenance = collect_provenance(args, root, checkpoint_dir)

    import torch
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
    )

    if torch.__version__ != "2.9.1+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected torch 2.9.1+cu130, got {torch.__version__}, CUDA {torch.version.cuda}")
    if importlib.metadata.version("xformers") != "0.0.33.post2":
        raise RuntimeError("Expected xformers 0.0.33.post2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not allowed")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires CUDA BF16 support")

    # Upstream model.py loads its ctypes library using a relative path.
    os.chdir(root / "gpu")
    sys.path.insert(0, str(root / "gpu"))
    model = importlib.import_module("model")
    if Path(model.__file__).resolve() != root / "gpu/model.py":
        raise RuntimeError("Imported model.py from the wrong directory")

    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "xformers": importlib.metadata.version("xformers"),
        "triton": importlib.metadata.version("triton"),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "threads_requested": args.threads,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_DYNAMIC", "MKL_DYNAMIC")},
        "tf32": False,
        "provenance": provenance,
    }

    with torch.inference_mode():
        def load_model(use_kernel, filename):
            config = model.ModelArgs(use_kernel=use_kernel)
            with torch.device(device):
                network = model.Transformer(config).eval()
            state = torch.load(checkpoint_dir / filename, map_location="cpu", mmap=True, weights_only=True)
            network.load_state_dict(state, strict=True)
            del state
            norms = [module.eps for module in network.modules() if isinstance(module, model.RMSNorm)]
            if len(norms) != 121 or any(value != config.norm_eps for value in norms):
                raise RuntimeError("RMSNorm epsilon fix is missing or model architecture differs")
            return network, config

        log("Loading expanded BF16 prefill model")
        prefill_model, prefill_config = load_model(False, "model_state_fp16.pt")
        log("Loading packed W2A8 decode model")
        decode_model, decode_config = load_model(True, "model_state_int2.pt")
        capacity = max(args.prompt_tokens, args.decode_tokens)
        cache = model.make_cache(args=decode_config, length=capacity, device=device, dtype=torch.bfloat16)
        if any(tensor.dtype != torch.bfloat16 or tensor.device != device for pair in cache for tensor in pair):
            raise RuntimeError("Unexpected KV-cache dtype or device")

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        token_ids = torch.randint(0, 128000, (args.prompt_tokens + args.decode_tokens,), generator=generator, dtype=torch.int32, device="cpu")
        input_list = token_ids.tolist()
        metadata["fixed_inputs"] = {
            "prompt_token_ids": input_list[:args.prompt_tokens],
            "decode_token_ids": input_list[args.prompt_tokens:],
            "sha256_int32_little_endian": hashlib.sha256(struct.pack(f"<{len(input_list)}i", *input_list)).hexdigest(),
        }
        prefill_input = token_ids[:args.prompt_tokens].to(device)
        decode_inputs = token_ids[args.prompt_tokens:].to(device)
        decode_input = decode_inputs[:1].clone()

        prefill_bias = AttnBias.from_seqlens(q_seqlen=[args.prompt_tokens], kv_seqlen=[args.prompt_tokens], kv_padding=capacity, device=device)
        # Capture with the largest supported K length in the CPU metadata, while
        # the device tensor carries the actual length at each graph replay.
        decode_bias = AttnBias.from_seqlens(q_seqlen=[1], kv_seqlen=[args.decode_tokens], kv_padding=capacity, device=device)
        decode_bias.k_seqinfo.seqlen.fill_(1)
        for bias in (prefill_bias, decode_bias):
            if bias.q_seqinfo.seqstart.device != device or bias.k_seqinfo.seqlen.device != device:
                raise RuntimeError("Attention-bias tensors must reside on the selected GPU")

        def clear_cache():
            for key, value in cache:
                # The head dimension is an expanded, aliased view. Clear each
                # underlying KV head once, avoiding an overlapping write.
                key.select(3, 0).zero_()
                value.select(3, 0).zero_()

        def prefill_forward():
            # Upstream computes every position's vocabulary logits. Restrict
            # only the final norm/head to the last position for this protocol.
            hidden = prefill_model.tok_embeddings(prefill_input)
            for index, layer in enumerate(prefill_model.layers):
                hidden = layer(hidden, cache[index], prefill_bias)
            return prefill_model.output(prefill_model.norm(hidden[-1:])).float()

        def decode_forward():
            return decode_model.forward_with_attn_bias(decode_input, decode_bias, cache)

        def check_logits(logits, phase):
            if tuple(logits.shape) != (1, prefill_config.vocab_size):
                raise RuntimeError(f"Unexpected {phase} logits shape: {tuple(logits.shape)}")
            if logits.device != device or not torch.isfinite(logits).all().item():
                raise RuntimeError(f"Invalid {phase} GPU logits")

        log("Checking last-position prefill wrapper against upstream full-head forward")
        clear_cache()
        reference_logits = prefill_model.forward_with_attn_bias(prefill_input, prefill_bias, cache)[-1:].clone()
        torch.cuda.synchronize(device)
        check_logits(reference_logits, "prefill full-head reference")
        clear_cache()
        candidate_logits = prefill_forward()
        torch.cuda.synchronize(device)
        check_logits(candidate_logits, "prefill last-head wrapper")
        wrapper_max_abs_error = (candidate_logits - reference_logits).abs().max().item()
        torch.testing.assert_close(candidate_logits, reference_logits, atol=2e-2, rtol=2e-2)
        metadata["prefill_wrapper_validation"] = {
            "reference": "upstream Transformer.forward_with_attn_bias(...)[-1:]",
            "cache_cleared_before_each_forward": True,
            "atol": 2e-2,
            "rtol": 2e-2,
            "max_abs_error": wrapper_max_abs_error,
            "status": "PASS",
        }
        del reference_logits, candidate_logits
        log(f"Prefill wrapper numerical check passed; max abs error={wrapper_max_abs_error:.6g}")

        def capture(forward, name):
            clear_cache()
            side_stream = torch.cuda.Stream(device=device)
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    warmup_logits = forward()
            torch.cuda.current_stream(device).wait_stream(side_stream)
            torch.cuda.synchronize(device)
            check_logits(warmup_logits, f"{name} eager warmup")
            del warmup_logits
            clear_cache()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side_stream, capture_error_mode="thread_local"):
                logits = forward()
            # Capture records work; execute the graph before reading its output.
            clear_cache()
            graph.replay()
            torch.cuda.synchronize(device)
            check_logits(logits, f"{name} graph validation replay")
            log(f"Captured {name} graph")
            return graph, logits

        prefill_graph, prefill_logits = capture(prefill_forward, "prefill")
        decode_graph, decode_logits = capture(decode_forward, "decode")

        def prefill_pass(timed):
            clear_cache()
            torch.cuda.synchronize(device)
            start = time.perf_counter_ns() if timed else None
            prefill_graph.replay()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter_ns() - start) / 1e9 if timed else None
            check_logits(prefill_logits, "prefill")
            return elapsed

        def decode_pass(timed):
            clear_cache()
            decode_bias.k_seqinfo.seqlen.fill_(1)
            decode_input.copy_(decode_inputs[:1])
            torch.cuda.synchronize(device)
            start = time.perf_counter_ns() if timed else None
            for step in range(args.decode_tokens):
                decode_input.copy_(decode_inputs[step:step + 1])
                decode_bias.k_seqinfo.seqlen.fill_(step + 1)
                decode_graph.replay()
                # llama-bench also completes each generation step before the
                # next one. Include this host/device boundary in our timing.
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter_ns() - start) / 1e9 if timed else None
            check_logits(decode_logits, "decode")
            if decode_bias.k_seqinfo.seqlen.item() != args.decode_tokens:
                raise RuntimeError("Decode did not reach the requested context length")
            return elapsed

        log(f"Running {args.warmup} untimed complete warmup passes per phase")
        for _ in range(args.warmup):
            prefill_pass(False)
            decode_pass(False)
        log(f"Measuring {args.repetitions} prefill repetitions")
        prefill_samples = [prefill_pass(True) for _ in range(args.repetitions)]
        log(f"Measuring {args.repetitions} decode repetitions")
        decode_samples = [decode_pass(True) for _ in range(args.repetitions)]

        metadata["model"] = {
            "config": vars(decode_config),
            "strict_checkpoint_load": True,
            "rmsnorm_modules_checked_per_model": 121,
            "rmsnorm_epsilon": decode_config.norm_eps,
            "finite_last_logits_all_passes": True,
            "quality_or_full_numeric_equivalence_test": False,
        }
        metadata["gpu_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_at_end_bytes": torch.cuda.memory_allocated(device),
            "reserved_at_end_bytes": torch.cuda.memory_reserved(device),
        }

    results = [
        {"test": method["prefill"]["label"], **summarize(prefill_samples, args.prompt_tokens)},
        {"test": method["decode"]["label"], **summarize(decode_samples, args.decode_tokens)},
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_file": str(method_output),
        "method_sha256": sha256_file(method_output),
        "method": method,
        "metadata": metadata,
        "results": results,
    }
    write_json_exclusive(output, payload)
    for result in results:
        log(f"PASS {result['test']}: {result['mean_tokens_per_second']:.3f} +/- {result['stdev_tokens_per_second']:.3f} tokens/s")
    log(f"Results saved to {output}")


if __name__ == "__main__":
    run(parse_args())
