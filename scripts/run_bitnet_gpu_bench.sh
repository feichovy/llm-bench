#!/usr/bin/env bash
# Run on the Ubuntu VM after installing the two Python helpers beside this file.
set -euo pipefail

if [ "$(uname -s)" != Linux ]; then
  printf 'Run this script inside the Ubuntu GPU VM.\n' >&2
  exit 1
fi

root="${LLM_BENCH_ROOT:-$HOME/projects/llm-inference-bench}"
cd "$root"
repo="$root/engines/BitNet"
gpu_python="$repo/.venv-gpu/bin/python"
checkpoints="$root/models/BitNet-b1.58-2B-4T-gpu"
cuda_bin="/usr/local/cuda-13.0/bin"
prepare="$root/scripts/bitnet_gpu_prepare_test.py"
bench="$root/scripts/bitnet_gpu_bench.py"

mkdir -p logs results
run_dir=$(mktemp -d "$root/logs/bitnet_gpu_all_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
run_id=${run_dir##*/}
printf 'Logs: %s\n' "$run_dir"

run_logged() {
  local stage="$1"
  shift
  printf '[%s] Running...\n' "$stage"
  if (
    set -x
    "$@"
  ) > "$run_dir/$stage.log" 2>&1; then
    printf '[%s] OK\n' "$stage"
  else
    local status=$?
    printf 'FAILED: %s (exit=%s)\nLog: %s/%s.log\n' \
      "$stage" "$status" "$run_dir" "$stage" >&2
    exit "$status"
  fi
}

run_logged python "$gpu_python" --version
run_logged cuda-compiler "$cuda_bin/nvcc" --version
run_logged gpu nvidia-smi
run_logged checkpoint-dense test -s "$checkpoints/model_state_fp16.pt"
run_logged checkpoint-packed test -s "$checkpoints/model_state_int2.pt"

{
  date -u +%FT%TZ
  hostname
  uname -r
  git -C "$repo" rev-parse HEAD
  /usr/bin/g++ --version
  sha256sum "$prepare" "$bench" \
    "$checkpoints/model_state_fp16.pt" "$checkpoints/model_state_int2.pt"
} > "$run_dir/provenance.log" 2>&1
"$gpu_python" -m pip freeze > "$run_dir/python-packages.txt"

run_logged patch "$gpu_python" -u "$prepare" patch \
  --repo "$repo" --log-dir "$run_dir"

cd "$repo/gpu/bitnet_kernels"
run_logged rebuild env -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS \
  PATH="$cuda_bin:$PATH" NVCC_CCBIN=/usr/bin/g++ \
  bash -eux ./compile.sh

cd "$repo/gpu"
run_logged kernel-check env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$gpu_python" -u "$prepare" check --repo "$repo" --log-dir "$run_dir"

git -C "$repo" diff HEAD -- gpu/model.py gpu/bitnet_kernels/bitnet_kernels.h \
  > "$run_dir/applied.patch"
sha256sum "$repo/gpu/model.py" \
  "$repo/gpu/bitnet_kernels/bitnet_kernels.h" \
  "$repo/gpu/bitnet_kernels/libbitnet.so" > "$run_dir/patched-files.sha256"

for threads in 1 2 4; do
  result="$root/results/bitnet_w2a8_cuda_t${threads}_${run_id}.json"
  run_logged "bench-t$threads" "$gpu_python" -u "$bench" \
    --bitnet-root "$repo" --checkpoint-dir "$checkpoints" \
    --threads "$threads" --repetitions 3 \
    --prompt-tokens 128 --decode-tokens 128 --warmup 3 --seed 42 \
    --output-json "$result" \
    --method-json "$run_dir/method-t$threads.json"
  printf 'Result: %s\n' "$result"
done

printf '\nDONE: BitNet GPU checks and all 1/2/4-thread benchmarks passed.\n'
printf 'Logs and patch record: %s\n' "$run_dir"
printf 'Results: %s/results/bitnet_w2a8_cuda_t*_%s.json\n' "$root" "$run_id"
