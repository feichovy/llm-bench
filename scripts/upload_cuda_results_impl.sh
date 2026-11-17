#!/usr/bin/env bash
# Run inside the benchmark VM. Authentication configured earlier is reused.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0

cd "$HOME/projects/llm-inference-bench"
root=$(pwd -P)
if [ "$(git rev-parse --show-toplevel)" != "$root" ]; then
  printf 'Stopped: this directory is not the benchmark Git repository root.\n' >&2
  exit 1
fi
branch=$(git symbolic-ref --quiet --short HEAD) || {
  printf 'Stopped: the benchmark repository is in detached HEAD state.\n' >&2
  exit 1
}
git_dir=$(git rev-parse --absolute-git-dir)
run_dir=$(mktemp -d "$git_dir/cuda_upload_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
printf 'Upload log directory: %s\n' "$run_dir"

run_logged() {
  local step="$1"
  shift
  printf '[%s] Running...\n' "$step"
  if (
    set -x
    "$@"
  ) > "$run_dir/$step.log" 2>&1; then
    printf '[%s] OK\n' "$step"
  else
    local status=$?
    printf 'FAILED: %s (exit=%s)\nLog: %s/%s.log\n' \
      "$step" "$status" "$run_dir" "$step" >&2
    exit "$status"
  fi
}

run_logged destination python3 - <<'PY'
import subprocess
from urllib.parse import urlsplit

for flag in ([], ["--push"]):
    urls = subprocess.check_output(
        ["git", "remote", "get-url", *flag, "--all", "origin"], text=True
    ).splitlines()
    if len(urls) != 1:
        raise SystemExit("Expected one origin fetch URL and one origin push URL.")
    value = urls[0]
    if value.startswith("git@github.com:"):
        path = value.split(":", 1)[1]
    else:
        parsed = urlsplit(value)
        if parsed.scheme not in ("https", "ssh") or parsed.hostname != "github.com":
            raise SystemExit("Origin must point to the expected GitHub repository.")
        path = parsed.path.lstrip("/")
    if path.rstrip("/").lower().removesuffix(".git") != "feichovy/llm-bench":
        raise SystemExit("Origin does not point to feichovy/llm-bench.")
print("PASS: origin fetch and push destinations verified.")
PY

run_logged fetch git fetch origin
base_ref="refs/remotes/origin/$branch"
if ! git show-ref --verify --quiet "$base_ref"; then
  printf 'Stopped: origin has no matching branch named %s.\n' "$branch" >&2
  exit 1
fi
if ! git merge-base --is-ancestor "$base_ref" HEAD; then
  printf 'Stopped: origin/%s has commits not in local HEAD; existing work is preserved.\n' "$branch" >&2
  exit 1
fi

run_logged ignore-rules python3 - <<'PY'
from pathlib import Path
path = Path(".gitignore")
text = path.read_text() if path.exists() else ""
marker = "# Additional exclusions for CUDA experiments"
extra = """
# Additional exclusions for CUDA experiments
*.pt
*.pth
*.ckpt
*.onnx
*.cubin
*.fatbin
*.ptx
*.whl
.venv*/
venv*/
.cache/
.env
.env.*
*.pem
*.key
"""
if marker not in text:
    path.write_text(text.rstrip() + "\n" + extra)
print("PASS: model, build and environment exclusions retained.")
PY

run_logged audit-before python3 scripts/audit_benchmark_upload.py --base-ref "$base_ref"
run_logged stage git add -- .gitignore configs logs results scripts
run_logged audit-staged python3 scripts/audit_benchmark_upload.py --base-ref "$base_ref"
git diff --cached --stat > "$run_dir/staged-summary.txt"

if git diff --cached --quiet; then
  printf '[commit] No new staged changes; checking any existing local commits.\n'
else
  run_logged commit git commit -m "Record CUDA inference benchmarks and BitNet GPU fixes"
fi

run_logged audit-commits python3 scripts/audit_benchmark_upload.py --base-ref "$base_ref"
run_logged push git push -u origin "$branch"

local_tip=$(git rev-parse HEAD)
remote_tip=$(git ls-remote --heads origin "refs/heads/$branch" | awk '{print $1}')
if [ "$local_tip" != "$remote_tip" ]; then
  printf 'FAILED: local and remote commit hashes do not match.\n' >&2
  exit 1
fi

printf '\nUpload verified.\nRepository: feichovy/llm-bench\nBranch: %s\nCommit: %s\n' \
  "$branch" "$local_tip"
printf 'Upload logs: %s\n' "$run_dir"
