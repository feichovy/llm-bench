from pathlib import Path
import hashlib
import json
import sys

from huggingface_hub import hf_hub_download

model_dir, log_dir = map(Path, sys.argv[1:3])
repo_id = "microsoft/bitnet-b1.58-2B-4T-bf16"
revision = "276681394656abdadb8e80e5b2c3db5e5d7fcaff"
expected_sha = "529637ff6dab1f5890767356928693f69ffe61d3b6040a43de9306b37bfd5ae1"
expected_size = 4825679400

files = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
    "LICENSE",
    "model.safetensors",
]

print("Repository:", repo_id)
print("Revision:", revision)
records = {}

for filename in files:
    print("Downloading:", filename, flush=True)
    path = Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(model_dir),
        token=False,
    ))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records[filename] = {
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    print(filename, records[filename], flush=True)

weight = records["model.safetensors"]
if weight["size_bytes"] != expected_size or weight["sha256"] != expected_sha:
    raise SystemExit("FAIL: model.safetensors size or SHA256 mismatch.")

manifest = {
    "repo_id": repo_id,
    "revision": revision,
    "model_directory": str(model_dir),
    "weight_sha256_verified": True,
    "expected_weight_sha256": expected_sha,
    "expected_weight_size_bytes": expected_size,
    "files": records,
}
(log_dir / "download-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("PASS: model.safetensors size and SHA256 verified.")
print("PASS: companion files downloaded; file hashes recorded.")
