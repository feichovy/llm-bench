from collections import Counter
from pathlib import Path
import json
import sys
import torch

out_dir, log_dir = map(Path, sys.argv[1:3])
layer_shapes = {
    "attention.wqkv.weight": (3840, 2560),
    "attention.wo.weight": (2560, 2560),
    "feed_forward.w13.weight": (13824, 2560),
    "feed_forward.w2.weight": (2560, 6912),
    "attention_norm.weight": (2560,),
    "ffn_norm.weight": (2560,),
    "attention.attn_sub_norm.weight": (2560,),
    "feed_forward.ffn_sub_norm.weight": (6912,),
}
dense = {
    "tok_embeddings.weight": ((128256, 2560), torch.bfloat16),
    "output.weight": ((128256, 2560), torch.bfloat16),
    "norm.weight": ((2560,), torch.bfloat16),
}
packed = dict(dense)
scale_counts = {}

for i in range(30):
    for suffix, shape in layer_shapes.items():
        key = f"layers.{i}.{suffix}"
        dense[key] = (shape, torch.bfloat16)
        if len(shape) == 2:
            packed[key] = ((shape[0], shape[1] // 4), torch.int8)
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            packed[scale_key] = ((4,), torch.bfloat16)
            scale_counts[scale_key] = (
                3 if "wqkv" in suffix else 2 if "w13" in suffix else 1
            )
        else:
            packed[key] = (shape, torch.bfloat16)

report = {}
for name, spec in [
    ("model_state.pt", dense),
    ("model_state_fp16.pt", dense),
    ("model_state_int2.pt", packed),
]:
    path = out_dir / name
    state = torch.load(
        path, map_location="cpu", mmap=True, weights_only=True
    )
    if set(state) != set(spec):
        raise RuntimeError(f"{name}: unexpected or missing tensor keys.")
    for key, (shape, dtype) in spec.items():
        tensor = state[key]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise RuntimeError(
                f"{name}: {key}: got {tuple(tensor.shape)}, {tensor.dtype}"
            )
    if name == "model_state_int2.pt":
        for key, count in scale_counts.items():
            scale = state[key]
            if not (
                torch.isfinite(scale).all().item()
                and (scale[:count] > 0).all().item()
                and (scale[count:] == 0).all().item()
            ):
                raise RuntimeError(f"{name}: invalid scale: {key}")
    report[name] = {
        "tensor_count": len(state),
        "dtype_counts": dict(Counter(str(v.dtype) for v in state.values())),
        "size_bytes": path.stat().st_size,
        "structure_check": "PASS",
    }
    print(f"PASS: {name}: {len(state)} tensors; shapes and dtypes verified.")
    del state

(log_dir / "conversion-check.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
