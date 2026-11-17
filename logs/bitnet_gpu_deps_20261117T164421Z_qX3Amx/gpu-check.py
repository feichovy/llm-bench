import importlib.metadata as md
import json

import torch
import torch.nn.functional as F
from xformers.ops import RMSNorm, fmha, rope_padded

if md.version("torch") != "2.9.1+cu130":
    raise SystemExit("Unexpected PyTorch version.")
if md.version("xformers") != "0.0.33.post2":
    raise SystemExit("Unexpected xFormers version.")
if torch.version.cuda != "13.0" or not torch.cuda.is_available():
    raise SystemExit("CUDA 13.0 PyTorch or GPU access is unavailable.")

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("xFormers:", md.version("xformers"))
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

torch.manual_seed(42)
torch.set_float32_matmul_precision("highest")

with torch.inference_mode():
    a = torch.randn(128, 128, dtype=torch.float32)
    b = torch.randn(128, 128, dtype=torch.float32)
    reference = a @ b
    result = a.to("cuda:0") @ b.to("cuda:0")
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), reference, rtol=1e-4, atol=1e-4)
    print("PASS: PyTorch CUDA matrix multiplication.")

    q, k, v = [
        torch.randn(1, 16, 2, 128, device="cuda:0", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    out = fmha.memory_efficient_attention_forward(
        q, k, v, op=fmha.flash.FwOp
    )
    norm = RMSNorm(128, eps=1e-5).to(device="cuda:0", dtype=torch.bfloat16)
    norm_out = norm(q)
    torch.cuda.synchronize()

    if not out.is_cuda or not norm_out.is_cuda:
        raise SystemExit("Expected CUDA outputs.")
    if not torch.isfinite(out).all().item() or not torch.isfinite(norm_out).all().item():
        raise SystemExit("Non-finite GPU output.")

    qr, kr, vr = [x.float().cpu().transpose(1, 2) for x in (q, k, v)]
    ref = F.scaled_dot_product_attention(
        qr, kr, vr, dropout_p=0.0, is_causal=False
    ).transpose(1, 2)
    actual = out.float().cpu()
    torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)

print("PASS: xFormers BF16 Flash Attention.")
print("PASS: xFormers CUDA RMSNorm execution.")
print(json.dumps({
    "status": "PASS",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "xformers": md.version("xformers"),
    "gpu": torch.cuda.get_device_name(0),
    "attention_max_abs_error": (actual - ref).abs().max().item()
}, indent=2))
