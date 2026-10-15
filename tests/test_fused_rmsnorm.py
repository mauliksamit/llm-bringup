"""Correctness gate for the fused RMSNorm CUDA kernel (kern-style discipline).

Compared against llm_bringup.model.rmsnorm_torch on awkward shapes: hidden
sizes 1, 7 (odd, scalar path), 8, 64, 2047 (prime-ish, not a multiple of the
vector width), 2048 (TinyLlama), 5632; row counts 1, 3, 1024; dtypes
float32, bfloat16, float16; plus very large and very small magnitudes and an
unaligned (offset) input view.

Tolerance, justified: both sides compute the same float32 statistics, cast
the normalized value to the dtype, then multiply by the weight in the dtype.
Only the order of the float32 sum of squares differs, which perturbs the
rsqrt factor by a few float32 ulps; after rounding to bf16/fp16 that is at
most one ulp of the output dtype (2^-7 relative for bf16, 2^-10 for fp16,
doubled because the weight multiply rounds again). float32 keeps ~1e-6.

Skipped when CUDA is unavailable or the extension cannot be built here.
"""
import pytest
import torch

from llm_bringup.model import rmsnorm_torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

REL = {torch.float32: 1e-5, torch.bfloat16: 2 * 2**-7, torch.float16: 2 * 2**-10}


def _ext():
    try:
        from llm_bringup.kernels import fused_rmsnorm

        fused_rmsnorm(torch.ones(1, 8, device="cuda"), torch.ones(8, device="cuda"), 1e-5)
        return fused_rmsnorm
    except Exception as e:  # build errors are reported as a skip with the reason
        pytest.skip(f"fused RMSNorm extension not built: {e}")


def _check(fused, x, w, eps=1e-5):
    ref = rmsnorm_torch(x, w, eps)
    out = fused(x, w, eps)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    err = ((out.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-30)).max().item()
    assert err <= REL[x.dtype], f"rel err {err} > {REL[x.dtype]}"
    return out, ref


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [1, 7, 8, 64, 2047, 2048, 5632])
@pytest.mark.parametrize("rows", [1, 3, 1024])
def test_matches_torch(dtype, hidden, rows):
    fused = _ext()
    g = torch.Generator(device="cuda").manual_seed(1234 + hidden + rows)
    x = torch.randn(rows, hidden, device="cuda", generator=g).to(dtype)
    w = torch.randn(hidden, device="cuda", generator=g).to(dtype)
    _check(fused, x, w)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_magnitudes(dtype, scale):
    fused = _ext()
    x = (torch.randn(16, 2048, device="cuda") * scale).to(dtype)
    w = torch.rand(2048, device="cuda").to(dtype) + 0.5
    _check(fused, x, w)


@cuda
def test_unaligned_view_and_3d_input():
    fused = _ext()
    base = torch.randn(4 * 2048 + 1, device="cuda", dtype=torch.bfloat16)
    x = base[1:].view(4, 2048)  # data pointer offset by 2 bytes -> scalar path
    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    _check(fused, x, w)
    x3 = torch.randn(2, 5, 2048, device="cuda", dtype=torch.bfloat16)
    out, ref = _check(fused, x3, w)
    assert out.shape == (2, 5, 2048)


@cuda
def test_bf16_mostly_bit_identical():
    """Report how often the output is bit-identical to the torch composition (expect nearly always)."""
    fused = _ext()
    x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    same = (fused(x, w, 1e-5) == rmsnorm_torch(x, w, 1e-5)).float().mean().item()
    assert same > 0.99
