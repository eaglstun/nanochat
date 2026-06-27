"""
Regression tests for KV-cache dtype handling in the inference Engine.

The KV cache must be allocated at the model's COMPUTE_DTYPE. It used to be
hardcoded as "bf16 on cuda, fp32 everywhere else", which broke
NANOCHAT_DTYPE=bfloat16 inference on CPU/MPS: the model emits bf16 q/k/v, but a
forced-fp32 cache made attention see mixed dtypes and raise
"Expected query, key, and value to have the same dtype".

COMPUTE_DTYPE is resolved from NANOCHAT_DTYPE at import time, so each case runs
in a subprocess with the env set. CPU-only (the crash is in core SDPA, not
device-specific), so these run in CI without a GPU.
"""
import os
import sys
import subprocess
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Builds a tiny real GPT, allocates a KV cache of the requested dtype, and runs a
# few cached decode steps -- exactly the dtype interaction Engine.generate relies on.
_WORKER = textwrap.dedent("""
    import sys, torch
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.engine import KVCache
    dev = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=4,
                    n_kv_head=4, n_embd=128, window_pattern="L")
    model = GPT(cfg).to(dev); model.init_weights()
    kv = dict(num_heads=4, head_dim=32, num_layers=2)
    cache_dtype = {"fp32": torch.float32, "compute": COMPUTE_DTYPE}[sys.argv[1]]
    cache = KVCache(batch_size=1, seq_len=32, device=dev, dtype=cache_dtype, **kv)
    ids = torch.tensor([list(range(8))], dtype=torch.long, device=dev)
    with torch.inference_mode():
        for _ in range(3):
            logits = model.forward(ids, kv_cache=cache)
            ids = logits[:, -1:, :].argmax(-1)
    assert cache.k_cache.dtype == cache_dtype
    print("OK")
""")


def _run(cache_kind, dtype_env):
    env = dict(os.environ)
    if dtype_env is not None:
        env["NANOCHAT_DTYPE"] = dtype_env
    else:
        env.pop("NANOCHAT_DTYPE", None)
    return subprocess.run(
        [sys.executable, "-c", _WORKER, cache_kind],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )


def test_kvcache_compute_dtype_supports_bf16_inference():
    """The fix: a KV cache at COMPUTE_DTYPE must support bf16 inference."""
    r = _run("compute", "bfloat16")
    assert r.returncode == 0 and "OK" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_kvcache_forced_fp32_crashes_under_bf16_compute():
    """The bug being fixed: a forced-fp32 cache crashes under bf16 compute."""
    r = _run("fp32", "bfloat16")
    assert r.returncode != 0, "expected a dtype-mismatch crash, but it succeeded"
    assert "same dtype" in r.stderr, f"unexpected error: {r.stderr!r}"


def test_kvcache_default_fp32_inference_unchanged():
    """No override (fp32 compute) keeps working with an fp32 cache."""
    r = _run("compute", None)
    assert r.returncode == 0 and "OK" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"
