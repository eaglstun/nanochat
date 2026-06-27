# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nanochat is a minimal, hackable, full-stack LLM training harness (Karpathy) that runs the entire pipeline on a single GPU node: tokenization → pretraining → SFT → RL → eval → inference → chat UI. The reference target is "GPT-2 capability in ~3 hours on 8×H100 for ~$100." It is deliberately **not** a configurable framework — there are no config objects, model factories, or registry/plugin systems. Keep changes minimal, readable, and forkable. Any architectural change must be *principled enough to work across all model depths*, not just the one you tested.

**This is the `eaglstun/nanochat` fork.** Beyond upstream's 8×H100 focus, it hardens the **Apple Silicon / MPS** path so the full pipeline (train → eval → inference → chat_web) runs end-to-end on a Mac. `origin` = the fork (push here), `upstream` = `karpathy/nanochat` (pull/sync). `master` tracks `origin/master`; sync upstream with `git fetch upstream && git merge upstream/master`. See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) → "Running on Apple Silicon" for the MPS fixes and real M4 Max numbers. Don't open upstream PRs from this fork without explicit say-so.

## The one dial: `--depth`

This is the load-bearing concept of the whole codebase. `--depth` (number of transformer layers) is the only complexity knob a user sets. Everything else is **derived** to stay compute-optimal:

- `model_dim = depth * aspect_ratio` (default aspect_ratio 64), nudged up to a multiple of `head_dim` (see `build_model_meta` in `scripts/base_train.py`).
- `n_head = n_kv_head = model_dim // head_dim`; number of training iterations, learning rate, weight decay, etc. all scale off depth (muP-style; **d12 is the reference scale where hyperparameters are tuned**, then transferred up).

When changing model/training code, do not hardcode anything that assumes a specific depth. GPT-2 grade is ~d24–d26 with current code; d12 (~5 min pretrain) is the fast iteration scale.

## Setup & environment

Uses **uv** for deps. CPU and GPU torch builds are mutually exclusive extras:

```bash
uv sync --extra gpu              # CUDA (A100/H100)
uv sync --extra cpu              # CPU-only / Apple MPS
uv sync --extra gpu --group dev  # adds pytest, matplotlib, transformers, etc.
source .venv/bin/activate
```

## Running scripts

Everything runs as a module (`python -m ...`), single-GPU or distributed. For `torchrun`, script args go **after a `--` separator** (torchrun args before, script args after):

```bash
# distributed (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --run=d12
# single GPU / CPU: just drop torchrun; code auto-switches to gradient accumulation
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --num-iterations=20
```

Args are parsed with plain `argparse` (no nanoGPT-style configurator). The single most common OOM fix is lowering `--device-batch-size` (32 → 16 → 8 → ...).

Full reference pipeline lives in **`runs/speedrun.sh`** (tokenizer → download data → base_train → base_eval → SFT → chat_eval → report). `runs/runcpu.sh` is the tiny CPU/MPS example. Read these to understand the canonical end-to-end flow and the exact flags used per stage.

### Pipeline stages (order matters; later stages load earlier checkpoints)

1. `python -m nanochat.dataset -n <shards>` — download pretraining shards (~250M chars each)
2. `python -m scripts.tok_train` / `tok_eval` — train/evaluate the BPE tokenizer (vocab 2^15 = 32768)
3. `scripts.base_train` → `scripts.base_eval` — pretrain base model; eval CORE score / bits-per-byte
4. `scripts.chat_sft` → `scripts.chat_eval -- -i sft` — supervised finetune on conversations
5. `scripts.chat_rl` — optional RL stage (note: RL does **not** support fp16/GradScaler)
6. `scripts.chat_cli -p "..."` or `python -m scripts.chat_web` — talk to the model

## Tests

```bash
python -m pytest                       # all tests (testpaths = tests/)
python -m pytest tests/test_engine.py -v
python -m pytest -m "not slow"         # skip tests marked @pytest.mark.slow
```

Tests use mock models (e.g. `MockModel`/`MockConfig` in `test_engine.py`) so they run on CPU without a real checkpoint.

## Precision / dtype (important, non-standard)

nanochat does **not** use `torch.amp.autocast`. Precision is one global `COMPUTE_DTYPE` in `nanochat/common.py`, auto-detected (bf16 on SM80+, else fp32; CPU/MPS default fp32). Override with `NANOCHAT_DTYPE=bfloat16|float16|float32`.

Mechanism: master weights stay **fp32** (for optimizer precision); the custom `Linear` layer in `gpt.py` casts weights to the input dtype in `forward`. Embeddings are stored directly in `COMPUTE_DTYPE`. So when you write new layers, follow this pattern rather than reaching for autocast. `float16` auto-enables a `GradScaler` in `base_train.py` (SFT too, but **not** RL).

**`COMPUTE_DTYPE` is the single source of truth for dtype** — don't reintroduce `device.type == "cuda"` dtype branches. In particular `Engine`'s KV cache (`engine.py`) allocates at `COMPUTE_DTYPE`; a mismatch makes attention see mixed dtypes and crash (this was a real bf16-on-MPS/CPU bug, fixed). On MPS, `NANOCHAT_DTYPE=bfloat16` works on recent macOS and saves ~25% memory (training + inference); default stays fp32 for compatibility.

## Running on Apple Silicon (MPS)

`uv sync --extra cpu` then run scripts normally (auto-detects MPS). `runs/runcpu.sh` is the tiny end-to-end demo. Device-specific gotchas, all handled via small helpers in `common.py` — reuse them, don't hardcode `cuda`:

- **`get_sync_fn(device_type)`** — `torch.cuda`/`torch.mps`.synchronize or no-op. Bare `torch.cuda.synchronize()` *raises* off-CUDA, and a no-op silently breaks timing on MPS.
- **`get_max_memory_fn(device_type)`** — peak memory; MPS has no true-peak API so it uses `torch.mps.driver_allocated_memory` (a high-water proxy).
- FA3 is Hopper-only; MPS uses the SDPA fallback (slower, expect a warning). MFU shows 0% on MPS (no peak-FLOPS entry). bf16 *training* speed on MPS is config-dependent; the reliable bf16 win is memory, not throughput.

## Architecture map

- `nanochat/gpt.py` — the GPT `nn.Module`. Notable: rotary embeddings (no positional emb), QK norm, untied embed/lm_head, relu² MLP, no-bias linears, parameter-free rmsnorm, GQA, Flash Attention 3, per-layer sliding-window attention via `GPTConfig.window_pattern` (e.g. `"SSSL"` tiled across layers; last layer always full/`L`), and value embeddings on alternating layers (`has_ve`).
- `nanochat/engine.py` — inference engine with KV cache (`KVCache`, `Engine`).
- `nanochat/optim.py` — Muon + AdamW (`MuonAdamW` single-GPU, `DistMuonAdamW` distributed).
- `nanochat/flash_attention.py` — `flash_attn` wrapper; FA3 on Hopper+, SDPA fallback elsewhere (`HAS_FA3`).
- `nanochat/dataloader.py` — tokenizing distributed data loader (BOS, best-fit packing).
- `nanochat/tokenizer.py` — GPT-4-style BPE wrapper (backed by `rustbpe`/`tiktoken`).
- `nanochat/checkpoint_manager.py` — save/load; `find_largest_model` picks the highest-depth `d<N>` checkpoint dir by default.
- `nanochat/core_eval.py` / `loss_eval.py` — CORE score (DCLM) and bits-per-byte eval.
- `nanochat/execution.py` — sandbox for the model to run Python as a tool.
- `nanochat/report.py` — markdown run reports; `python -m nanochat.report reset|generate` brackets a speedrun.
- `tasks/` — eval/training task definitions (arc, gsm8k, mmlu, humaneval, smoltalk, spellingbee, customjson); `tasks/common.py` has `TaskMixture`/`TaskSequence` for composing them.
- `nanochat/ui.html` — the chat web frontend served by `scripts.chat_web`.

## Artifacts & state

All intermediates (downloaded data, tokenizer, checkpoints, reports) go to **`NANOCHAT_BASE_DIR`** (default `~/.cache/nanochat`), *not* the repo. Checkpoints live under per-phase dirs like `<base_dir>/base_checkpoints/<model_tag>/` (model_tag defaults to `d<depth>`; SFT/RL phases use their own dirs), with `model_*.pt`, `optim_*_rank*.pt`, and `meta_*.json` per step. Pretraining data shards land in `<base_dir>/base_data_climbmix/`.

## Logging

wandb is optional. `--run=dummy` (the default) uses `DummyWandb` and logs nothing. Set a real `--run` name (or `WANDB_RUN=` env for `speedrun.sh`) and `wandb login` first to enable it. Key metrics to watch: `val_bpb`, `core_metric`, `train/mfu`, `train/tok_per_sec`.

## Contributing norm

Disclosure policy: when submitting a PR, declare any parts with substantial LLM contribution that you did not write or do not fully understand.
</content>
</invoke>
