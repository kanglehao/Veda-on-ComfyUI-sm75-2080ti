<div align="center">

<img src="https://raw.githubusercontent.com/veda-sparse/Veda-on-ComfyUI/main/assets/icon.svg" width="120" alt="Veda logo">

# Veda Sparse Attention for ComfyUI — with Turing / RTX 20-series (sm_75) support

**Enables Veda's learned sparse attention on NVIDIA Turing GPUs (RTX 2080 Ti,
GTX 16-series and every other compute-capability 7.5 card), which upstream
does not support.**

This is a fork of [`veda-sparse/Veda-on-ComfyUI`](https://github.com/veda-sparse/Veda-on-ComfyUI).
Upstream ships one Triton kernel that starts at **SM80** (RTX 30-series and
newer), so on a 2x RTX 2080 Ti machine it stops with *"no sparse kernel covers
this device"*. This fork adds a hand-written **CUDA kernel for sm_75**, using
the int8 tensor-core instructions Turing actually has (`mma.m8n8k16`,
`mma.m16n8k8`) instead of the FP16 tensor cores of Ampere.

[Paper](https://arxiv.org/abs/2605.30325) · [Upstream project](https://github.com/veda-sparse/Veda-on-ComfyUI) · [Predictor](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) · [中文说明](README.zh-CN.md)

---

## What this fork adds

1. **A CUDA sparse-attention kernel for `sm_75`** (compute capability 7.5).
   Upstream's Triton path cannot be used there: Triton does not emit int8
   `mma` for Turing, so a rewrite was required.
2. **Both quantisation paths.** The full tensor-core kernel
   (`cuda-int8-mma-sm75`, QK `mma.m8n8k16.s8` + PV `mma.m16n8k8.f16`) and the
   simpler DP4A kernel (`cuda-int8-sm75`) as a fallback.
3. **Cross-platform build flags.** Compiler flags are selected per platform:
   the MSVC-only `/Zc:preprocessor` and `/utf-8` are now passed on Windows
   only, so the kernel also builds on Linux (they previously made `gcc` fail
   with *"cannot specify -o with -c with multiple files"*).

## Measured on 2x RTX 2080 Ti (22 GB)

MiniMax-H3, Turbo 8-step LoRA, identical seed, `VedaSparseAttention` in the
graph vs. the same graph without it:

| Resolution | dense | **with Veda** | speed-up |
|---|---|---|---|
| 512x512, 124 frames (5.2 s) | 63.3 s/step | **34.6 s/step** | **1.83x** |
| 768x768, 124 frames (5.2 s) | 251.6 s/step | **60.5 s/step** | **4.16x** |

Single-GPU numbers, measured on one 2080 Ti. The gain grows with clip length,
exactly as upstream reports for SM80: on a short clip there is little to
skip, on a long one most attention tiles are dropped. Veda's own attention
pass at 768x768 computed 24.1% of full attention (75.9% skipped).

Frame-to-frame quality was checked by eye: same scene, same subject, same
pose and lighting as the dense run; the difference is in fine fur and snow
grain detail.

### A note on multi-GPU

Veda **cannot be combined with ComfyUI's tensor-parallel multi-GPU mode.**
ComfyUI splits the 56 attention heads of MiniMax-H3 evenly across GPUs
(`56 / 2 = 28` per card), while the Veda predictor is trained for the full
head set and validates the head count before it runs. With two GPUs you get:

```
This model has 50 x 28 x 128 (blocks x heads x head dim),
but the predictor was trained for 50 x 56 x 128.
```

Use **one GPU** (`--cuda-device 0`, or `COMFYUI_GPU=0`) with Veda. On a 2x 2080 Ti
box this is not a downgrade: single-GPU + Veda at 768x768 was ~2.4x faster
than dual-GPU dense on the same machine.

> H3VM and other head-splitting multi-GPU nodes on ComfyUI are incompatible
> for the same reason: they take ownership of the attention call the same way
> Veda does.

## Installation

Requires ComfyUI >= 0.37.0 (this fork was validated on 0.37.0 and 0.37.4;
upstream declares >= 0.38.0, but the APIs Veda uses are present earlier).

```bash
git clone https://github.com/<owner>/<this-repo>.git \
    ComfyUI/custom_nodes/veda_sm75
```

or copy the folder into `ComfyUI/custom_nodes/` and restart.

**Linux extra step** — install `ninja` and put a CUDA toolkit matching your
torch build on `PATH` (the kernel is JIT-compiled on first use):

```bash
pip install ninja
export CUDA_HOME=/usr/local/cuda-13.0     # match torch's CUDA version
export PATH="$CUDA_HOME/bin:$PATH"
```

**Windows** — install `ninja` too; ComfyUI's launcher does not put
`python_embeded/Scripts` on `PATH`, and the kernel build looks for the
`ninja` executable there. Everything else is found automatically.

The predictor (275 MB, MiniMax H3 Community License) goes in
`ComfyUI/models/veda/`; the node prints the download URL if it is missing.

## Usage

One node, MODEL in and MODEL out. Put it on the MODEL wire **after** the model
and any LoRA loaders and **before** the guider/sampler. Press **Ctrl+B** to
bypass it and render the same graph without acceleration.

Skip `generated_sparsity` / `reference_sparsity` unless you have a reason;
the defaults are the predictor's trained budget. `verbose` adds per-stage
timings and the percentage of attention actually computed.

Two things worth reading on the node:

- `Attention computed: x% (y% skipped)` — if `skipped` is 0, the clip is too
  short for Veda to help and you are paying the predictor for nothing.
  Bypass it.
- `Tile plan: nearest trained size: ...` — the predictor was trained on
  1344x768 / 768x768 / 1024x768 at 5 / 10 / 14 s, so other sizes and step
  counts work but sit outside the training distribution.

## Hardware

| GPU | Status |
|---|---|
| **RTX 2080 Ti / GTX 16-series / any sm_75 (Turing)** | **Supported by this fork.** int8 tensor-core kernel, verified on 2080 Ti |
| RTX 3090 / 4090 / A100 (sm_80/86/90) | Use upstream's Triton path; this fork's CUDA kernel is compiled for sm_75 only |
| Apple silicon (M-series) | Upstream MLX path |

The `sm_75` kernel is deliberately compiled for Turing alone
(`-gencode arch=compute_75,code=sm_75`), so it does not slow down newer cards
if the node is present; the backend selector picks the right kernel per device.

## Credits and licence

This fork keeps upstream's MIT `LICENSE` and `NOTICE.md`, which credit
SageAttention (BSD-3-Clause, Thu-ML), Miowtion (MIT) and the predictor
weights (MiniMax H3 Community License). Please read `NOTICE.md`; the int8
quantisation arithmetic in `veda_comfy/kernels/sage/sparse_int8.py` is derived
from SageAttention 1.0.6 and is kept unchanged on purpose, because keeping it
identical to what ComfyUI runs behind `--use-sage-attention` is what keeps
the accuracy.

The Turing CUDA kernel (`veda_comfy/kernels/sage_cuda/`) is new in this fork.