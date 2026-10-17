"""Turing (SM75) CUDA backend: a hand-written DP4A kernel.

Why this exists
---------------
`triton-int8` covers CUDA from SM80 on. A 2080 Ti is SM75, and on this
toolchain Triton 3.7 emits no tensor-core instruction at all for `cc < 8`
(measured: `mma.sync` count 0, every dot lowered to `fma.rn.f16`), so the
Triton path is not merely gated -- it is structurally unavailable there.
This backend is the replacement: CUDA compiled for `sm_75` alone.

What is shared with `triton-int8` and what is not
-------------------------------------------------
The arithmetic is SageAttention's -- INT8 Q and K with one scale per block,
the softmax scale folded into Q's quantisation so the kernel can use `exp2`,
P and V in fp16 -- so the accuracy bar is ComfyUI's own low-precision
attention, the same bar `triton-int8` is held to. Only the execution engine
differs, and `kernels/sage_cuda` states the details.

The two dots currently go through `__dp4a` (CUDA cores), not tensor cores.
That is the deliberate first milestone: get the whole pipeline correct
before touching MMA layouts, which fail as NaN or as a silent wrong answer
rather than as an exception. Swapping them for `m8n8k16.s8` /
`m16n8k8.f16` is a separate, single-point change.

Self-contained on purpose (see backends/base.py): imports only this file,
torch, `..core.selection` and its own kernel package.
"""

from __future__ import annotations

import functools

import torch

from . import base
from ..core import selection

# Turing..just below Ampere. Ampere and up use `triton-int8`, which is one
# kernel for every card from SM80 on; nothing here tries to replace it.
_MIN_CC = (7, 5)


@functools.cache
def _kernel():
    from ..kernels.sage_cuda import sparse_int8_cuda  # pylint: disable=import-outside-toplevel
    return sparse_int8_cuda


class CudaInt8Backend(base.Backend):
    """SageAttention's INT8 arithmetic, walking only the kept tiles."""

    name = 'cuda-int8-sm75'
    display = 'CUDA INT8 DP4A (sm75)'
    dtypes = (torch.bfloat16, torch.float16)
    # Same as triton-int8: INT8 Q/K with one scale per block is the format's
    # own floor (~1.3% relative dense, ~2.7% pointwise on the padding-heavy
    # self-test problem), not a fault. 5% still catches the real ones.
    tolerance = 0.05

    def attend(self, q, k, v, block_mask, layout):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        with torch.no_grad():
            return _kernel().attend(q, k, v, index, count,
                                    layout.valid_count)

    def warmup_note(self) -> str:
        return ('building the sm75 CUDA kernel for this GPU '
                '(first run only, about a minute)／首次运行为本卡编译 CUDA '
                '核（约一分钟，只做一次）')

    def explain_failure(self, error: BaseException) -> str | None:
        """Recognise a toolchain failure and say how to fix it.

        The kernel is JIT-built with nvcc + MSVC, so the failures a user can
        actually hit are a missing CUDA toolkit, a missing C++ build tools,
        or the CUDA/CCCL preprocessor mismatch. Each has one fix; say it
        rather than making the node blame the GPU.
        """
        text = f'{type(error).__name__}: {error}'
        low = text.lower()
        if 'cuda_home' in low or 'nvcc' in low:
            return ('The sm75 CUDA kernel needs the CUDA toolkit (nvcc) on '
                    'PATH. Install CUDA 12.4 or newer, or point CUDA_HOME at '
                    'it, then restart ComfyUI.\n'
                    'sm75 的 CUDA 核需要 CUDA Toolkit（nvcc）。请安装 CUDA '
                    '12.4+ 或把 CUDA_HOME 指到它，然后重启 ComfyUI。')
        if 'cl.exe' in low or 'msvc' in low or 'vcvars' in low:
            return ('The sm75 CUDA kernel is built with MSVC. Install '
                    '"Desktop development with C++" from Visual Studio Build '
                    'Tools, then restart ComfyUI.\n'
                    'sm75 的 CUDA 核需要 MSVC 编译。请安装 Visual Studio '
                    'Build Tools 的「使用 C++ 的桌面开发」，然后重启 ComfyUI。')
        if 'preprocessor' in low or 'cccl' in low or 'cub/' in low:
            return ('This CUDA toolkit needs the conforming MSVC '
                    'preprocessor. Update Visual Studio 2022 to 17.5+ or '
                    'pass -Xcompiler=/Zc:preprocessor.\n'
                    '该 CUDA 版本需要新版 MSVC 预处理器：请把 VS2022 升到 '
                    '17.5+，或加 -Xcompiler=/Zc:preprocessor。')
        return None


def create(info) -> base.Backend:
    if info.kind != 'cuda' or info.cc is None or info.cc < _MIN_CC:
        raise base.BackendUnavailable(
            'cuda-int8-sm75 needs a Turing (SM75) or newer NVIDIA GPU／'
            'cuda-int8-sm75 需要 Turing（SM75）或更新的 NVIDIA 显卡')
    try:
        _kernel().ensure_built()
    except ImportError as error:
        raise base.MissingDependency(
            f'The sm75 CUDA kernel could not be imported ({error}). '
            'PyTorch with CUDA support is required.\n'
            f'sm75 的 CUDA 核导入失败（{error}），需要带 CUDA 的 PyTorch。'
        ) from error
    except Exception as error:  # build failures: nvcc / MSVC / flags
        raise base.MissingDependency(
            'The sm75 CUDA kernel failed to build. It is compiled on first '
            'use with nvcc and MSVC; see the hint below and the ComfyUI '
            'console for the compiler output.\n'
            f'sm75 的 CUDA 核构建失败（{type(error).__name__}: {error}）。'
            '它会在首次使用时用 nvcc + MSVC 现场编译；请见下方提示与 '
            'ComfyUI 控制台里的编译输出。'
        ) from error
    return CudaInt8Backend()
