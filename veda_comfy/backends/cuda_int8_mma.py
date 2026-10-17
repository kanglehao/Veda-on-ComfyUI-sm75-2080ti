"""Turing (SM75) CUDA backend: hand-written **tensor-core** kernel.

Where this sits
---------------
`triton-int8` covers CUDA from SM80 on. A 2080 Ti is SM75, and on this
toolchain Triton 3.7 emits no tensor-core instruction at all for `cc < 8`
(measured: `mma.sync` count 0, every dot lowered to `fma.rn.f16`), so the
Triton path is not merely gated -- it is structurally unavailable there.

`cuda-int8-sm75` (DP4A) was the first answer to that: same arithmetic, but
both dots on CUDA cores. It is correct, and it is slow -- 62 ms against a
38 ms dense baseline on 一段, because `__dp4a` stalls on instruction issue
and its register footprint caps occupancy at 25%. Two peak microbenchmarks
settled it: DP4A tops out at 60.6 int8 TOPS, `mma.m8n8k16.s8` at 200.5, and
the tensor-core number does not move with occupancy.

This backend is that swap, and nothing else (see `kernels/sage_cuda`):

    QK  mma.m8n8k16.s8   (Turing has int8 tensor cores)
    PV  mma.m16n8k8.f16  (Turing's f16 shape is k8; k16 is SM80+)

Measured on the 1142 geometries (2080 Ti, SM clock warmed to ~1770 MHz):

                一段 864x480x277      二段 1216x672x277
    this kernel        12.41 ms            32.21 ms
    DP4A               65.66 ms           188.53 ms
    sageattn dense     37.84 ms           168.52 ms
    -> vs dense         3.05x               5.23x

The arithmetic is unchanged -- SageAttention's INT8 Q/K with one scale per
block, softmax scale folded into Q's quantisation so the kernel can use
`exp2`, P and V in fp16 -- so the accuracy bar is the same one
`triton-int8` and `cuda-int8-sm75` are held to, and the self-test passes at
the identical 2.208% relative error. Only the engine differs.

Self-contained on purpose (see backends/base.py): imports only this file,
torch, `..core.selection` and its own kernel package.
"""

from __future__ import annotations

import functools

import torch

from . import base
from ..core import selection

# Turing through Ampere-less. Ampere and up use `triton-int8`.
_MIN_CC = (7, 5)


@functools.cache
def _kernel():
    from ..kernels.sage_cuda import sparse_int8_mma  # pylint: disable=import-outside-toplevel
    return sparse_int8_mma


class CudaMmaInt8Backend(base.Backend):
    """SageAttention's INT8 arithmetic on sm75 tensor cores."""

    name = 'cuda-int8-mma-sm75'
    display = 'CUDA INT8 tensor cores (sm75)'
    dtypes = (torch.bfloat16, torch.float16)
    # Identical to cuda-int8-sm75 / triton-int8: INT8 Q/K with one scale per
    # block is the format's own floor (~2.2% pointwise on the padding-heavy
    # self-test problem), not a fault. 5% still catches the real ones.
    tolerance = 0.05

    def attend(self, q, k, v, block_mask, layout):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        with torch.no_grad():
            return _kernel().attend(q, k, v, index, count,
                                    layout.valid_count)

    def warmup_note(self) -> str:
        return ('building the sm75 CUDA tensor-core kernel for this GPU '
                '(first run only, about a minute)／首次运行为本卡编译 CUDA '
                '张量核（约一分钟，只做一次）')

    def explain_failure(self, error: BaseException) -> str | None:
        """Recognise a toolchain failure and say how to fix it.

        Same three failure modes as the DP4A backend -- both are JIT-built
        with nvcc + MSVC through the same code path.
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
            'cuda-int8-mma-sm75 needs a Turing (SM75) or newer NVIDIA GPU／'
            'cuda-int8-mma-sm75 需要 Turing（SM75）或更新的 NVIDIA 显卡')
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
            'The sm75 CUDA tensor-core kernel failed to build. It is '
            'compiled on first use with nvcc and MSVC; see the hint below '
            'and the ComfyUI console for the compiler output.\n'
            f'sm75 的 CUDA 张量核构建失败（{type(error).__name__}: {error}）。'
            '它会在首次使用时用 nvcc + MSVC 现场编译；请见下方提示与 '
            'ComfyUI 控制台里的编译输出。'
        ) from error
    return CudaMmaInt8Backend()
