"""sm75 张量核版内核的 Python 侧（QK=m8n8k16.s8 / PV=m16n8k8.f16）。

与 `sparse_int8_cuda.py`（DP4A 版）的**唯一差别**是 .cu 源文件与扩展名：
算术、量化、key block 展开全部**直接复用**，不复制一份 —— 换引擎，不换算法。
这样两个内核的正确性可以直接对比（实测 self_test 相对误差都是 2.208%，逐位一致）。

实测（2080Ti，SM 时钟拉到 ~1770 MHz，热身 2 s 后测）：
    一段 864x480x277   12.41 ms  (DP4A 65.66 / sageattn 稠密 37.84)  -> 3.05x
    二段 1216x672x277  32.21 ms  (DP4A 188.53 / sageattn 稠密 168.52) -> 5.23x

⚠️ 上面这组绝对值**只在同一进程内互相比较才有意义**：192.4 W 功耗墙下 SM 时钟
   在 1350~1890 MHz 之间漂，跨会话同一个内核能差 16%（实测 12.41 → 14.45 ms，
   同轮的 DP4A 与 sageattn 也一起慢 10~19%）。跨版本比较请用 `c2_real_size.py`
   （四引擎同进程）。

`sparse_int8_mma.cu` 里有两个 C2 优化开关（`VEDA_VINT` / `VEDA_ACCFAST`），
**实测都失败，默认关**（见内核头注释与 `结论C2_两处优化_2026-10-16.md`）。
所以这个后端跑的仍然是已验证的 C1 内核。
"""

from __future__ import annotations

import functools
import os

import torch

from . import sparse_int8_cuda as _dp4a

# 算术部分逐条复用（同一份代码 ⇒ 两个内核的数值必然同源）。
TILE = _dp4a.TILE
KEY_BLOCK = _dp4a.KEY_BLOCK
LOG2E = _dp4a.LOG2E
quantize = _dp4a.quantize
key_blocks = _dp4a.key_blocks

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, 'src', 'sparse_int8_mma.cu')
_EXT_NAME = 'veda_sparse_int8_mma_sm75'


@functools.cache
def _kernel():
    """编译并返回扩展模块（只在第一次调用时真正编译）。"""
    return _dp4a.build(_SOURCE, _EXT_NAME)


def ensure_built():
    """提前触发编译（有缓存则秒回），供后端在创建阶段就暴露构建问题。

    Raises:
        编译失败时原样抛出（缺 nvcc / 缺 MSVC / 编译报错）。
    """
    return _kernel()


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           index: torch.Tensor, count: torch.Tensor,
           valid_count: torch.Tensor,
           softmax_scale: float | None = None) -> torch.Tensor:
    """tile 序张量上的块稀疏 INT8 注意力（张量核引擎）。

    参数与返回值的语义与 `sparse_int8_cuda.attend` **完全相同**。
    """
    slots, heads, dim = q.shape
    scale = (dim ** -0.5) if softmax_scale is None else softmax_scale
    q_i8, q_scale = quantize(q, TILE, pre_scale=scale * LOG2E)
    k_i8, k_scale = quantize(k, KEY_BLOCK)
    blocks, block_count, valid = key_blocks(index, count, valid_count,
                                            KEY_BLOCK)
    out = _kernel().sparse_int8_attention(
        q_i8, q_scale, k_i8, k_scale, v.to(torch.float16),
        blocks, block_count, valid)
    return out.to(q.dtype)


def available() -> bool:
    """这张卡能不能跑这个核（张量核要 Turing 起步）。"""
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (7, 5))
