"""sm75 手写内核的 Python 侧：量化 + key block 展开 + JIT 构建。

对外接口**刻意与 `kernels/sage/sparse_int8.py` 完全一致**（同名 `attend`、
同样的参数顺序与语义），这样 `backends/cuda_int8.py` 与
`backends/triton_int8.py` 的差别只有一行 import —— 换引擎，不换算法。

算术与 Triton 版逐条对齐：
  * Q 每 128 个 slot 一个 scale（= 每个 query tile 一个），K 每 64 个 slot 一个；
  * softmax 的 scale 折进 Q 的量化（`pre_scale = softmax_scale * log2e`），
    核里于是用 exp2；
  * 取整是 `+0.5*sign(x)` 后向零截断（四舍五入远离零），与 upstream 一致；
  * 全 padding 的块 scale 取 1.0，保持零还是零。

构建：第一次调用时才 JIT 编译（`warmup_note()` 会把这件事告诉用户）。
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import sys

import torch

TILE = 128
KEY_BLOCK = 64
LOG2E = 1.4426950408889634

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, 'src', 'sparse_int8_cuda.cu')
_EXT_NAME = 'veda_sparse_int8_sm75'


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------

# CUDA 13 的 CCCL 拒绝 MSVC 传统预处理器；/utf-8 让源文件里的中文注释能过。
# ⚠️ 但这**两个都是 MSVC 专用旗标** —— 把它们带到 Linux/gcc 下会炸：
#    实测（Ubuntu 服务器 2026-10-17）：
#      nvcc ... -Xcompiler=/Zc:preprocessor -Xcompiler=/utf-8
#      gcc: fatal error: cannot specify '-o' with '-c' with multiple files
#    gcc 不认 `/` 开头的旗标，会把它当输入文件，于是出现"多个文件"的报错。
#    => 旗标必须按平台分开；Linux 下只给通用的 -O3 / --use_fast_math。
_CUDA_FLAGS = ['-O3', '--use_fast_math']
if sys.platform == 'win32':
    _CUDA_FLAGS += ['-Xcompiler=/Zc:preprocessor', '-Xcompiler=/utf-8']

# 主机编译器旗标同理：MSVC 要 /utf-8，gcc/clang 要 -O2。
_HOST_CFLAGS = ['/utf-8', '-O2'] if sys.platform == 'win32' else ['-O2']

_MSVC_ROOTS = (
    r'C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC',
    r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC',
    r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC',
    r'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC',
    r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC',
)
_SDK_ROOT = r'C:\Program Files (x86)\Windows Kits\10'


def _ver(name: str):
    return tuple(int(p) for p in name.split('.') if p.isdigit())


def _ensure_ninja_on_path() -> None:
    """把 python_dir/Scripts 补进 PATH —— 那里装着 ninja。

    torch 的 `cpp_extension` 用 `shutil.which('ninja')`（即 PATH）判断有没有
    构建系统，找不到就直接 `RuntimeError("Ninja is required ...")`，根本不看
    `import ninja` 能不能成。而 ComfyUI 的启动器**不带 Scripts 进 PATH**，
    于是 ninja 明明装在 `sys.prefix/Scripts/ninja.exe` 也照样被判为"没装" ——
    在 ComfyUI 里跑就是张量核编不过、静默退回 DP4A（本机 2026-10-16 实测）。

    只做一件事：PATH 里真没有 ninja、而 Scripts 里真有 ninja.exe 时补上。
    不满足就什么都不做，让原来的错误原样冒上去。
    """
    if shutil.which('ninja'):
        return
    scripts = os.path.join(sys.prefix, 'Scripts')
    if os.path.isfile(os.path.join(scripts, 'ninja.exe')):
        os.environ['PATH'] = scripts + os.pathsep + os.environ.get('PATH', '')
        logging.info('Veda: added %s to PATH so ninja is found', scripts)


def _prepare_toolchain() -> None:
    """在**第一次**调用 `cpp_extension` 之前，把 ninja 与 MSVC 备齐。

    为什么不能"先失败、修好环境、再试一次"：torch 的编译器探测
    `get_compiler_abi_compatibility_and_version` 是 `functools.cache` 的，
    第一次在坏环境（PATH 里没有 `cl`）下失败会被**永久记住**，之后再修 PATH，
    第二次照样 `where cl` 失败。

    本机 2026-10-16 实测的完整事故链：ComfyUI 的启动器 PATH 里既没有
    Scripts（ninja 在那里）也没有 MSVC → 张量核 attempt1 以 "Ninja is
    required" 失败 → fallback 修好 PATH → attempt2 被上面那个缓存毒死 →
    `raise first` 只报出 ninja，看起来像"没装 ninja"，真因（工具链没备齐）
    完全看不见 → 静默退回 DP4A。DP4A 能成功纯属捡了现成：那时环境已被
    mma 的 fallback 修好了。
    """
    _ensure_ninja_on_path()
    if shutil.which('cl') is not None:
        return
    # python_embeded / 普通安装都以 sys.prefix 为根
    embedded = os.path.join(sys.prefix, 'python_embeded')
    _hand_built_env(embedded if os.path.isdir(embedded) else sys.prefix)


def _hand_built_env(python_dir: str) -> None:
    """把 MSVC / Windows SDK 的 INCLUDE / LIB / PATH 拼出来。

    只在正常构建失败后作为退路：有些环境里 distutils 自己去调
    `vcvarsall.bat` 会失败（缺组件、被策略拦），但 `cl.exe` 本身能跑，
    把三条路径拼好就够。找不到任何工具链就静默返回，让正常路径的错误
    原样冒上去（那才是用户该看到的）。
    """
    msvc, best = None, ()
    for root in _MSVC_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            cl = os.path.join(root, name, 'bin', 'Hostx64', 'x64', 'cl.exe')
            if os.path.isfile(cl) and _ver(name) > best:
                best, msvc = _ver(name), os.path.join(root, name)
    if msvc is None or not os.path.isdir(_SDK_ROOT):
        return
    inc_root = os.path.join(_SDK_ROOT, 'Include')
    sdk = max((d for d in os.listdir(inc_root)
               if os.path.isdir(os.path.join(inc_root, d, 'ucrt'))),
              key=_ver, default=None)
    if sdk is None:
        return
    include = [os.path.join(msvc, 'include'),
               os.path.join(inc_root, sdk, 'ucrt'),
               os.path.join(inc_root, sdk, 'shared'),
               os.path.join(inc_root, sdk, 'um'),
               os.path.join(python_dir, 'Include')]
    lib = [os.path.join(msvc, 'lib', 'x64'),
           os.path.join(_SDK_ROOT, 'Lib', sdk, 'ucrt', 'x64'),
           os.path.join(_SDK_ROOT, 'Lib', sdk, 'um', 'x64'),
           os.path.join(python_dir, 'libs')]
    path = [os.path.join(msvc, 'bin', 'Hostx64', 'x64'),
            python_dir, os.path.join(python_dir, 'Scripts')]
    os.environ['INCLUDE'] = os.pathsep.join(include)
    os.environ['LIB'] = os.pathsep.join(lib)
    os.environ['PATH'] = (os.pathsep.join(path) + os.pathsep
                          + os.environ.get('PATH', ''))
    os.environ['DISTUTILS_USE_SDK'] = '1'
    os.environ['MSSdk'] = '1'
    # ⚠️ 绝不能设 CL：torch 探测编译器版本时会裸调 `cl`（无输入文件），
    #    CL 里塞任何 flag 都会让那次探测 exit 2。
    os.environ.pop('CL', None)


def build(source: str = _SOURCE, ext_name: str = _EXT_NAME):
    """JIT 编译并返回扩展模块。

    DP4A 版（sparse_int8_cuda.cu）与张量核版（sparse_int8_mma.cu）**共用这一条
    构建路径** —— 两个内核的差别只在 .cu 与扩展名，编译链必须完全一致，否则
    又在同一个坑里踩两次（`/Zc:preprocessor`、不能设 `CL`、`vcvarsall` 退路）。
    """
    from torch.utils import cpp_extension as ce

    if not os.path.isfile(source):
        raise RuntimeError(f'kernel source missing: {source}')
    _prepare_toolchain()
    # 只为这张卡编：不生成 PTX 兜底，二进制更小、加载更快。
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.pop('CL', None)

    def attempt():
        return ce.load(name=ext_name, sources=[source],
                       extra_cuda_cflags=list(_CUDA_FLAGS),
                       extra_cflags=list(_HOST_CFLAGS), verbose=False)

    try:
        return attempt()
    except Exception as first:                       # noqa: BLE001
        if sys.platform != 'win32':
            raise
        # python_embeded / 普通安装都以 sys.prefix 为根
        embedded = os.path.join(sys.prefix, 'python_embeded')
        _hand_built_env(embedded if os.path.isdir(embedded) else sys.prefix)
        try:
            return attempt()
        except Exception as second:                  # noqa: BLE001
            # 第一次失败通常只是环境没配好（ninja / cl.exe 不在 PATH），第二次
            # 才是真正的编译错误。以前这里只 `raise first`，把第二次吞了 ——
            # 本机在 ComfyUI 里就因此把一次真实的编译失败误读成"缺 ninja"。
            # 错误类型仍旧抛第一次的（`explain_failure` 靠它给提示），但真因
            # 一定写进日志。
            logging.warning('%s: the toolchain fallback did not help either; '
                            'second error was: %s', ext_name, second)
            raise first from None


@functools.cache
def _kernel():
    """编译并返回扩展模块（只在第一次调用时真正编译）。"""
    return build()


# --------------------------------------------------------------------------
# 算术（与 Triton 版逐条对齐）
# --------------------------------------------------------------------------


def ensure_built():
    """提前触发编译（有缓存则秒回），供后端在创建阶段就暴露构建问题。

    Raises:
        编译失败时原样抛出（缺 nvcc / 缺 MSVC / 编译报错）。
    """
    return _kernel()


def quantize(x: torch.Tensor, block: int,
             pre_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, H, D] -> ([N, H, D] int8, [H, N // block] fp32 scales).

    `pre_scale` 在量化前乘进去：调用方借此把 softmax 的 scale 折进 Q。
    """
    slots, heads, dim = x.shape
    nblk = slots // block
    xv = (x.float() * pre_scale).view(nblk, block, heads, dim)
    scale = xv.abs().amax(dim=(1, 3)) / 127.0        # [nblk, H]
    # 全 padding 的块没有 scale；1.0 让零保持零。
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = xv / scale[:, None, :, None]
    q = q + 0.5 * torch.sign(q)                      # 四舍五入远离零
    return (q.to(torch.int8).view(slots, heads, dim).contiguous(),
            scale.t().contiguous())


def key_blocks(index: torch.Tensor, count: torch.Tensor,
               valid_count: torch.Tensor,
               block: int = KEY_BLOCK) -> tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor]:
    """已保留的 tile -> 已保留的 key block，即核真正遍历的单位。

    一个 128 行的 tile 是若干个 key block；在 host 侧展开让核里的循环保持
    平坦，并让「真实行是前缀」这个性质可以按同样的方式切。
    """
    sub = TILE // block
    blocks = (index.to(torch.int32) * sub)[..., None] + torch.arange(
        sub, device=index.device, dtype=torch.int32)
    offsets = torch.arange(sub, device=valid_count.device) * block
    valid = (valid_count[:, None] - offsets).clamp_(0, block)
    return (blocks.flatten(-2).contiguous(), (count * sub).contiguous(),
            valid.flatten().to(torch.int32).contiguous())


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           index: torch.Tensor, count: torch.Tensor,
           valid_count: torch.Tensor,
           softmax_scale: float | None = None) -> torch.Tensor:
    """tile 序张量上的块稀疏 INT8 注意力。

    Args:
        q, k, v: [N, H, D] contiguous，16 位，N 是 TILE 的整数倍。
        index: [H, N // TILE, max_kept] int32，每行保留的 key tile。
        count: [H, N // TILE] int32。
        valid_count: [n_tiles] int32，每 tile 的真实行数（前缀）。
        softmax_scale: 默认 D ** -0.5。

    Returns:
        [N, H, D]，q 的 dtype。
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
    """这张卡能不能跑这个核。"""
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (7, 5))
