"""Backend registry: which kernel runs on which device.

    CUDA sm80 and up    triton-int8
    CUDA sm75 (Turing)  cuda-int8-mma-sm75, then cuda-int8-sm75  [VEDA-SM75]
    ROCm gfx11, gfx12   triton-int8
    Apple MPS           mlx
    anything else       nothing (the node stops with an error)

One kernel per device family, no fallback chain and nothing for the user
to pick. Earlier there were five: four FA4 CuTe backends (one per SM
family, plus an FP8 variant) behind a FlexAttention and a plain-torch
path. `triton-int8` replaced all of them because it is better on every
axis that mattered - 1.50x the FA4 kernel's speed on the same sparse
problem, the same accuracy as ComfyUI's own low-precision attention, and
SM80-and-up coverage from a single Triton kernel - and because a chain of
fallbacks silently hands a user a slower or less accurate kernel than the
one they think they are running. See docs/features/int8_kernel.md.

Backend modules are imported lazily and only here, so a missing Triton
is reported (with the install command) instead of breaking the import of
the node itself.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
from collections.abc import Callable

import torch

from . import base
from .. import hardware

_MODULES = {'triton-int8': 'triton_int8', 'mlx': 'mlx_gather',
            # [VEDA-SM75] Turing 没有走 Triton 的路：本机实测 Triton 3.7 在
            # cc < 8 上不发任何张量核指令（mma.sync = 0），而 upstream 的闸门
            # 就是 (8, 0)。2080 Ti 是唯一没有 Triton 路径的 NVIDIA 家族，
            # 这两个后端就是它的路径 —— 单独为 sm_75 编的手写 CUDA 核。
            #   cuda-int8-mma-sm75  张量核（m8n8k16.s8 / m16n8k8.f16）—— 首选
            #   cuda-int8-sm75      DP4A 兜底 —— 同一算术，只在首选编不过时用
            'cuda-int8-mma-sm75': 'cuda_int8_mma',
            'cuda-int8-sm75': 'cuda_int8'}
# Triton covers CUDA from SM80 on.
_MIN_CC = (8, 0)
# Turing: hand-written CUDA instead (see the note above).
_TURING_CC = (7, 5)
# RDNA3 and RDNA4, where Triton lowers the kernel's int8 dot to WMMA.
_ROCM_ARCHS = ('gfx11', 'gfx12')


def _load(name: str, info: hardware.DeviceInfo) -> base.Backend:
    module = importlib.import_module(f'.{_MODULES[name]}', __name__)
    return module.create(info)


@dataclasses.dataclass
class Resolution:
    """Outcome of picking a backend for one device.

    Attributes:
        device: The device info.
        backend: The backend to use, None if nothing works (run dense).
        attempts: (candidate, 'ok' or why it was skipped), in order.
        hints: What the user can do about the failures, from the backends
            that recognised their own (`Backend.explain_failure`).
        error: The last failure itself, for the traceback in an issue
            report; None when nothing failed.
    """

    device: hardware.DeviceInfo
    backend: base.Backend | None
    attempts: list[tuple[str, str]]
    hints: list[str] = dataclasses.field(default_factory=list)
    error: BaseException | None = None

    @property
    def has_candidate(self) -> bool:
        """Whether a kernel covers this device at all. False means the
        hardware is out of scope; True with `backend` None means the
        kernel that covers it did not start here."""
        return bool(self.attempts)

    def report(self) -> str:
        return '; '.join(f'{name}: {status}' for name, status in self.attempts)


def candidates(info: hardware.DeviceInfo) -> list[str]:
    """The backend to try on a device; empty if none applies."""
    if info.kind == 'cuda' and info.subtype == 'rocm':
        return ['triton-int8'] if info.family.startswith(_ROCM_ARCHS) else []
    if info.kind == 'cuda' and info.cc:
        if info.cc >= _MIN_CC:
            return ['triton-int8']
        # [VEDA-SM75] Turing: Triton emits no tensor-core instruction below
        # cc 8, so the hand-written sm75 kernels are the only candidates.
        # Two of them, and this is the one place a second entry is right:
        # both are the same arithmetic at the same tolerance, so the fallback
        # cannot silently hand the user a less accurate answer the way the
        # old FA4 chain could -- it can only hand them a slower one, and
        # only if the tensor-core source fails to build on their toolchain.
        if info.cc >= _TURING_CC:
            return ['cuda-int8-mma-sm75', 'cuda-int8-sm75']
        return []
    if info.kind == 'mps':
        return ['mlx']
    return []


def _add_hint(hints: list[str], backend: base.Backend | None,
              error: BaseException) -> None:
    """Appends the backend's advice for `error`, if it has any."""
    if backend is None:
        return
    try:
        hint = backend.explain_failure(error)
    except Exception:  # advice must never be the thing that breaks
        logging.warning('Veda: %s could not explain a failure',
                        backend.name, exc_info=True)
        return
    if hint and hint not in hints:
        hints.append(hint)


_LOCK = threading.RLock()
_RESOLVED: dict[str, Resolution] = {}


def resolve(device: torch.device,
            notify: Callable[[str], None] | None = None) -> Resolution:
    """Loads and self-tests the backend for `device` (cached).

    Args:
        device: The device attention runs on.
        notify: Called with a short status line before slow steps (kernel
            compilation on the first call).
    """
    info = hardware.describe(device)
    key = str(device)
    with _LOCK:
        if key in _RESOLVED:
            return _RESOLVED[key]
        attempts: list[tuple[str, str]] = []
        hints: list[str] = []
        chosen = failure = None
        for name in candidates(info):
            backend = None
            try:
                backend = _load(name, info)
                note = backend.warmup_note()
                if notify is not None and note:
                    notify(f'{backend.display}: {note}')
                base.self_test(backend, device)
            except base.BackendUnavailable as error:
                attempts.append((name, str(error)))
                _add_hint(hints, backend, error)
                failure = error
                continue
            except Exception as error:  # reported by the caller, with context
                logging.warning('Veda: backend %s failed to load', name,
                                exc_info=True)
                attempts.append((name, f'{type(error).__name__}: {error}'))
                _add_hint(hints, backend, error)
                failure = error
                continue
            attempts.append((backend.name, 'ok'))
            chosen = backend
            break
        resolution = Resolution(info, chosen, attempts, hints,
                                None if chosen else failure)
        _RESOLVED[key] = resolution
        return resolution


def reset() -> None:
    """Forgets resolutions (tests, or after installing kernels)."""
    with _LOCK:
        _RESOLVED.clear()
