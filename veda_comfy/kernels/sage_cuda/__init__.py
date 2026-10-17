"""sm75 (Turing) CUDA kernels for Veda, generated with `-gencode arch=compute_75`.

Triton cannot be used here: on this machine's toolchain Triton 3.7 emits no
tensor-core instruction at all for `cc < 8` (see
`docs/…`/the workspace report `结论A_保留比与三道闸`), so upstream's
`triton-int8` backend has no reachable path on a 2080 Ti. This package
holds the hand-written replacement.

Importing it does **not** build anything; the JIT build happens on the
first `attend()` call, which is where `warmup_note()` points the user.
"""

from . import sparse_int8_cuda  # noqa: F401
