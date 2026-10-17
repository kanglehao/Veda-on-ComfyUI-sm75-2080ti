<div align="center">

<img src="https://raw.githubusercontent.com/veda-sparse/Veda-on-ComfyUI/main/assets/icon.svg" width="120" alt="Veda logo">

# Veda 稀疏注意力 ComfyUI 节点 —— 新增 Turing / RTX 20 系（sm_75）支持

**让 Veda 的学习型稀疏注意力能在英伟达 Turing 显卡上跑（RTX 2080 Ti、GTX 16 系，
以及所有算力 7.5 的卡），这是上游不支持的。**

本仓库是 [`veda-sparse/Veda-on-ComfyUI`](https://github.com/veda-sparse/Veda-on-ComfyUI)
的分支。上游只发布了一个**从 SM80（RTX 30 系）起**的 Triton 核，所以在本机的
2×2080Ti 上它会直接报"没有覆盖这个设备的稀疏 kernel"。本分支新增了一个为 **sm_75**
手写的 **CUDA 核**，用 Turing 真正拥有的 int8 张量核指令
（`mma.m8n8k16`、`mma.m16n8k8`），而不是 Ampere 的 FP16 张量核。

[论文](https://arxiv.org/abs/2605.30325) · [上游项目](https://github.com/veda-sparse/Veda-on-ComfyUI) · [打分器](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) · [English](README.md)

---

## 本分支新增了什么

1. **面向 `sm_75`（算力 7.5）的 CUDA 稀疏注意力核**。
   上游的 Triton 路径在这里用不了：Triton 不会为 Turing 生成 int8 `mma`，必须重写。
2. **两条量化路径都保留**：完整张量核（`cuda-int8-mma-sm75`，QK 用
   `mma.m8n8k16.s8` + PV 用 `mma.m16n8k8.f16`），以及作为退路的 DP4A 核
   （`cuda-int8-sm75`）。
3. **编译旗标按平台区分**：MSVC 专用的 `/Zc:preprocessor`、`/utf-8` 现在
   只在 Windows 传，因此 Linux 也能编译（此前它们会让 `gcc` 报
   *"cannot specify -o with -c with multiple files"*）。

## 2×2080Ti（22 GB）上的实测

MiniMax-H3 + Turbo 8 步 LoRA，同 seed，图里放 / 不放 `VedaSparseAttention` 对照：

| 分辨率 | dense | **开 Veda** | 加速 |
|---|---|---|---|
| 512×512，124 帧（5.2 s） | 63.3 s/step | **34.6 s/step** | **1.83×** |
| 768×768，124 帧（5.2 s） | 251.6 s/step | **60.5 s/step** | **4.16×** |

以上是**单卡**数字（只用一张 2080 Ti）。收益随片段长度增长，和上游在 SM80 上
报告的一致：短片没什么可跳，长片则大部分 attention tile 被跳过。
Veda 自己的 attention 在 768×768 上只算了全量的 24.1%（跳过 75.9%）。

画质用抽帧肉眼核过：同一场景、同一主体、同一姿势与光照，差别在毛发细节和雪地颗粒。

### 关于多卡

Veda **不能和 ComfyUI 的 tensor-parallel 多卡模式同时用**。
ComfyUI 会把 MiniMax-H3 的 56 个 attention head 平均分到各张卡
（每卡 `56 / 2 = 28`），而 Veda 的打分器是按完整 head 集训练的，运行前会校验 head 数。
双卡时会得到：

```
This model has 50 x 28 x 128 (blocks x heads x head dim),
but the predictor was trained for 50 x 56 x 128.
```

用 Veda 时请**只跑一张卡**（`--cuda-device 0`，或 `COMFYUI_GPU=0`）。
在 2×2080Ti 上这不是降级：同一台机器上，768×768 的「单卡 + Veda」
比「双卡 + dense」还快约 2.4 倍。

> H3VM 及其它会拆 head 的多卡节点同样不兼容，原因是同一个：
> 它们和 Veda 一样要独占那次 attention 调用。

## 安装

需要 ComfyUI >= 0.37.0（本分支在 0.37.0 与 0.37.4 上验证过；
上游声明 >= 0.38.0，但 Veda 用到的 API 在更早版本就已存在）。

```bash
git clone https://github.com/<owner>/<this-repo>.git \
    ComfyUI/custom_nodes/veda_sm75
```

或直接把文件夹复制到 `ComfyUI/custom_nodes/` 后重启。

**Linux 额外一步** —— 装 `ninja`，并把与 torch 构建版本一致的 CUDA toolkit
放进 `PATH`（内核在首次使用时现场编译）：

```bash
pip install ninja
export CUDA_HOME=/usr/local/cuda-13.0     # 与 torch 的 CUDA 版本一致
export PATH="$CUDA_HOME/bin:$PATH"
```

**Windows** —— 同样要装 `ninja`；ComfyUI 启动器不会把
`python_embeded/Scripts` 放进 `PATH`，而内核构建会在那里找 `ninja` 可执行文件。
其余路径会自动找到。

打分器（275 MB，MiniMax H3 社区许可）放在 `ComfyUI/models/veda/`；
缺失时节点会打印下载地址。

## 使用

单个节点，MODEL 进、MODEL 出。接在 MODEL 线上，**在模型与 LoRA 加载之后、
guider / 采样器之前**。按 **Ctrl+B** 可旁路掉它，用同一张图渲染不做加速。

`generated_sparsity` / `reference_sparsity` 没有特别理由就别改，默认就是打分器的
训练预算。`verbose` 会加上各阶段耗时与实际计算的 attention 比例。

节点上有两行值得看：

- `Attention computed: x% (y% skipped)` —— 若 `skipped` 是 0，说明片段太短、
  Veda 帮不上忙，你只是在白付 predictor 的开销。直接旁路。
- `Tile plan: nearest trained size: ...` —— 打分器训练于
  1344×768 / 768×768 / 1024×768、5 / 10 / 14 秒，其它尺寸与步数也能跑，
  但不在训练分布内。

## 硬件

| 显卡 | 状态 |
|---|---|
| **RTX 2080 Ti / GTX 16 系 / 任意 sm_75（Turing）** | **本分支支持。** int8 张量核，已在 2080 Ti 上验证 |
| RTX 3090 / 4090 / A100（sm_80/86/90） | 请用上游的 Triton 路径；本分支的 CUDA 核只针对 sm_75 编译 |
| Apple silicon（M 系） | 上游的 MLX 路径 |

`sm_75` 核刻意**只为 Turing 编译**（`-gencode arch=compute_75,code=sm_75`），
所以装了本节点也不会拖慢更新的卡；后端选择器会按设备挑对应的核。

## 致谢与许可

本分支保留上游的 MIT `LICENSE` 与 `NOTICE.md`，其中声明了
SageAttention（BSD-3-Clause, Thu-ML）、Miowtion（MIT）以及打分器权重
（MiniMax H3 社区许可）。请读 `NOTICE.md`：
`veda_comfy/kernels/sage/sparse_int8.py` 里的 int8 量化算术派生自
SageAttention 1.0.6，**刻意保持不改**，因为与 ComfyUI 在
`--use-sage-attention` 下跑的算术完全一致，正是精度得以保持的原因。

Turing 的 CUDA 核（`veda_comfy/kernels/sage_cuda/`）是本分支新增的。