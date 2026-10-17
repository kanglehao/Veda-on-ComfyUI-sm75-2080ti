// Veda 块稀疏 INT8 注意力：sm75（Turing）手写 CUDA 核 —— DP4A 兜底版（C3）。
//
// 为什么存在这个文件
// ------------------
// 上游 Veda 的核是 Triton 写的，闸门是 `cc >= (8, 0)`。本机是 2x2080Ti（cc 7.5），
// 且实测 Triton 3.7 在 sm75 上**任何形状/精度都不发张量核指令**（mma.sync = 0），
// 所以 Triton 那条路结构性走不通。
//
// 这个核**不做任何新算法**，它把 `kernels/sage/sparse_int8.py`（Triton 版）的
// 算术逐条照搬，只换执行引擎：
//
//   * INT8 Q / K，每 128(query tile) / 64(key block) 个 slot 一个 scale —— 与
//     upstream SageAttention 一致；softmax 的 scale 折进 Q 的量化
//     （pre_scale = softmax_scale * log2e），于是核里能直接用 exp2。
//   * 只走 `index` 里列出的已保留 key block（Veda 的全部意义）。
//   * padding 行由 `valid_count` 屏蔽（借用「真实行是每块前缀」这个不变量）。
//   * online softmax，与 Triton 版同一套公式。
//   * P 转 fp16 后与 V 相乘，累加在 fp32（Triton 版 out_dtype=fp16，这里更准）。
//
// C3 用 **__dp4a**（CUDA core 上 4 个 int8 MAC/指令），**不用张量核**。这是刻意的：
// 先把整条流水线的正确性立起来，C2 再把两个 dot 换成 `m8n8k16.s8` / `m16n8k8.f16`。
//
// 版本史（都是实测，不是推测）
// ----------------------------
// v1  256 线程（2 线程/行，acc[64]，qk[64]，内层全展开）
//     ⇒ REG:255 STACK:0 —— 寄存器顶格。256x255 = 65280，每 SM 只塞得下 1 个 block
//       （occupancy 8 warps / 64 = 12.5%）。一段 65.00 ms（sageattn 稠密 35.87 ms，慢 1.81x）。
//       **稠密对照**证明稀疏逻辑本身是对的：稠密 284.77 ms → 稀疏 19.8% 65.00 ms
//       = 4.38x，理论 5.05x ⇒ 达到 87%。所以瓶颈不在稀疏，在执行效率。
// v2  = v1 + __launch_bounds__(256, 2)
//     ⇒ REG:128 **STACK:632** —— 硬压出 632 B/线程栈溢出，一段反而 105.18 ms。
//       证明 v1 的 255 是「真实需求」，只能结构性削减。
// v3  512 线程（4 线程/行，acc[32]）+ qk 分 4 子块（qk[16]）
//     ⇒ REG:87，但内层用了 `#pragma unroll 4` ⇒ 一段 **146.28 ms**（更慢）。
// v3' v3 但内层改回全展开 `#pragma unroll`
//     ⇒ REG:128 STACK:24，一段 **64.95 ms**，与 v1 持平。
//       **结论：`#pragma unroll 4` 砍掉指令级并行是灾难（2.25x）。dp4a 必须全展开。**
//
// 当前实测：一段 64.95 ms vs sageattn 稠密 33.68 ms（0.52x）。
// 但 1.95e10 dp4a / 64.95 ms = 3.0e11 dp4a/s，而 2080Ti 峰值约 6.7e12 dp4a/s ⇒ **只有 4.5%**。
// 这个效率缺口才是真正要查的东西，不是 occupancy（v3' 已把 occupancy 翻倍到 25%，
// 时间却没动，说明卡在别处 —— 最大嫌疑是 sK 的共享内存加载吞吐：每 1 个 dp4a 配 1 个 LDS）。
// `VEDA_VEC_K=1` 就是为验证这个假设准备的：用 int4 把 LDS 指令数降 4 倍。
//
// 可调宏
// ------
//   VEDA_THREADS  512（默认，4 线程/行）或 256（2 线程/行）
//   VEDA_SUB      QK 子块大小，默认 16；=64 即「不分块」（回到 v1 的 qk[64]）
//   VEDA_VEC_K    1 = QK 用 int4 向量化读 sK（LDS 指令数 /4）；0 = 标量读
//
// QK 子块（SUB）
// --------------
//   一个 KEY_BLOCK 的 64 个 key 分 SUB 个一组处理，每组算完立刻做 online-softmax
//   更新 + PV 累加。acc 因此每子块重标定一次，数学上与整块处理完全等价。
//   ⚠️ 子块循环**必须全展开**，否则 qk[t] 索引变动态、数组掉到 local memory。
//   内层 dp4a 也必须全展开（见 v3 的教训）。
//
// 共享内存：sK 8 KB + sV 16 KB = 24 KB（静态）。Q 不进共享内存 ——
//   读 sQ[row][w] 时同 warp 的 32 个线程地址差是 row*128，而 128 B 步长 ≡ 0 mod 128 B，
//   即全落在同一个 bank（32 路冲突）。Q 打进寄存器反而免费。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifndef VEDA_THREADS
#define VEDA_THREADS 512
#endif
#ifndef VEDA_SUB
#define VEDA_SUB 16
#endif
#ifndef VEDA_VEC_K
#define VEDA_VEC_K 0
#endif
#ifndef VEDA_NO_LDS
#define VEDA_NO_LDS 0
#endif

namespace {

constexpr int TILE = 128;        // 一个 Veda tile 的行数
constexpr int KEY_BLOCK = 64;    // 一个 key block 的 slot 数（upstream SageAttention 的粒度）
constexpr int HEAD_DIM = 128;    // D
constexpr int THREADS = VEDA_THREADS;
constexpr int SEGS = THREADS / TILE;         // 每条 query 行几个线程
constexpr int SEG_DIM = HEAD_DIM / SEGS;     // 每线程负责的输出维度数
constexpr int Q_WORDS = HEAD_DIM / 4;        // q 行打包成 int32 的字数 = 32
constexpr int SUB = VEDA_SUB;                // QK 子块大小
constexpr int SUB_ITERS = KEY_BLOCK / SUB;

// ⚠️ 下面这些消息必须是纯 ASCII。nvcc 的前端在解析**字符串字面量**时并不吃
//    -Xcompiler=/utf-8（那只是给 host 编译器的）—— 中文注释没事（预处理阶段就没了），
//    但中文字面量会直接报 "missing closing quote"，而且 torch 的 ninja 还会因为
//    中文 Windows 的 oem 代码页把这行错误信息也吞掉。别用非 ASCII 字面量。
static_assert(THREADS % TILE == 0, "THREADS must be a multiple of TILE");
static_assert(HEAD_DIM % SEGS == 0, "head dim must divide evenly by SEGS");
static_assert(KEY_BLOCK % SUB == 0, "KEY_BLOCK must be a multiple of SUB");
static_assert(512 % THREADS == 0, "K load: 512 int4 must divide by THREADS");
static_assert(1024 % THREADS == 0, "V load: 1024 int4 must divide by THREADS");

__global__ __launch_bounds__(THREADS, 1)
void sparse_int8_attention_kernel(
    const int8_t* __restrict__ Q,      // [N, H, D] int8，tile 序
    const float* __restrict__ QScale,  // [H, n_qtiles] fp32
    const int8_t* __restrict__ K,      // [N, H, D] int8
    const float* __restrict__ KScale,  // [H, n_kblocks] fp32
    const __half* __restrict__ V,      // [N, H, D] fp16
    const int32_t* __restrict__ Index, // [H, n_qtiles, max_kept] key block id
    const int32_t* __restrict__ Count, // [H, n_qtiles]
    const int32_t* __restrict__ Valid, // [n_kblocks] 每块真实行数（前缀）
    __half* __restrict__ Out,          // [N, H, D] fp16
    const int heads,
    const int n_qtiles,
    const int n_kblocks,
    const int stride_ih,               // Index 的 head 步长
    const int stride_iq)               // Index 的 query-tile 步长
{
  const int qtile = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int row = tid & (TILE - 1);       // 0..127
  const int seg = tid >> 7;               // 0..SEGS-1

  __shared__ int8_t sK[KEY_BLOCK][HEAD_DIM];
  __shared__ __half sV[KEY_BLOCK][HEAD_DIM];

  // ---- 本行 Q 的 int8 打进寄存器（32 个 int32）------------------------------
  const int8_t* q_row = Q + ((size_t)(qtile * TILE + row)) * heads * HEAD_DIM
                        + (size_t)head * HEAD_DIM;
  int q_words[Q_WORDS];
#pragma unroll
  for (int i = 0; i < Q_WORDS / 4; i++) {
    const int4 v = reinterpret_cast<const int4*>(q_row)[i];
    q_words[i * 4 + 0] = v.x;
    q_words[i * 4 + 1] = v.y;
    q_words[i * 4 + 2] = v.z;
    q_words[i * 4 + 3] = v.w;
  }
  const float q_scale = __ldg(QScale + (size_t)head * n_qtiles + qtile);

  float m_i = -1e30f;
  float l_i = 0.0f;
  float acc[SEG_DIM];
#pragma unroll
  for (int d = 0; d < SEG_DIM; d++) acc[d] = 0.0f;

  const int count = __ldg(Count + (size_t)head * n_qtiles + qtile);
  const int32_t* idx_row = Index + (size_t)head * stride_ih
                           + (size_t)qtile * stride_iq;

  const size_t slot_stride = (size_t)heads * HEAD_DIM;   // q/k/v 的 slot 步长
  const int8_t* k_base = K + (size_t)head * HEAD_DIM;
  const __half* v_base = V + (size_t)head * HEAD_DIM;
  const int d0 = seg * SEG_DIM;          // 本线程负责的输出段起点

  for (int i = 0; i < count; i++) {
    const int kblk = __ldg(idx_row + i);
    const int start = kblk * KEY_BLOCK;

    // ---- 协作载入 K / V 块 ------------------------------------------------
    {
      // ⚠️ 不能当成连续块拷：q/k/v 是 [slot, head, dim]，一个 head 只占 128 B（K）
      //    / 256 B（V），slot 之间步长是 heads*128 —— 直接整块读会读进别的 head。
      //    L -> (slot s, 段内 int4 偏移 w)：K 每 slot 8 个 int4，s = L>>3, w = L&7。
      //    L = j*THREADS + tid 让同 warp 内 L 连续，全局读取是合并的。
      const int4* src = reinterpret_cast<const int4*>(
          k_base + (size_t)start * slot_stride);
      int4* dst = reinterpret_cast<int4*>(&sK[0][0]);
      const int stride4 = heads * 8;
#pragma unroll
      for (int j = 0; j < 512 / THREADS; j++) {
        const int L = j * THREADS + tid;
        dst[L] = src[(size_t)(L >> 3) * stride4 + (L & 7)];
      }
    }
    {
      // V 是 fp16：每 slot 16 个 int4（256 B），s = L>>4, w = L&15。
      const int4* src = reinterpret_cast<const int4*>(
          v_base + (size_t)start * slot_stride);
      int4* dst = reinterpret_cast<int4*>(&sV[0][0]);
      const int stride4 = heads * 16;
#pragma unroll
      for (int j = 0; j < 1024 / THREADS; j++) {
        const int L = j * THREADS + tid;
        dst[L] = src[(size_t)(L >> 4) * stride4 + (L & 15)];
      }
    }
    __syncthreads();

    const float k_scale = __ldg(KScale + (size_t)head * n_kblocks + kblk);
    const float qk_scale = q_scale * k_scale;
    const int valid = __ldg(Valid + kblk);

    // ---- 分 SUB 个子块：QK -> online softmax -> PV -----------------------
#pragma unroll
    for (int s = 0; s < SUB_ITERS; s++) {
      float qk[SUB];

#pragma unroll
      for (int t = 0; t < SUB; t++) {
        const int k = s * SUB + t;
        int dot = 0;
#if VEDA_NO_LDS
        // 诊断用：把 K 换成寄存器里的假数据，彻底摘掉 LDS。
        // 结果当然是错的 —— 只为回答「LDS 是不是瓶颈」。
        // 若时间掉到 1/5 以下，说明瓶颈是 K 的共享内存加载，
        // 那么唯一出路是 register tiling（一次 LDS 服务多个 dp4a）。
#pragma unroll
        for (int w = 0; w < Q_WORDS; w++) {
          dot = __dp4a(q_words[w], q_words[(w + k) & (Q_WORDS - 1)], dot);
        }
#elif VEDA_VEC_K
        // 向量化：int4 一次拿 4 个字 ⇒ warp 级 LDS 指令数降 4 倍。
        // sK[k] 起点是 k*128 字节，恒 16 B 对齐。
        int k_words[Q_WORDS];
        const int4* k4 = reinterpret_cast<const int4*>(&sK[k][0]);
#pragma unroll
        for (int i = 0; i < Q_WORDS / 4; i++) {
          const int4 kv = k4[i];
          k_words[i * 4 + 0] = kv.x;
          k_words[i * 4 + 1] = kv.y;
          k_words[i * 4 + 2] = kv.z;
          k_words[i * 4 + 3] = kv.w;
        }
#pragma unroll
        for (int w = 0; w < Q_WORDS; w++) {
          dot = __dp4a(q_words[w], k_words[w], dot);
        }
#else
        const int* krow = reinterpret_cast<const int*>(&sK[k][0]);
#pragma unroll
        for (int w = 0; w < Q_WORDS; w++) {
          dot = __dp4a(q_words[w], krow[w], dot);
        }
#endif
        qk[t] = (k < valid) ? ((float)dot * qk_scale) : -INFINITY;
      }

      // online softmax（线程本地，SUB 元归约）
      float m_new = m_i;
#pragma unroll
      for (int t = 0; t < SUB; t++) m_new = fmaxf(m_new, qk[t]);
      const float alpha = exp2f(m_i - m_new);
      float row_sum = 0.0f;
#pragma unroll
      for (int t = 0; t < SUB; t++) {
        const float p = exp2f(qk[t] - m_new);
        qk[t] = p;                 // 原地复用：qk 变成 p
        row_sum += p;
      }
      l_i = l_i * alpha + row_sum;
#pragma unroll
      for (int d = 0; d < SEG_DIM; d++) acc[d] *= alpha;
      m_i = m_new;

      // PV：p(fp16) x V(fp16) -> fp32 累加（只累加本线程的 SEG_DIM 维）
#pragma unroll
      for (int t = 0; t < SUB; t++) {
        const float p = __half2float(__float2half(qk[t]));
        const __half* vrow = &sV[s * SUB + t][d0];
#pragma unroll
        for (int d = 0; d < SEG_DIM; d += 2) {
          const float2 fv = __half22float2(
              *reinterpret_cast<const half2*>(vrow + d));
          acc[d] = fmaf(p, fv.x, acc[d]);
          acc[d + 1] = fmaf(p, fv.y, acc[d + 1]);
        }
      }
    }

    __syncthreads();   // 本块读完，下一轮才能覆盖 sK / sV
  }

  // 对角线永远被保留，所以 l_i > 0；这里仍守一手，防畸形 mask 除零。
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 1.0f;
  __half* out_row = Out + ((size_t)(qtile * TILE + row)) * heads * HEAD_DIM
                    + (size_t)head * HEAD_DIM;
#pragma unroll
  for (int d = 0; d < SEG_DIM; d += 2) {
    half2 h;
    h.x = __float2half(acc[d] * inv);
    h.y = __float2half(acc[d + 1] * inv);
    *reinterpret_cast<half2*>(out_row + d0 + d) = h;
  }
}

}  // namespace

// ---------------------------------------------------------------------------

torch::Tensor sparse_int8_attention(
    const torch::Tensor& q_i8,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_i8,
    const torch::Tensor& k_scale,
    const torch::Tensor& v16,
    const torch::Tensor& index,
    const torch::Tensor& count,
    const torch::Tensor& valid)
{
  TORCH_CHECK(q_i8.is_cuda() && q_i8.is_contiguous(), "q must be contiguous CUDA");
  TORCH_CHECK(q_i8.scalar_type() == torch::kChar, "q must be int8");
  TORCH_CHECK(q_i8.dim() == 3, "q must be [N, H, D]");
  TORCH_CHECK(q_i8.size(2) == HEAD_DIM, "D must be 128");
  const int64_t slots = q_i8.size(0);
  TORCH_CHECK(slots % TILE == 0, "N must be a multiple of 128");
  const int heads = (int)q_i8.size(1);
  const int n_qtiles = (int)(slots / TILE);
  const int n_kblocks = (int)(slots / KEY_BLOCK);

  auto out = torch::empty({slots, (int64_t)heads, (int64_t)HEAD_DIM},
                          q_i8.options().dtype(torch::kHalf));

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n_qtiles, heads);
  sparse_int8_attention_kernel<<<grid, THREADS, 0, stream>>>(
      q_i8.data_ptr<int8_t>(), q_scale.data_ptr<float>(),
      k_i8.data_ptr<int8_t>(), k_scale.data_ptr<float>(),
      reinterpret_cast<const __half*>(v16.data_ptr<at::Half>()),
      index.data_ptr<int32_t>(), count.data_ptr<int32_t>(),
      valid.data_ptr<int32_t>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      heads, n_qtiles, n_kblocks,
      (int)index.stride(0), (int)index.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_int8_attention", &sparse_int8_attention,
        "Veda block-sparse INT8 attention (sm75 DP4A fallback)");
}
