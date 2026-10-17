// Veda 块稀疏 INT8 注意力：sm75（Turing）张量核版（C1）。
//
// 为什么有第二个内核
// ------------------
// cuda_int8.py 那个 DP4A 版（sparse_int8_cuda.cu）**正确性全过、性能不达标**：
// 一段 62 ms vs sageattn 稠密 38 ms。两个峰值微基准把原因量化清楚了：
//
//     引擎                             实测峰值
//     DP4A（CUDA core 的 __dp4a）      7.575e12 dp4a/s = 60.6 int8 TOPS
//     张量核 mma.m8n8k16.s8            1.003e14 MAC/s  = 200.5 int8 TOPS   (3.31x)
//
// 而 DP4A 版只跑到自己峰值的 **8.2%** —— 卡在「发射/stall」，而 stall 的根因是
// 寄存器锁死 occupancy（q_words[32]+acc[32]+qk[16] = 80，512 线程下 REG 顶到 128
// = 100% 寄存器文件 ⇒ 每 SM 只能 1 个 block ⇒ 25% occupancy）。这个死结 DP4A
// 解不开；而张量核微基准 **189–200 TOPS 全程平稳、不随 occupancy 变化**。
//
// 这个文件只做一件事：**把两个 dot 换成张量核**，算法/量化/掩码语义与 DP4A 版
// 以及 `kernels/sage/sparse_int8.py`（Triton 版）逐条一致。
//
// 结构（关键设计）
// ----------------
//   * 8 个 warp / 256 线程，**每个 warp 固定负责 16 条 query 行**。
//     为什么是 16 而不是 8？因为要让「QK 的输出」直接就是「PV 的输入」：
//
//         QK  mma.m8n8k16 -> C 片段是 8x8，线程 (g,t) 持 C[g][2t], C[g][2t+1]
//         PV  mma.m16n8k8 -> A 片段是 16x8，线程 (g,t) 持 A[g][2t], A[g][2t+1],
//                                                       A[g+8][2t], A[g+8][2t+1]
//
//     把同一个 nt（key 子块）下 **mt=0 / mt=1 两个 QK C 片段**拼起来，注册器里
//     的 4 个 s32 直接就是 PV 的 2 个 .b32 A 片段 —— **零 shuffle、零访存**。
//     若每 warp 只负责 8 行就退化成 m8n8k8，而 sm75 的 f16 mma 只有 m16n8k8 这一档
//     （m16n8k16 是 sm80+）。
//
//   * Q（128x128 int8，16 KB）**打进寄存器**：qreg[2][8] = 16 个 int32。
//     Q 在整段 key-block 循环里被复用 ~120 次，进 smem 反而占 16 KB（会把
//     occupancy 从 2 blocks/SM 压到 1），读 sQ 又是 32 路 bank 冲突。
//
//   * 共享内存只放 K / V，且**行步长刻意错开**以消 bank 冲突：
//       sK 行步长 144 B（36 words ≡ 4 mod 32）⇒ QK 取 B 片段时 8 个 g 组
//         落在 bank {4g + t}，32 个 lane 正好铺满 0..31，**零冲突**。
//         若用自然步长 128 B（32 words ≡ 0），32 个 lane 全砸在同一组 bank 上。
//       sV 行步长 136 halfs（68 words ≡ 4 mod 32）⇒ PV 取 B 片段时
//         bank {8t + 4dt + 0..3}，冲突在同一个 4 B 字内、走广播，**免费**。
//
//   * online softmax 在**寄存器内**对 16 行做：行 g 的值散布在 lane g*4 .. g*4+3
//     这 4 个线程里（每线程 2 个 × 8 个 nt = 16 个值），所以行归约就是
//     `__shfl_xor_sync(m, v, 1)` 再 `, 2)` —— 一次 4-lane 蝶形，不用 smem。
//
//   * key 的 padding 用 `valid`（每块真实行数的前缀）掩成 -INFINITY，
//     再 `exp2f` ⇒ 得 0，与 DP4A 版同一套（m_i 初始化成 -1e30f 而不是 -INFINITY，
//     这样 valid==0 的块 alpha = exp2(0) = 1 而不是 NaN）。
//
// C2/C3：在 C1 基础上试削几处（**都已实现、都已否决，默认关**）
// ---------------------------------------------------------
//   立项依据是 C1 的 `c1_variants.py` 拆解：每次 key block 迭代 127 ns 里
//   「纯计算 83 ns（65%）| LDS 读 36 ns（28%，其中 V 占 21）| 载入+同步 8 ns（6%）」。
//   ⚠️ **那份拆解是坏的**：当时没锁频（SM 时钟在 1725~1935 MHz 之间跳）、变体按
//   固定顺序测（轮内位置越靠后越慢，实测差到 7%）。锁频 1500 MHz + 轮次旋转后
//   重测的真实分布（`c3_variants.py --group=probe`，见 `结论C3_…md`）：
//       标量 softmax 段（掩码+max+exp2+求和+重标定）  11.3 / 9.5 / 10.5 %
//       其中 I2F + 缩放                            13 %（含解开 si 依赖的收益）
//       K/V 的 smem 读                              26.5 / 27.6 / 26.0 %
//       协作载入 + 同步                             仅 0.6 %（27.1 - 26.5）
//       mma + 循环（不可压）                         44.4 / 46.9 / 49.3 %
//   ⇒ 「V 的 LDS 占 17%」是坏测量（真实 4~6%），opt1 从一开始就瞄错了目标；
//     「双缓冲载入」的天花板也只有 0.6%，不值得做（早先记的 6% 也偏高）。
//
//   全部实测结论（锁 1500 MHz + 多轮旋转交错 + 取最小，详见结论报告）：
//     VEDA_VINT（V 的 LDS 指令数减半）        ±2% 噪声带内 = 零
//     VEDA_ACCFAST（跳 69~75% 的 acc 重标定）  慢 4~10%
//     VEDA_C3_MASKHOIST（删 48 条 setp/sel）   慢 2~4%
//     VEDA_C3_EX2F16（MUFU 减半）              慢 2.5% ~ 快 1.6%
//     VEDA_C3_PVPIPE（PV 批量载入）            慢 1%
//     VEDA_C4_FUSE（消 sf/pa 数组）            慢 3~4%（且寄存器一点没降：192）
//     VEDA_MINBLK=2（要 2 block/SM）           慢 4~8%，融合版慢 23~46%
//
//   **真正的根因（三轮实验一致指向）**：瓶颈是**延迟**而不是发射槽 ——
//     ① 把 K/V 的 smem 读整段停掉（操作数变寄存器常数）省 26.5%，
//        但把 V 的读指令数减半（VINT）是 0% ⇒ 省的是**依赖**，不是条数；
//     ② 整个 softmax 段值 11.3%，但拆开一段段去掉都是 0 或负；
//     ③ 每个「寄存器更少」的变体都更慢（191→168/165/128），因为那 191 个寄存器
//        正是编译器用来做载入预取的缓冲。
//     ④ 唯一出路是把常驻 warp 从 8 提到 16，但 65536 寄存器/SM 装不下本设计的
//        工作集（acc 64 + qreg 16 + si 32 = 112 起步）—— 强压到 128 必然溢出，
//        溢出把收益全吃掉（`STACK:88` → −5%，`STACK:160` → −28%）。
//     ⑤ 进一步只能减工作集（把 head_dim 拆到两组 warp，或把 PV 的累加器换成
//        fp16）—— 那是一次结构性重写，未做。
//
//   保留以下代码（用开关关着），把设计理由与实测结论都写在这里：
//
//   [1] V 的 key 对交错布局（VEDA_VINT，实测 **±2% 噪声带内 = 没收益**）
//       PV 的 B 片段要 {V[k][d], V[k+1][d]}（k 为偶数）。C1 走「两次 LDS.16 再拼」，
//       因为 [key][dim] 布局里这两行隔着一整个行步长。
//       改法：smem 改成 **key 对交错**布局
//             word(p, d) = { V[2p][d], V[2p+1][d] }        p = key/2
//       那么 PV 要的那个 word 就在一个地址上，**一次 LDS.32** 拿全 ——
//       每线程每迭代 256 次 LDS.16 → 128 次 LDS.32。
//       代价在载入端：一个线程读同一 key 对的**两行同段**（各 1 个 int4），
//       再用 8 条 PRMT（`prmt_low`/`prmt_high`）交错成 8 个 word，写 2 个 STS.128。
//       净账：−128 LDS + 16 PRMT ⇒ 理论少 ~110 条指令/线程/迭代，**实测为零**。
//
//       行步长 272 halfs = 136 words（≡ 8 mod 32）而不是紧凑的 256 halfs：
//         读 banks = (136*(4nt+t) + 8dt + g) mod 32 = (8t + 8dt + g) mod 32，
//         nt / dt 在 warp 内是均匀量 ⇒ 32 个 lane 覆盖 0..31 各一次，**零冲突**。
//         紧凑 256 halfs（≡0）会让整个 warp 砸在同一组 bank 上。
//       写端有 2 路冲突（w 与 w+4 同 bank），但写只占读的 1/32 频次，忽略。
//       占用不变：32 对 × 272 halfs × 2 B = 17408 B，与 C1 的 sV 一模一样。
//
//   [2] acc 重标定只在 m 真变了才做（VEDA_ACCFAST，实测 **慢 4~10%**）
//       online softmax 每块无条件把 64 个 acc 乘 alpha，而 alpha = exp2(m_old -
//       m_new) 在 m 没长时恒为 1.0。判据必须**整个 warp 一致**才有意义（逐 lane
//       分支在 SASS 上只是谓词化，发射槽照占）：`__all_sync` 一把。
//       结果：跳过率 69~75%（实测），但总时间反而变慢 —— 那个分支正好插在
//       softmax 与 PV mma 之间，把两段之间的软件流水打断了，代价大于收益。
//       数值上与无条件版**完全等价**（乘 1.0 是精确的），所以否决纯粹因为速度。
//
// 与 DP4A 版的关系
// ----------------
//   两份 .cu 并存，各自独立编译。DP4A 版是**已知正确的兜底**，这个文件是提速版。
//   只有当本文件的 [1]/[2] 正确性全过、[3] 快过 sageattn 稠密，才谈得上替换。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---- 诊断开关（只用于定位瓶颈，打开后数值必错）----------------------------
//   VEDA_STUB_K     1 = QK 的 K 片段不读 sK，用寄存器假数
//   VEDA_STUB_V     1 = PV 的 V 片段不读 sV，用常数
//   VEDA_STUB_SMEM  1 = 连协作载入 K/V 到 smem 也跳过（配合上面两个用）
//   VEDA_MINBLK     __launch_bounds__ 的最少 block 数（1 = 基线；2 = 硬压到 128 寄存器）
// 拆解逻辑：base - stubKV = LDS 读的成本；stubKV - stubSMEM = 全局载入 + 同步的成本。
#ifndef VEDA_STUB_K
#define VEDA_STUB_K 0
#endif
#ifndef VEDA_STUB_V
#define VEDA_STUB_V 0
#endif
#ifndef VEDA_STUB_SMEM
#define VEDA_STUB_SMEM 0
#endif
#ifndef VEDA_MINBLK
#define VEDA_MINBLK 1
#endif

// ---- C2 优化开关 -----------------------------------------------------------
// ⚠️ 两个都写完了、也都实测过，结论是**默认全部关掉**（2026-10-16）。
//    测量条件：SM 时钟锁死 1500 MHz + 7 轮旋转交错 + 取最小（`c2_variants.py`）。
//
//   VEDA_VINT=1     V 的 B 片段从「两次 LDS.16」改成「一次 LDS.32」。
//                   V 的 LDS 确实降下来了，但**总时间没变**（三组几何落在 ±2%
//                   噪声带内）。原因：省下的发射槽被载入端新增的 16 条 PRMT +
//                   交错写吃回去了。更关键的是**重测发现 V 的 LDS 本来就只有
//                   ~5%** —— C1 报告 §5 那个「V 占 17%」是在未锁频、变体固定
//                   顺序下测的坏数据（见 `结论C2_两处优化_2026-10-16.md` §二/§四）。
//                   也就是说这条优化从立项起就瞄错了目标。
//
//   VEDA_ACCFAST=1  alpha 全 1 时整 warp 跳过 acc 重标定。
//                   分支命中率实测 **69~75%**（计数器实测，不是估计），
//                   但总时间**反而慢 4~10%**：warp 级 `__all_sync` + 分支把
//                   重标定前后那段软件流水打断了，代价大于省下的 ~46 条 fmul/块。
//                   与 B1/C1 的老教训同型：**负收益的结构性改动，压指标不换收益。**
//
//   两个开关都保留（随时可 `-D` 打开复测），但**出厂内核 = 已验证的 C1**。
#ifndef VEDA_VINT
#define VEDA_VINT 0
#endif
#ifndef VEDA_ACCFAST
#define VEDA_ACCFAST 0
#endif

// ---- C3：往「算术 65~70%」里打（2026-10-16 晚）----------------------------
// 先量再改。C2 的教训是「照着坏测量立项」，所以这次先加两个**只用于计时**的
// 探针，把 65~70% 的算术再劈开，看清哪一段才值得动；两处真优化用开关关着。
//
//   探针（打开后数值必错，只为计时）
//     VEDA_STUB_SOFT  跳过掩码 / 行 max / exp2 / 行求和 / acc 重标定，
//                     但**保留 `(float)si * qk_scale`** —— 这样 QK mma 不会
//                     被 DCE 掉。base - stubsoft = 整个标量 softmax 段的上限。
//     VEDA_STUB_CONV  在上一档基础上连 I2F + 缩放也换成常数（用一个恒假条件
//                     保住 si）。stubsoft - stubconv = I2F + 缩放的成本。
//                     有了这两档就能把算术劈成三段：
//                       [I2F+缩放] / [掩码+max+exp2+求和+重标定] / [mma+LDS+载入]
//
//   真优化（默认关，等 A/B 定）
//     VEDA_C3_MASKHOIST  `valid` 在本 block 内是均匀量（同一条 __ldg 地址），
//                        所以「整块有效」可以做成**真分支**而不是谓词：全块有效
//                        的块整段删掉 64 条 setp/sel。真实数据里除最后一个
//                        key block 外全是全块有效（39296 token / 64 = 614 块里
//                        只有 1 块是部分有效）⇒ 命中率 ~99.8%。
//     VEDA_C3_EX2F16     用 `ex2.approx.f16x2`（**已由 c3_probe 在 sm75 上实证
//                        存在**，SASS 里是 MUFU.EX2.F16，一条吃 2 个 fp16）：
//                          32 次 MUFU.EX2.F32  ->  16 次 MUFU.EX2.F16
//                        并且 P 直接以 fp16 产出，PV 前那 16 次 __floats2half2_rn
//                        一并省掉；行求和改成 fp16x2 的 __hadd2（一条顶两条 FADD，
//                        且一个寄存器同时累加两行）。
//                        精度实测（c3_probe）：最大绝对误差 6.0e-4、绝对误差
//                        >1e-3 的样本 0 个、-inf 正确出 0；平均相对误差 1.3e-3。
#ifndef VEDA_STUB_SOFT
#define VEDA_STUB_SOFT 0
#endif
#ifndef VEDA_STUB_CONV
#define VEDA_STUB_CONV 0
#endif
#ifndef VEDA_C3_MASKHOIST
#define VEDA_C3_MASKHOIST 0
#endif
#ifndef VEDA_C3_EX2F16
#define VEDA_C3_EX2F16 0
#endif
//   VEDA_C3_PVPIPE   把 PV 的 V 片段「先把 16 个 dt 全载进寄存器、再做 16 次 mma」
//                    显式拉开「载入到使用」的距离。C3 诊断显示瓶颈是依赖/延迟
//                    而不是发射槽（VINT 把 V 的读指令数减半却零收益），所以直接
//                    给编译器指定这个距离，而不是指望它自己排。
#ifndef VEDA_C3_PVPIPE
#define VEDA_C3_PVPIPE 0
#endif
//   VEDA_C4_FUSE     把 softmax 的中间数组全消掉：不再落 `sf[2][8][2]`（32 个 fp32）
//                    与 `pa[2][8]`（16 个），改成两趟直接读 `si` ——
//                    第一趟只求行 max，第二趟算 P、求和、打包、当场喂 PV。
//                    代价：I2F / FMUL / 掩码比较各多做一遍（约 +128 条指令/线程/迭代）。
//                    动机（C3 第二轮实测）：瓶颈是延迟而不是发射槽，唯一出路是把
//                    常驻 warp 从 8 提到 16；而强压 128 寄存器会溢出（STACK:88），
//                    溢出把收益全吃了 —— 所以必须**先真降压力**，再要 2 block/SM。
#ifndef VEDA_C4_FUSE
#define VEDA_C4_FUSE 0
#endif

// ---- A：把 head_dim 拆到两组 warp（VEDA_DIMSPLIT）--------------------------
//
// C3 §四④ 把唯一剩下的路指死在这里：**瓶颈是延迟，出路是把常驻 warp 从 8 提到
// 16**（256 线程 × 191 REG ⇒ 每 SM 只能 1 个 block），而 65536 寄存器/SM 装不下
// 本设计的工作集 —— 强压 128 必然溢出，溢出把收益全吃光（`STACK:88` → −5%，
// `STACK:160` → −28%，一一对应）。所以要先把工作集**真的**降下来。
//
// 工作集里最大的一块是 `acc`（16 段 × 4 个 fp32 = 64 个寄存器），它覆盖 128 个
// 输出 dim 且必须活过整个 key-block 循环，压不掉。唯一的减法就是**让每个 warp
// 只负责一半的输出 dim**：
//
//     warp 2i   -> 行组 i（16 行）× 输出 dim 0..63    acc[8][4] = 32
//     warp 2i+1 -> 行组 i（16 行）× 输出 dim 64..127  acc[8][4] = 32
//
// 于是 16 warp / 512 线程、每线程 65536/512 = **128 寄存器**，恰好 1 block/SM
// = **16 warp/SM（50% occupancy，C1 是 25%）**。
//
// 代价说清楚（这是「赌」的部分）：
//   * **QK 必须算两遍**。score 要对**全部** 128 个 dim 求和，两个 warp 各算一遍
//     才能各自拿到完整 si —— 没有跨 warp 归约（那要走 smem + 额外 __syncthreads，
//     会让配对的 warp 在 QK/softmax 期间干等，把延迟藏匿能力反而打没）。
//     ⇒ IMMA 128 → 256/block，总 mma 2048 → 3072（+50%）。
//   * **标量 softmax 也翻倍**（worker 数翻倍、每个 worker 的行数没变）：
//     score 段 + prob 段各 ×2。
//   * **K 的 LDS 翻倍**、**V 的 LDS 总量不变**（各读自己那半）。
//
// 换来的：issue 效率从 19% 起步翻倍（warp 数 ×2）。是赚是亏只能实测 ——
// 本项目已经三次证明「看起来合理的推断」在这个内核上会落空。
#if defined(VEDA_DIMSPLIT) && VEDA_DIMSPLIT
#define VEDA_DIMSPLIT_ON 1
#else
#define VEDA_DIMSPLIT_ON 0
#endif

// 只在显式 `-DVEDA_ACCFAST_STAT=1` 时编进去：给 opt2 那个分支装个计数器，
// 回答「到底有多少比例的块真的跳过了重标定」。带 atomicAdd，会污染计时，
// 所以**默认产物里一行都不存在**（连符号都没有）。
#ifdef VEDA_ACCFAST_STAT
__device__ unsigned long long g_accfast_total = 0ull;   // warp x block 总数
__device__ unsigned long long g_accfast_skip  = 0ull;   // 其中整 warp 跳过的
#endif

namespace {

constexpr int TILE          = 128;   // 一个 Veda tile 的行数
constexpr int KEY_BLOCK     = 64;    // 一个 key block 的 slot 数
constexpr int HEAD_DIM      = 128;   // D
// VEDA_DIMSPLIT=1：16 warp，warp 对 = (行组, 输出维组)；=0：C1 的 8 warp。
constexpr int NWARP         = VEDA_DIMSPLIT_ON ? 16 : 8;
constexpr int THREADS       = NWARP * 32;        // 512 / 256
constexpr int NGRP          = VEDA_DIMSPLIT_ON ? NWARP / 2 : NWARP;   // 本 tile 的行组数
constexpr int ROWS_PER_WARP = TILE / NGRP;       // 16（两种配置都是 16）

constexpr int SK_STRIDE = 144;        // bytes / key 行（36 words）
constexpr int SV_STRIDE = 136;        // halfs / key 行（68 words）—— C1 的 [key][dim]
constexpr int NPAIR      = KEY_BLOCK / 2;     // 32    key 对（交错布局的行数）
constexpr int SV2_STRIDE = 272;       // halfs / key 对行（136 words ≡ 8 mod 32）
constexpr int QWORDS    = HEAD_DIM / 4;          // 32
constexpr int NKSTEP    = HEAD_DIM / 16;         // 8  （m8n8k16 的 k=16）
constexpr int NNT       = KEY_BLOCK / 8;         // 8  （QK 的 n=8；PV 的 k=8）
constexpr int NDT       = HEAD_DIM / 8;          // 16 （PV 的 n=8）
// 每个 warp 负责的输出 8-dim 段数：拆了就是 8（64 个 dim），没拆就是全部 16。
constexpr int NDT_W     = VEDA_DIMSPLIT_ON ? NDT / 2 : NDT;

// 消息必须纯 ASCII：nvcc 前端解析**字符串字面量**时不吃 -Xcompiler=/utf-8，
// 中文字面量会直接报 missing closing quote（注释里的中文没事）。
static_assert(ROWS_PER_WARP == 16, "PV packing assumes 16 query rows per warp");
static_assert(NGRP * ROWS_PER_WARP == TILE, "row groups must tile the whole tile");
static_assert(2 * NDT_W == (VEDA_DIMSPLIT_ON ? NDT : 2 * NDT),
              "dim-split warp pairs must cover all output segments exactly once");
static_assert(NWARP * 32 == THREADS, "warp/thread count mismatch");
static_assert(THREADS % 32 == 0, "threads must be a warp multiple");
static_assert(THREADS <= 512, "block must stay within one SM's 1024-thread budget");

// K 每 block: 64 slot x 128 B = 8192 B = 512 int4；V: 64 x 256 B = 1024 int4。
// （曾经把 K 写成 2048，slot = L>>3 直冲 255 ⇒ 越界非法访存。）
static_assert(512 % THREADS == 0, "K load: 512 int4 must divide by THREADS");
static_assert(1024 % THREADS == 0, "V load: 1024 int4 must divide by THREADS");
static_assert(SK_STRIDE % 16 == 0, "sK row stride must keep 16B alignment");
static_assert((SV_STRIDE * 2) % 16 == 0, "sV row stride must keep 16B alignment");
// 交错布局的两个硬约束：
//   ① 每行要放得下 128 个 word，且整行必须 16 B 对齐（int4 存）
//   ② 行步长（words）≡ 8 mod 32 —— 这是「读 banks = 8t + 8dt + g」无冲突的唯一条件
//      ⇒ 128 words 起、满足 ≡ 8 mod 32 的最小值是 136（= 64 对 + 8 个 word 的 padding）
static_assert((SV2_STRIDE / 2) >= HEAD_DIM / 2, "sV2 pair row must hold 128 words");
static_assert((SV2_STRIDE * 2) % 16 == 0, "sV2 pair row stride must keep 16B alignment");
static_assert((SV2_STRIDE / 2) % 32 == 8, "sV2 stride must land on bank 8t + g");
static_assert(2 * NPAIR == KEY_BLOCK, "pair count must cover the key block");
// 交错布局的载入单位：32 对 × 16 段 = 512 个 unit（每个 unit 读 2 个 int4）
static_assert((NPAIR * (HEAD_DIM / 8)) % THREADS == 0,
              "V interleaved load: 512 units must divide by THREADS");

// ---------------------------------------------------------------- mma 包装
//
// m8n8k16（QK）：A 8x16 row / B 16x8 col -> C 8x8 s32，每线程 1 个 .b32 operand
//               （fragment 映射由 b0_smoke/smoke_ext.cu 数值验证过）。
//   线程 (g,t)：a = A[g][4t..4t+3]，b = B[4t..4t+3][g]，c = C[g][2t..2t+1]
__device__ __forceinline__ void mma_s8_m8n8k16(int32_t& c0, int32_t& c1,
                                               uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%0, %1};\n"
      : "+r"(c0), "+r"(c1)
      : "r"(a), "r"(b));
#else
  c0 = 0; c1 = 0;    // host 轨迹只为编过，不参与数值
#endif
}

// m16n8k8（PV）：A 16x8 row / B 8x8 col -> C 16x8 f32，A 每线程 2 个 .b32
//   线程 (g,t)：a0 = {A[g][2t], A[g][2t+1]}，a1 = {A[g+8][2t], A[g+8][2t+1]}
//               b0 = {B[2t][g], B[2t+1][g]}
//               c  = C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1,
                                                float& c2, float& c3,
                                                uint32_t a0, uint32_t a1,
                                                uint32_t b0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(b0));
#else
  c0 = 0.f; c1 = 0.f; c2 = 0.f; c3 = 0.f;
#endif
}

// 4-lane 蝶形（lane g*4 .. g*4+3 一组），把一条 query 行的局部量归约成整行值。
__device__ __forceinline__ float row_reduce_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float row_reduce_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}

// ---- MUFU.EX2.F16：一条指令对 2 个 fp16 做 exp2（sm75 起）------------------
// 由 `c3_probe.cu` 实证：nvcc -arch=sm_75 下 ptxas 真的生成 `MUFU.EX2.F16`，
// 并且 -inf 输入正确产出 0。不是查文档推的。
__device__ __forceinline__ uint32_t ex2_f16x2_u32(uint32_t x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  uint32_t r;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
  return r;
#else
  return x;    // host 轨迹：只为编过
#endif
}

// ---- 标量 softmax 拆成两段，各自可用模板参数选快/慢路径 --------------------
//
// 第一段：I2F + 缩放 +（可选）掩码 + 逐行 max。
// MASK 做成模板参数而不是运行时判断 —— 这是 `VEDA_C3_MASKHOIST` 的关键：
// 谓词化的 setp/sel 在 SASS 上照样占发射槽，只有**真分支**才能让编译器整段删掉。
template <bool MASK>
__device__ __forceinline__ void score_stage(const int32_t (*si)[NNT][2],
                                            float (*sf)[NNT][2],
                                            float& mm0, float& mm1,
                                            const float qk_scale,
                                            const int valid, const int t) {
  mm0 = -1e30f;
  mm1 = -1e30f;
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
#pragma unroll
    for (int nt = 0; nt < NNT; nt++) {
      float a = (float)si[mt][nt][0] * qk_scale;
      float b = (float)si[mt][nt][1] * qk_scale;
      if (MASK) {
        const int k0 = nt * 8 + 2 * t;
        if (k0 >= valid)     a = -INFINITY;
        if (k0 + 1 >= valid) b = -INFINITY;
      }
      sf[mt][nt][0] = a;
      sf[mt][nt][1] = b;
      if (mt == 0) { mm0 = fmaxf(mm0, a); mm0 = fmaxf(mm0, b); }
      else         { mm1 = fmaxf(mm1, a); mm1 = fmaxf(mm1, b); }
    }
  }
}

// 第二段：exp2 + 打包成 PV 的 A 片段 + 逐行求和。
// 两个分支都产出同一份 `pa`（每线程 2mt x 8nt 个 .b32），所以 PV 段与开关无关。
template <bool F16>
__device__ __forceinline__ void prob_stage(const float (*sf)[NNT][2],
                                           uint32_t (*pa)[NNT],
                                           float& sum0, float& sum1,
                                           const float m0, const float m1) {
  sum0 = 0.f;
  sum1 = 0.f;
#if VEDA_C3_EX2F16
  if (F16) {
    // 自变量打包成 fp16（这正是省下一半 MUFU 的地方），
    // 行和用 __hadd2 累加：一条指令同时喂两行（.x 收 j=0 的 8 个、.y 收 j=1 的 8 个）。
    __half2 hs0 = __floats2half2_rn(0.f, 0.f);
    __half2 hs1 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
    for (int nt = 0; nt < NNT; nt++) {
      const __half2 x0 = __floats2half2_rn(sf[0][nt][0] - m0, sf[0][nt][1] - m0);
      const __half2 x1 = __floats2half2_rn(sf[1][nt][0] - m1, sf[1][nt][1] - m1);
      const uint32_t r0 = ex2_f16x2_u32(*reinterpret_cast<const uint32_t*>(&x0));
      const uint32_t r1 = ex2_f16x2_u32(*reinterpret_cast<const uint32_t*>(&x1));
      const __half2 p0 = *reinterpret_cast<const __half2*>(&r0);
      const __half2 p1 = *reinterpret_cast<const __half2*>(&r1);
      pa[0][nt] = *reinterpret_cast<const uint32_t*>(&p0);
      pa[1][nt] = *reinterpret_cast<const uint32_t*>(&p1);
      hs0 = __hadd2(hs0, p0);
      hs1 = __hadd2(hs1, p1);
    }
    sum0 = __low2float(hs0) + __high2float(hs0);
    sum1 = __low2float(hs1) + __high2float(hs1);
    return;
  }
#endif
#pragma unroll
  for (int nt = 0; nt < NNT; nt++) {
    const float p00 = exp2f(sf[0][nt][0] - m0);
    const float p01 = exp2f(sf[0][nt][1] - m0);
    const float p10 = exp2f(sf[1][nt][0] - m1);
    const float p11 = exp2f(sf[1][nt][1] - m1);
    const __half2 h0 = __floats2half2_rn(p00, p01);
    const __half2 h1 = __floats2half2_rn(p10, p11);
    pa[0][nt] = *reinterpret_cast<const uint32_t*>(&h0);
    pa[1][nt] = *reinterpret_cast<const uint32_t*>(&h1);
    sum0 += p00 + p01;
    sum1 += p10 + p11;
  }
}

// ---- V 的 key 对交错：两条 PRMT 把两个 .b32（各含 2 个 fp16）拼成交错 word ----
//
//   prmt.b32 d, a, b, sel：a 占源字节 0..3、b 占 4..7，sel 的 4 个 nibble
//   从低到高依次选 d 的字节 0..3。
//     0x5410 -> 字节 0,1,4,5 = {a 的低半字, b 的低半字} = {a.x, b.x}
//     0x7632 -> 字节 2,3,6,7 = {a 的高半字, b 的高半字} = {a.y, b.y}
//   比「移位 + LOP3」手写省一半 ALU（8 条 PRMT 顶 32 条逻辑运算）。
__device__ __forceinline__ uint32_t prmt_low(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(r) : "r"(a), "r"(b));
  return r;
#else
  return (a & 0x0000ffffu) | (b << 16);
#endif
}
__device__ __forceinline__ uint32_t prmt_high(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(r) : "r"(a), "r"(b));
  return r;
#else
  return (a >> 16) | (b & 0xffff0000u);
#endif
}

// ---- PV 的 B 片段：{V[8nt+2t][8dt+g], V[8nt+2t+1][8dt+g]} ------------------
// 抽成函数是为了让 VEDA_C3_PVPIPE 能把「载入」与「使用」拆成两个循环。
__device__ __forceinline__ uint32_t v_frag(const __half* __restrict__ sV,
                                           const int nt, const int dt,
                                           const int g, const int t) {
#if VEDA_STUB_V
  (void)sV; (void)nt; (void)dt; (void)g; (void)t;
  return 0x3c003c00u;                        // {1.0, 1.0} fp16
#elif VEDA_VINT
  // 交错布局：key 对 (4nt+t) 的这个 word 直接就是 b0，一次 LDS.32 拿全。
  // banks = (136*(4nt+t) + 8dt + g) mod 32 = (8t + 8dt + g) mod 32，
  // nt / dt 在 warp 内均匀 ⇒ 32 lane 覆盖 0..31 各一次，零冲突。
  const uint32_t* vp = reinterpret_cast<const uint32_t*>(sV)
                     + (size_t)(4 * nt + t) * (SV2_STRIDE / 2) + 8 * dt + g;
  return vp[0];
#else
  // C1 布局：两行隔着一整个行步长，只能两次 LDS.16 再拼
  const __half* vp = sV + (nt * 8 + 2 * t) * SV_STRIDE + dt * 8 + g;
  __half2 bh;
  bh.x = vp[0];
  bh.y = vp[SV_STRIDE];
  return *reinterpret_cast<const uint32_t*>(&bh);
#endif
}

// ------------------------------------------------------------------- 内核

__global__ __launch_bounds__(THREADS, VEDA_MINBLK)
void sparse_int8_mma_kernel(
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
    const int stride_ih,
    const int stride_iq)
{
  const int qtile = blockIdx.x;
  const int head  = blockIdx.y;
  const int tid   = threadIdx.x;
  const int warp  = tid >> 5;

  // VEDA_DIMSPLIT：warp 对 (2i, 2i+1) 共用行组 i，各负责一半输出 dim。
#if VEDA_DIMSPLIT_ON
  const int rg = warp >> 1;      // 0..7  行组
  const int dg = warp & 1;       // 0/1   输出维组
#else
  const int rg = warp;           // 0..7
  const int dg = 0;
#endif

  const int lane = tid & 31;
  const int g    = lane >> 2;     // 0..7  片段内的行/列组
  const int t    = lane & 3;      // 0..3  片段内的列/行组

  // 必须 16 B 对齐：K / V 都用 int4 向量存取（错位会直接算非法访存）
  __shared__ __align__(16) int8_t sK[KEY_BLOCK * SK_STRIDE];   // 9216 B
#if VEDA_VINT
  // key 对交错：word(p, d) = { V[2p][d], V[2p+1][d] }，PV 的 B 片段一次 LDS.32 到手
  __shared__ __align__(16) __half sV[NPAIR * SV2_STRIDE];      // 17408 B
#else
  __shared__ __align__(16) __half sV[KEY_BLOCK * SV_STRIDE];   // 17408 B
#endif
  // 两种布局字节数完全相同（32*272*2 == 64*136*2 == 17408），占用不受影响

  // ---- Q 进寄存器：qreg[mt][ks] = dims [16ks+4t, 16ks+4t+3] 的 4 个 int8 ----
  int32_t qreg[2][NKSTEP];
  {
    const size_t row_stride = (size_t)heads * HEAD_DIM;
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      const int row = rg * ROWS_PER_WARP + mt * 8 + g;
      const int8_t* qp = Q + ((size_t)(qtile * TILE + row)) * row_stride
                            + (size_t)head * HEAD_DIM;
#pragma unroll
      for (int ks = 0; ks < NKSTEP; ks++) {
        qreg[mt][ks] = *reinterpret_cast<const int32_t*>(qp + ks * 16 + t * 4);
      }
    }
  }

  const float q_scale = __ldg(QScale + (size_t)head * n_qtiles + qtile);
  const int   count   = __ldg(Count  + (size_t)head * n_qtiles + qtile);
  const int32_t* idx_row = Index + (size_t)head * stride_ih
                                 + (size_t)qtile * stride_iq;

  const size_t  kv_stride = (size_t)heads * HEAD_DIM;   // q/k/v 的 slot 步长
  const int8_t* k_base    = K + (size_t)head * HEAD_DIM;
  const __half* v_base    = V + (size_t)head * HEAD_DIM;

  // 本线程负责的两条输出行：row0 = g（mt=0），row1 = g+8（mt=1）
  float m0 = -1e30f, m1 = -1e30f;
  float l0 = 0.f,    l1 = 0.f;
  // acc 覆盖本 warp 负责的那 NDT_W 个 8-dim 段（C1：16 段 = 128 dim；拆后 8 段 = 64 dim）
  float acc[NDT_W][4];
#pragma unroll
  for (int dt = 0; dt < NDT_W; dt++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[dt][j] = 0.f;

  for (int ib = 0; ib < count; ib++) {
    const int kblk  = __ldg(idx_row + ib);
    const int start = kblk * KEY_BLOCK;

    // ---- 协作载入 K / V 到 padded smem ---------------------------------
    // ⚠️ 不能当成连续块拷：q/k/v 是 [slot, head, dim]，一个 head 只占 128 B（K）
    //    / 256 B（V），slot 之间步长是 heads*128。整块读会读进别的 head。
    //    L -> (slot, 段内偏移)：K 每 slot 8 个 int4，slot = L>>3, w = L&7。
    {
#if !VEDA_STUB_SMEM
      const int4* src = reinterpret_cast<const int4*>(
          k_base + (size_t)start * kv_stride);
#pragma unroll
      for (int j = 0; j < 512 / THREADS; j++) {
        const int L    = j * THREADS + tid;
        const int slot = L >> 3;
        const int w    = L & 7;
        *reinterpret_cast<int4*>(sK + slot * SK_STRIDE + w * 16) =
            src[(size_t)slot * (heads * 8) + w];
      }
#endif
    }
    {
#if !VEDA_STUB_SMEM
      const int4* src = reinterpret_cast<const int4*>(
          v_base + (size_t)start * kv_stride);
#if VEDA_VINT
      // 512 个 unit = 32 个 key 对 × 16 个 int4 段。
      // 每个 unit：读同一对的**两行同段**（各 1 个 int4 = 8 个 half），
      // 用 8 条 PRMT 交错成 8 个 word（word i = {行A 的 half_i, 行B 的 half_i}），
      // 两次 STS.128 写进交错布局。全局读完全连续：同一 p 的 16 个线程正好
      // 读满 16 个 int4 = 256 B。
#pragma unroll
      for (int j = 0; j < (NPAIR * (HEAD_DIM / 8)) / THREADS; j++) {
        const int U = j * THREADS + tid;
        const int p = U >> 4;         // key 对
        const int w = U & 15;         // 段内第几个 int4（8 个 dim）
        const int4 A  = src[(size_t)(2 * p) * (heads * 16) + w];
        const int4 Bv = src[(size_t)(2 * p + 1) * (heads * 16) + w];
        const uint32_t* aw = reinterpret_cast<const uint32_t*>(&A);
        const uint32_t* bw = reinterpret_cast<const uint32_t*>(&Bv);
        uint32_t o[8];
#pragma unroll
        for (int k = 0; k < 4; k++) {
          o[2 * k]     = prmt_low(aw[k], bw[k]);
          o[2 * k + 1] = prmt_high(aw[k], bw[k]);
        }
        uint32_t* dst = reinterpret_cast<uint32_t*>(sV)
                      + (size_t)p * (SV2_STRIDE / 2) + 8 * w;
        *reinterpret_cast<int4*>(dst)     = *reinterpret_cast<const int4*>(o);
        *reinterpret_cast<int4*>(dst + 4) = *reinterpret_cast<const int4*>(o + 4);
      }
#else
#pragma unroll
      for (int j = 0; j < 1024 / THREADS; j++) {
        const int L    = j * THREADS + tid;
        const int slot = L >> 4;      // V 每 slot 16 个 int4
        const int w    = L & 15;
        *reinterpret_cast<int4*>(sV + slot * SV_STRIDE + w * 8) =
            src[(size_t)slot * (heads * 16) + w];
      }
#endif
#endif
    }
    __syncthreads();

    const float k_scale  = __ldg(KScale + (size_t)head * n_kblocks + kblk);
    const float qk_scale = q_scale * k_scale;
    const int   valid    = __ldg(Valid + kblk);

    // ---- QK：si[mt][nt] = 16 行 x 8 key 的 int32 点积 -------------------
    int32_t si[2][NNT][2];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
      for (int nt = 0; nt < NNT; nt++) { si[mt][nt][0] = 0; si[mt][nt][1] = 0; }

#pragma unroll
    for (int ks = 0; ks < NKSTEP; ks++) {
      uint32_t kb[NNT];
#pragma unroll
      for (int nt = 0; nt < NNT; nt++) {
        // B[t*4+i][g] = K[key = 8nt+g][dim = 16ks + 4t + i]  ⇒ 4 个连续 int8
#if VEDA_STUB_K
        kb[nt] = (uint32_t)(tid * 2654435761u) + (uint32_t)nt;
#else
        const int key = nt * 8 + g;
        kb[nt] = *reinterpret_cast<const uint32_t*>(
            sK + key * SK_STRIDE + ks * 16 + t * 4);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NNT; nt++) {
        mma_s8_m8n8k16(si[0][nt][0], si[0][nt][1],
                       (uint32_t)qreg[0][ks], kb[nt]);
        mma_s8_m8n8k16(si[1][nt][0], si[1][nt][1],
                       (uint32_t)qreg[1][ks], kb[nt]);
      }
    }

#if VEDA_C4_FUSE && !VEDA_STUB_SOFT && !VEDA_STUB_CONV
    // ---- 融合版：不落 sf / pa，两趟直接吃 si -----------------------------
    // 第一趟：只求行 max（掩码在这里做，因为 max 必须排除被掩掉的位置）。
    float mm0 = -1e30f, mm1 = -1e30f;
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
      for (int nt = 0; nt < NNT; nt++) {
        float a = (float)si[mt][nt][0] * qk_scale;
        float b = (float)si[mt][nt][1] * qk_scale;
        const int k0 = nt * 8 + 2 * t;
        if (k0 >= valid)     a = -INFINITY;
        if (k0 + 1 >= valid) b = -INFINITY;
        if (mt == 0) { mm0 = fmaxf(mm0, a); mm0 = fmaxf(mm0, b); }
        else         { mm1 = fmaxf(mm1, a); mm1 = fmaxf(mm1, b); }
      }
    }
    mm0 = row_reduce_max(mm0);
    mm1 = row_reduce_max(mm1);

    const float m_new0 = fmaxf(m0, mm0);
    const float m_new1 = fmaxf(m1, mm1);
    const float alpha0 = exp2f(m0 - m_new0);
    const float alpha1 = exp2f(m1 - m_new1);

    l0 *= alpha0;
    l1 *= alpha1;
#pragma unroll
    for (int dt = 0; dt < NDT_W; dt++) {
      acc[dt][0] *= alpha0; acc[dt][1] *= alpha0;
      acc[dt][2] *= alpha1; acc[dt][3] *= alpha1;
    }

    // 第二趟：算 P -> 求和 -> 打包 -> 当场做这一 nt 的 16 次 PV mma。
    // si 在这里被**再读一遍**，所以它必须活过第一趟 —— 这是用 32 个 si 寄存器
    // 换掉 sf(32) + pa(16) 的净赚。
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < NNT; nt++) {
      const int k0 = nt * 8 + 2 * t;
      float x00 = (float)si[0][nt][0] * qk_scale;
      float x01 = (float)si[0][nt][1] * qk_scale;
      float x10 = (float)si[1][nt][0] * qk_scale;
      float x11 = (float)si[1][nt][1] * qk_scale;
      if (k0 >= valid)     { x00 = -INFINITY; x10 = -INFINITY; }
      if (k0 + 1 >= valid) { x01 = -INFINITY; x11 = -INFINITY; }
      const float p00 = exp2f(x00 - m_new0);
      const float p01 = exp2f(x01 - m_new0);
      const float p10 = exp2f(x10 - m_new1);
      const float p11 = exp2f(x11 - m_new1);
      sum0 += p00 + p01;
      sum1 += p10 + p11;

      const __half2 h0 = __floats2half2_rn(p00, p01);
      const __half2 h1 = __floats2half2_rn(p10, p11);
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&h0);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&h1);
#pragma unroll
      for (int dt = 0; dt < NDT_W; dt++) {
        mma_f16_m16n8k8(acc[dt][0], acc[dt][1], acc[dt][2], acc[dt][3],
                        a0, a1, v_frag(sV, nt, dg * NDT_W + dt, g, t));
      }
    }
    sum0 = row_reduce_sum(sum0);
    sum1 = row_reduce_sum(sum1);
    l0 += sum0;
    l1 += sum1;
    m0 = m_new0;
    m1 = m_new1;

    __syncthreads();   // 本块读完，下一轮才能覆盖 sK / sV
    continue;
#else
    // ---- 掩码 + 缩放 + online softmax ----------------------------------
    // 线程持行 g（mt=0）与行 g+8（mt=1）的 key {8nt+2t, 8nt+2t+1}
    float sf[2][NNT][2];
    uint32_t pa[2][NNT];        // PV 的 A 片段：pa[mt][nt] = {P[..][8nt+2t], P[..][8nt+2t+1]}
    float sum0, sum1;

#if VEDA_STUB_SOFT || VEDA_STUB_CONV
    // ⚠️ 计时探针，数值无意义：跳过掩码 / max / exp2 / 求和 / 重标定
    const float m_new0 = m0, m_new1 = m1;
    const float alpha0 = 1.f, alpha1 = 1.f;
    sum0 = 1.f; sum1 = 1.f;
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
      for (int nt = 0; nt < NNT; nt++) {
#if VEDA_STUB_CONV
        sf[mt][nt][0] = 0.5f; sf[mt][nt][1] = 0.5f;
#else
        // 保留 I2F + 缩放，QK mma 才不会被 DCE
        sf[mt][nt][0] = (float)si[mt][nt][0] * qk_scale;
        sf[mt][nt][1] = (float)si[mt][nt][1] * qk_scale;
#endif
      }
#if VEDA_STUB_CONV
    // 恒假条件，只为把 si 留活
    if (si[0][0][0] == 0x7fffffff) sf[0][0][0] = 1.f;
#endif
#pragma unroll
    for (int nt = 0; nt < NNT; nt++) {
      const __half2 h0 = __floats2half2_rn(sf[0][nt][0], sf[0][nt][1]);
      const __half2 h1 = __floats2half2_rn(sf[1][nt][0], sf[1][nt][1]);
      pa[0][nt] = *reinterpret_cast<const uint32_t*>(&h0);
      pa[1][nt] = *reinterpret_cast<const uint32_t*>(&h1);
    }
#else
    float mm0, mm1;
#if VEDA_C3_MASKHOIST
    // valid 在本 block 内均匀（同一地址的 __ldg）⇒ 这个分支不发散。
    // 全块有效的块走 <false>，64 条 setp/sel 被整段删掉。
    if (valid < KEY_BLOCK) score_stage<true >(si, sf, mm0, mm1, qk_scale, valid, t);
    else                   score_stage<false>(si, sf, mm0, mm1, qk_scale, valid, t);
#else
    score_stage<true>(si, sf, mm0, mm1, qk_scale, valid, t);
#endif
    mm0 = row_reduce_max(mm0);
    mm1 = row_reduce_max(mm1);

    const float m_new0 = fmaxf(m0, mm0);
    const float m_new1 = fmaxf(m1, mm1);
    const float alpha0 = exp2f(m0 - m_new0);
    const float alpha1 = exp2f(m1 - m_new1);

#if VEDA_C3_EX2F16
    prob_stage<true >(sf, pa, sum0, sum1, m_new0, m_new1);
#else
    prob_stage<false>(sf, pa, sum0, sum1, m_new0, m_new1);
#endif
#endif  // VEDA_STUB_SOFT || VEDA_STUB_CONV

    sum0 = row_reduce_sum(sum0);
    sum1 = row_reduce_sum(sum1);
    l0 = l0 * alpha0 + sum0;
    l1 = l1 * alpha1 + sum1;
    m0 = m_new0;
    m1 = m_new1;

#if VEDA_ACCFAST && !VEDA_STUB_SOFT
    // alpha = exp2(m_old - m_new)：m 没长就是 1.0，64 次 fmul 全是白乘。
    // 逐 lane 判断没用（SASS 只会谓词化，发射槽照占），必须**整 warp 一把**：
    // __all_sync 为真 ⇒ 这一块的 16 行最大值全都没变 ⇒ 整段跳过。
    // 数值与无条件版完全等价（乘 1.0 是精确的）。
    // 注意 __all_sync 的返回值在 warp 内是**统一**的，所以这个分支不发散。
    const bool m_stable = (alpha0 == 1.f) & (alpha1 == 1.f);
    const bool unstable = (__all_sync(0xffffffffu, m_stable) == 0);
#ifdef VEDA_ACCFAST_STAT
    if (lane == 0) {
      atomicAdd(&g_accfast_total, 1ull);
      if (!unstable) atomicAdd(&g_accfast_skip, 1ull);
    }
#endif
    if (unstable)
#endif
#if !VEDA_STUB_SOFT
    {
#pragma unroll
      for (int dt = 0; dt < NDT_W; dt++) {
        acc[dt][0] *= alpha0; acc[dt][1] *= alpha0;   // 行 g
        acc[dt][2] *= alpha1; acc[dt][3] *= alpha1;   // 行 g+8
      }
    }
#endif

    // ---- PV：p(fp16) x V(fp16) -> fp32 累加 -----------------------------
    // A 片段 = 两个 QK C 片段直接拼（sf 的 4 个 float -> 2 个 .b32），零 shuffle
#pragma unroll
    for (int nt = 0; nt < NNT; nt++) {
      const uint32_t a0 = pa[0][nt];
      const uint32_t a1 = pa[1][nt];
#if VEDA_C3_PVPIPE
      // 先把这一 nt 的 NDT_W 个 dt 片段全载进寄存器，再做 NDT_W 次 mma ——
      // 把「载入到使用」的窗口显式撑开，而不是每条 mma 现等自己的 LDS。
      uint32_t bb[NDT_W];
#pragma unroll
      for (int dt = 0; dt < NDT_W; dt++)
        bb[dt] = v_frag(sV, nt, dg * NDT_W + dt, g, t);
#pragma unroll
      for (int dt = 0; dt < NDT_W; dt++) {
        mma_f16_m16n8k8(acc[dt][0], acc[dt][1], acc[dt][2], acc[dt][3],
                        a0, a1, bb[dt]);
      }
#else
#pragma unroll
      for (int dt = 0; dt < NDT_W; dt++) {
        const uint32_t b0 = v_frag(sV, nt, dg * NDT_W + dt, g, t);
        mma_f16_m16n8k8(acc[dt][0], acc[dt][1], acc[dt][2], acc[dt][3],
                        a0, a1, b0);
      }
#endif
    }

#endif  // VEDA_C4_FUSE

    __syncthreads();   // 本块读完，下一轮才能覆盖 sK / sV
  }

  // ---- 归一化 + 写回 ---------------------------------------------------
  // 对角线永远被保留，真实行必然 l > 0；这里仍守一手防畸形 mask 除零。
  const float inv0 = (l0 > 0.f) ? (1.f / l0) : 1.f;
  const float inv1 = (l1 > 0.f) ? (1.f / l1) : 1.f;

  const int row0 = qtile * TILE + rg * ROWS_PER_WARP + g;
  const int row1 = row0 + 8;
  __half* o0 = Out + (size_t)row0 * heads * HEAD_DIM + (size_t)head * HEAD_DIM;
  __half* o1 = Out + (size_t)row1 * heads * HEAD_DIM + (size_t)head * HEAD_DIM;

#pragma unroll
  for (int dt = 0; dt < NDT_W; dt++) {
    const int d = (dg * NDT_W + dt) * 8 + 2 * t;   // 恒为偶数 ⇒ 4 B 对齐
    __half2 ha, hb;
    ha.x = __float2half(acc[dt][0] * inv0);
    ha.y = __float2half(acc[dt][1] * inv0);
    hb.x = __float2half(acc[dt][2] * inv1);
    hb.y = __float2half(acc[dt][3] * inv1);
    *reinterpret_cast<__half2*>(o0 + d) = ha;
    *reinterpret_cast<__half2*>(o1 + d) = hb;
  }
}

}  // namespace

// ---------------------------------------------------------------------------

torch::Tensor sparse_int8_attention_mma(
    const torch::Tensor& q_i8,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_i8,
    const torch::Tensor& k_scale,
    const torch::Tensor& v16,
    const torch::Tensor& index,
    const torch::Tensor& count,
    const torch::Tensor& valid)
{
  TORCH_CHECK(q_i8.is_cuda() && q_i8.is_contiguous(),
              "q must be contiguous CUDA");
  TORCH_CHECK(q_i8.scalar_type() == torch::kChar, "q must be int8");
  TORCH_CHECK(q_i8.dim() == 3, "q must be [N, H, D]");
  TORCH_CHECK(q_i8.size(2) == HEAD_DIM, "D must be 128");
  TORCH_CHECK(v16.scalar_type() == torch::kHalf, "v must be fp16");

  const int64_t slots = q_i8.size(0);
  TORCH_CHECK(slots % TILE == 0, "N must be a multiple of 128");
  const int heads      = (int)q_i8.size(1);
  const int n_qtiles   = (int)(slots / TILE);
  const int n_kblocks  = (int)(slots / KEY_BLOCK);

  auto out = torch::empty({slots, (int64_t)heads, (int64_t)HEAD_DIM},
                          q_i8.options().dtype(torch::kHalf));

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n_qtiles, heads);
  sparse_int8_mma_kernel<<<grid, THREADS, 0, stream>>>(
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
  m.def("sparse_int8_attention", &sparse_int8_attention_mma,
        "Veda block-sparse INT8 attention (sm75 tensor cores)");
#ifdef VEDA_ACCFAST_STAT
  // 诊断入口（只在统计构建里存在）：清零 -> 自己跑一次内核 -> 读回。
  // 三段都挂在当前 CUDA 流上，reset 与 read 之间的内核严格有序。
  // 刻意用两个无参函数而不是「传一个 callable 进来」，免得再拉 pybind11 的头依赖。
  m.def("accfast_reset", []() {
    auto stream = at::cuda::getCurrentCUDAStream();
    const unsigned long long zero = 0ull;
    cudaMemcpyToSymbolAsync(g_accfast_total, &zero, sizeof(zero), 0,
                            cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_accfast_skip, &zero, sizeof(zero), 0,
                            cudaMemcpyHostToDevice, stream);
  });
  m.def("accfast_read", []() {
    auto stream = at::cuda::getCurrentCUDAStream();
    unsigned long long h[2] = {0ull, 0ull};
    cudaMemcpyFromSymbolAsync(&h[0], g_accfast_total, sizeof(h[0]), 0,
                              cudaMemcpyDeviceToHost, stream);
    cudaMemcpyFromSymbolAsync(&h[1], g_accfast_skip, sizeof(h[1]), 0,
                              cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return pybind11::make_tuple(h[0], h[1]);
  });
#endif
}
