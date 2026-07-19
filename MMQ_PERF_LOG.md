# MMQ perf log

Hill-climbing a W4A16 (Marlin-style, w4 weights x f16 activations, f16.f16.f32 tensor cores)
GEMM for Q4_K on NVIDIA GB10 (DGX Spark, sm_121), vs the int8 MMQ baseline.

Benchmark: `llama-bench -m models/gemma-4-12B-it-Q4_K_M.gguf -p 2048 -n 0 -r 5` (pp2048, tokens/s).

NOTE on baselines: the first baseline measurement (1561 t/s) was taken while the local machine
was contended; the honest int8 MMQ baseline is ~1950 t/s (local 1958, spark-a934 1935).

Dashboard target: 2500 tokens/s

## W4A16 sidecar experiment (July 17, 2026)

| revision | variant | throughput | relative | result |
| --- | --- | --- | --- | --- |
| baseline | int8 MMQ, contended measurement (superseded) | 1561.48 t/s | 1.000x | reference |
| v17 | w4a16 swizzle (previous state) | 1697.23 t/s | 1.087x | accepted |
| v18 | M_TILE 64->32: 4 blocks/SM, phase decorrelation | 2072.25 t/s | 1.327x | accepted |
| v19 | per-shape cfg dispatch + N_MAX lift (up/gate w4a16) | 2063.25 t/s | 1.321x | accepted |
| v20 | w6a16: Q6_K (ffn_down/attn_v) via f16 mma | 2209.76 t/s | 1.415x | accepted |
| v21 | w6 cfg m64w64s3 + logits->int8 + convert f32x4 vectorize | 2261.87 t/s | 1.449x | accepted |

## Honest baseline (re-measured, uncontended)

| revision | variant | throughput | relative | result |
| --- | --- | --- | --- | --- |
| baseline2 | int8 MMQ (local, uncontended) | 1958.38 t/s | 1.000x | reference |
| baseline2r | int8 MMQ (spark-a934) | 1934.98 t/s | 0.988x | reference |
| v21 | current tree (spark-a934) | 2257.84 t/s | 1.145x | accepted |

## Findings (July 19, 2026)

Headline: tensor-core utilization 28.1% (int8 MMQ) -> 52-58% (w4a16/w6a16), measured with ncu
`sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed`.

Hardware facts (GB10, sm_121, 48 SMs):
- FP16 tensor peak (f16.f16.f32 HMMA m16n8k16): ~125.5 TFLOPS (microbenchmark, saturates at 8 warps/SM).
- smem: 100KB/SM optin, 64K regs/block. cuBLAS f16xfxf32 on our shapes: 75-84 TF.

Kernel-level TFLOPS (test-backend-ops perf, m=512), int8 MMQ -> current:
- attn 3840x3840 (Q4_K): 57.7 -> 75.9 (1.32x), cfg m32w32s3
- up/gate 15360x3840 (Q4_K): 65.5 -> 69.0 (1.05x), cfg m128w64s3
- down 3840x15360 (Q4_K): 43.9 -> 70.7 (1.61x), cfg m32w32s4
- ffn_down 3840x15360 (Q6_K): 33.2 -> 59.0 (1.78x), w6a16 cfg m64w64s3
- attn_v 2048x3840 (Q6_K): 42.2 -> 56.1 (1.33x), w6a16 cfg m64w64s3
- logits 262144x3840 (Q6_K): 55.6 -> stays int8 (w6a16 44.0, routed back)

What worked:
- M_TILE=32 (4 blocks/SM decorrelates ALU/HMMA phases): the single biggest kernel win.
- Per-shape config dispatch (M_TILE/M_WARP/STAGES by n,k,m). N_TILE=256 is sidecar-fixed.
- w6a16 for Q6_K: 6-bit sidecar (ql4+qh2 interleaved), ~2x int8 on ffn_down.
- Removing back-to-back dependent HMMAs on the same accumulator (k16 outer loop).
- depth-2 B fragment prefetch pipeline (b[3] rotation).

What did NOT work (rejected, kept for the record):
- K_STAGE=64 (v2): occupancy loss beats barrier savings (smem wall at 100KB).
- Cross-group register pipelining (v3): swap-copy overhead + fewer HMMA/group.
- Warp-autonomous barrier-free pipeline (v4, A via LDG): LDG latency exposure, loses to cp.async+barrier.
- f32-direct A in kernel (v5): same LDG latency problem; separate convert kernel stays.
- N_WARP=32 any variant: HMMA:dequant ratio dominates everything.
- L2 policies (sidecar streaming, activation persisting): ~0 or crash.
- STAGES>3: flat.

Current e2e breakdown per pp2048 (~910ms at 2258 t/s): up/gate w4a16 ~350ms (69 TF, 39%),
w6a16 ~104ms, attn w4a16 ~102ms, down w4a16 ~86ms, logits int8 ~74ms, convert f32->f16 ~52ms,
gelu ~74ms, flash attn ~33ms, rms_norm ~43ms.

Next candidate levers:
- up/gate: stuck at ~69 TF (latency-bound at 58% TC). Fusing w1+w2 (same input) into one GEMM
  would halve activation rereads and double per-block work; needs graph-level changes.
- stream-k / persistent CTA for wave tails (attn/down grids are 1.25 waves).
- convert (52ms) and gelu (74ms) are memory-bound at peak; need f16 producer dtypes (graph surgery).
- TMA + mbarrier producer/consumer (sm_121) as a cp.async+barrier replacement.

