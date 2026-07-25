# P1 results — IQ4_XS on Arm's fast path

**Run:** [actions/runs/30156929580](https://github.com/Marc-Dvci/fastpath64/actions/runs/30156929580) ·
upstream `555881e` + [patches](../patches) · `llama-bench -r 5 -t 4`

Stock and patched were built **in the same job on the same physical runner**, so the comparison
carries no machine-to-machine variance. Timings were gated behind the correctness check below;
the workflow refuses to report numbers if it fails.

**Silicon** — GitHub's free `ubuntu-24.04-arm`, from its own `lscpu`:

```
Model name: Neoverse-N2
Features:   ... asimddp sve sve2 ... svei8mm svebf16 i8mm bf16
```

## Prefill

| model | quant | case | stock t/s | FastPath64 t/s | change |
|---|---|---|---:|---:|---:|
| Llama-3.2-3B | **iq4_xs** | pp512 | 24.95 ±0.02 | **52.85 ±0.02** | **2.12x** |
| Llama-3.2-3B | **iq4_xs** | pp2048 | 17.72 ±0.01 | **28.31 ±0.01** | **1.60x** |
| OLMoE-1B-7B (MoE) | **iq4_xs** | pp512 | 66.11 ±0.04 | **74.71 ±0.09** | **1.13x** |
| OLMoE-1B-7B (MoE) | **iq4_xs** | pp2048 | 46.80 ±0.02 | **51.03 ±0.04** | **1.09x** |
| Llama-3.2-3B | q4_k *(control)* | pp512 | 42.70 ±0.06 | 42.74 ±0.04 | 1.00x |
| Llama-3.2-3B | q4_k *(control)* | pp2048 | 25.11 ±0.01 | 25.09 ±0.05 | 1.00x |
| OLMoE-1B-7B | q4_k *(control)* | pp512 | 72.18 ±0.04 | 72.08 ±0.14 | 1.00x |
| OLMoE-1B-7B | q4_k *(control)* | pp2048 | 49.70 ±0.02 | 49.69 ±0.02 | 1.00x |

**The controls land at 1.00x, to two decimal places, on all four cases.** Q4_K is a format this
work does not touch. If the patched build were faster for some incidental reason — different
compiler behaviour, a warmer cache, a quieter neighbour — the controls would have moved too. They
did not, so what the IQ4_XS rows measure is the kernel.

### The result worth stating plainly

Before this work, IQ4_XS ran prefill at **0.58x of Q4_K** despite being the smaller format
(24.95 vs 42.70 t/s). It now runs at **1.24x of Q4_K** (52.85 vs 42.70).

> The format the MoE ecosystem standardised on went from the slowest 4-bit option on Arm to the
> fastest — and it was already the smallest.

It ends up ahead of Q4_K because IQ4_XS carries no min/`dmin` term: one 6-bit scale covers each
32-element sub-block, so the kernel does strictly less work per byte than the Q4_K equivalent.

### Why the MoE gain is smaller

The first explanation I reached for — small expert matrices — does not survive the arithmetic.

A dense prefill reads each weight once and reuses it across all T tokens in the batch. A MoE
prefill reads **every** expert's weights while computing only the active fraction, so its
arithmetic intensity is lower by roughly the sparsity ratio. OLMoE is 1B active of 7B total, so its
expert matmuls run at about **1/7th** the FLOPs-per-byte of a dense model at the same batch size.
Lower intensity means more bandwidth-bound, and a compute kernel cannot help what is waiting on
memory.

That reframes the expectation for bigger models rather than raising it. gemma-4-26B-A4B is 4B
active of 26B — a sparsity ratio of 6.5, essentially OLMoE's 7. On the intensity argument alone it
should gain about as much, not more. Pulling the other way, its expert matrices are far larger,
which amortises GEMM setup better.

Two effects in opposite directions is not a prediction I can make from the armchair, so it is
being measured: `bench-bigmoe.yml` runs gemma-4-26B-A4B IQ4_XS on the same free runner. At 12.7 GB
it does fit in 16 GB — just. Qwen3.6-35B-A3B is 17.4 GB at IQ4_XS and genuinely does not fit, and
no number will be claimed for it.

The same argument explains why *both* dense and MoE gains fall from `pp512` to `pp2048`: attention
grows with sequence length and is untouched by this work, so the FFN's share of the total shrinks.

## Decode

| model | quant | case | stock t/s | FastPath64 t/s | change |
|---|---|---|---:|---:|---:|
| Llama-3.2-3B | iq4_xs | tg128 | 16.78 ±0.02 | 16.22 ±0.10 | **0.97x** |
| OLMoE-1B-7B | iq4_xs | tg128 | 43.21 ±0.23 | 44.51 ±0.29 | 1.03x |
| Llama-3.2-3B | q4_k *(control)* | tg128 | 16.45 ±0.06 | 16.25 ±0.02 | 0.99x |
| OLMoE-1B-7B | q4_k *(control)* | tg128 | 42.69 ±0.18 | 42.22 ±0.18 | 0.99x |

Decode is memory-bandwidth-bound and no matmul kernel changes that — as predicted before any of
this was written.

**There is a small regression on dense decode: 0.97x, about 3%.** The spread (±0.02 / ±0.10) does
not cover it, so it is real rather than noise. The repacked GEMV carries per-sub-block scale
decoding that the existing `vec_dot` path does not, and at batch 1 there are only 8 columns of
output to amortise it over — a quarter of what the GEMM enjoys. Note the Q4_K control also sits at
0.99x, so part of the movement is run-level and not attributable to the kernel.

Traded against 2.12x on prefill this is a good deal for any prefill-heavy workload, which is what
agent traffic is. It is still a regression, it is not hidden, and it is being worked on — the
scale bytes can be repacked sub-block-major as 8x8 six-bit values, an exact 48-byte fit, making
the decode contiguous and vectorisable.

## Correctness, on the same silicon

Run before any timing, on real Neoverse N2:

```
type=iq4_xs N=64  K=512  M=1   max_abs=3.815e-06  PASS
type=iq4_xs N=64  K=512  M=5   max_abs=3.815e-06  PASS
type=iq4_xs N=128 K=1024 M=9   max_abs=7.629e-06  PASS
type=iq4_xs N=8   K=256  M=16  max_abs=0.000e+00  PASS
```

M=1 exercises GEMV only, M=8/16 GEMM only, M=5/9 both plus remainder handling. These match the
QEMU figures to the digit, which is a useful cross-check on the emulation harness itself.

## What this means for an agent turn

An agent turn is overwhelmingly prefill: a large system prompt, tool schemas, history and
retrieved context go in; a few dozen tokens of JSON come out. Dividing out the measured pp2048
rates, a 2048-token agent prompt in IQ4_XS costs about **115 s** of prefill on stock and **72 s**
patched on this 4 vCPU instance — derived from the throughput above, not separately timed.

The end-to-end agent-turn measurement (`bench/agentbench.py`) landed after this run and is wired
into the workflow for the next one.
