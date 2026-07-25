# Phase 0 — the gap, measured

**Run:** [actions/runs/30148951999](https://github.com/Marc-Dvci/fastpath64/actions/runs/30148951999) ·
upstream `555881e` · `llama-bench -r 5 -t 4` · raw CSV + `lscpu` in the run artifacts.

**Silicon** (from the runner's own `lscpu`, `results/raw/phase0-n2-stock/hardware-n2-stock.txt`):

```
Model name: Neoverse-N2
Flags: ... asimddp sve sve2 ... svei8mm svebf16 i8mm bf16
```

The I8MM unit is present. Hold that thought.

## The controlled experiment

Three builds of the *same* pinned upstream commit on the *same* Neoverse N2 runner, differing only
in `-DGGML_CPU_REPACK` and `-DGGML_CPU_KLEIDIAI`. Same GGUF files throughout.

### Llama-3.2-3B (dense), prefill `pp512`, tokens/s

| build | Q4_K_M | IQ4_XS |
|---|---:|---:|
| stock (repack ON) | **42.78** | **25.08** |
| repack OFF | 27.65 | 24.94 |
| KleidiAI ON | 42.78 | 25.01 |
| **what the fast path is worth** | **1.55x** | **1.01x** |

Read the last row twice. Turning llama.cpp's Arm fast path off costs Q4_K **35% of its prefill
throughput**. It costs IQ4_XS **0.6%** — which is smaller than the run-to-run spread (±0.08).

IQ4_XS is not merely slower. **It is not on the fast path at all**, and the toggle proves it: you
cannot lose what you never had. That is the difference between a benchmark observation and a
demonstrated mechanism.

Enabling KleidiAI — Arm's own microkernel library — changes **nothing** for either format
(42.78 → 42.78, 25.08 → 25.01), because it only accepts Q4_0 and Q8_0.

### OLMoE-1B-7B (MoE), prefill `pp512`, tokens/s

| build | Q4_K_M | IQ4_XS |
|---|---:|---:|
| stock (repack ON) | **72.16** | **65.98** |
| repack OFF | 55.88 | 66.51 |
| **what the fast path is worth** | **1.29x** | **0.99x** |

Same signature on MoE expert matmuls. And note the second column of the `repack OFF` row: with both
formats on the generic path, **IQ4_XS is 19% _faster_ than Q4_K** (66.51 vs 55.88). The format is
intrinsically fine. It loses only because someone wrote a kernel for one and not the other.

### Decode (`tg128`) — the honest column

| model | Q4_K | IQ4_XS |
|---|---:|---:|
| Llama-3.2-3B | 16.21 | 17.00 |
| OLMoE-1B-7B | 43.39 | 42.94 |

Flat, and unaffected by the repack toggle. Batch-1 decode is memory-bandwidth-bound; no matmul
kernel changes that. **This project will not claim a decode speedup**, and this table is here so
that promise is auditable.

## What it costs today

On this 4-vCPU Neoverse N2 instance, choosing IQ4_XS over Q4_K to fit a bigger model in less RAM
costs **41% of prefill throughput** on the dense model (25.08 vs 42.78 t/s). For an agent turn —
which is overwhelmingly prefill, since a tool-calling request re-sends a large system prompt, tool
schemas and history to emit a few dozen tokens — that is close to a straight 1.7x on wall-clock
time to first token.

## Estimated ceiling for P1

If a repacked IQ4_XS kernel reaches the same efficiency relative to its generic path that Q4_K's
does, the expected landing zone is:

| model | today | projected | uplift |
|---|---:|---:|---:|
| Llama-3.2-3B `pp512` | 25.08 | ~38.5 | **~1.54x** |
| OLMoE-1B-7B `pp512` | 65.98 | ~85.9 | **~1.30x** |

Derived by scaling each format's generic-path ratio (IQ4_XS/Q4_K with repack OFF: 0.90x dense,
1.19x MoE) onto Q4_K's repacked throughput. These are projections, not results — they exist to be
falsified by the P1 branch. Note they are **below** the 1.8–2.5x the project plan originally
guessed; the measurement moved the target down and the target moved.

The MoE figure is the conservative one: OLMoE has 1B active parameters, so its expert matrices are
small and GEMM efficiency is limited. Larger MoEs (gemma-4-26B-A4B, Qwen3.6-35B-A3B) have much
bigger expert matrices and should gain more — that is a P2 question.

## x86 reference

The x86 job landed on an **AMD EPYC 9V74**, which has AVX-512 but no AMX. It shows the same
qualitative pattern (IQ4_XS at 0.52x of Q4_K on `pp512`), since IQ4_XS is absent from the repack
table on every architecture.

One clarification, stated plainly: `qtype_has_amx_kernels()` in `ggml/src/ggml-cpu/amx/common.h:106`
lists `GGML_TYPE_IQ4_XS`, so **Intel** AMX hardware has a tiled path for this format. That is a
source-level fact about upstream, **not something measured here** — no AMX-capable machine was
available, and none is claimed. What *is* measured is everything above: on Neoverse N2, with an
I8MM unit sitting right there in the flags, IQ4_XS issues no `smmla` at all.

## Verdict

Gate passed. The gap is real, the mechanism is proven by the toggle, and the ceiling is worth the
work. Proceed to P1.
