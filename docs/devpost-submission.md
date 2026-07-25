# Devpost submission text — Track 2, Cloud AI

Paste-ready. Numbers are from CI runs linked inline; refresh them if a later run supersedes.

---

## Project Overview

**IQ4_XS — the 4-bit format nearly every MoE GGUF ships in — had an accelerated matmul kernel on
Intel AMX and none on Arm. FastPath64 writes the missing kernels. Prefill on Neoverse N2 goes up
2.12x, on the same GGUF file, with no change to the model.**

I did not start from an idea; I started from a `grep`:

```
$ grep -c IQ4_XS ggml/src/ggml-cpu/repack.cpp      # Arm's row-interleaved smmla kernels
0
$ grep -n IQ4_XS ggml/src/ggml-cpu/amx/common.h    # Intel AMX tiled kernels
114:        (type == GGML_TYPE_IQ4_XS);
```

llama.cpp's Arm fast path (`repack.cpp`) covers Q4_0 and the K-quants. KleidiAI, Arm's own
microkernel library, covers Q4_0 and Q8_0 and only `MUL_MAT` — not `MUL_MAT_ID`, the op every MoE
expert matmul goes through. IQ4_XS appears in neither. So a Qwen3.6-35B-A3B or gemma-4-26B-A4B in
IQ4_XS, served on Graviton or Cobalt, ran its expert matmuls without issuing a single `smmla` —
on a core whose `lscpu` advertises `i8mm`.

There is a vicious loop here worth naming. On a cost-efficient Arm instance you want the biggest
model in the least RAM, so you reach for the most aggressive 4-bit format. That was precisely the
format with no Arm kernel. **The choice you made to fit the instance was the choice that gave back
the performance you were trying to buy.**

**Why it should win.** The challenge asks for optimization with measurable improvement, not an app
that runs on Arm. This is a kernel contribution to the framework Arm's own Cloud AI learning paths
use, aimed at the exact format and architecture the 2026 ecosystem standardised on, and it is the
direct sequel to work Arm's own engineers published — their AI blog took `smmla` to Q4_K/Q6_K on
Neoverse N2 and stopped there. Every number is reproducible by clicking "Run workflow" on a public
repo, at zero cost, on free hardware.

## Functionality / Output

**Output: three kernels and a patch series against upstream llama.cpp**, plus the harness that
proves them.

| path | instruction | hardware covered |
|---|---|---|
| GEMM | `smmla` (I8MM) | Graviton3/4, Cobalt 100, Axion |
| GEMM | `sdot` (DOTPROD) | Graviton2, Ampere Altra |
| GEMV | `sdot` (DOTPROD) | all of the above |

IQ4_XS is IQ4_NL's 16-entry codebook carrying K-quant style super-block scales — the missing kernel
was the *intersection of two kernels Arm already shipped*, which is why it fell between two owners
and why it is straightforwardly upstreamable.

**Measured** on GitHub's free Neoverse N2 runners, stock and patched built in the same job on the
same physical machine:

| Llama-3.2-3B, IQ4_XS | stock | FastPath64 |
|---|---:|---:|
| prefill `pp512` | 24.95 ±0.02 t/s | **52.85 ±0.02 t/s — 2.12x** |
| prefill `pp2048` | 17.72 ±0.01 t/s | **28.31 ±0.01 t/s — 1.60x** |
| Q4_K control | 42.70 ±0.06 t/s | 42.74 ±0.04 t/s — 1.00x |

IQ4_XS ran prefill at **0.58x of Q4_K**; it now runs at **1.24x of Q4_K**, while remaining the
smaller file. The format went from the slowest 4-bit option on Arm to the fastest.

**The control is the point.** Q4_K is untouched by this work and lands at 1.00x to two decimals on
all four cases. Had the patched build been faster for any incidental reason — compiler luck, cache
warmth, a quieter neighbour — the control would have moved too.

**What I do not claim:** decode does not get faster (it is memory-bandwidth-bound, measured
0.97–1.03x); output is numerically equivalent, not bit-identical, since `smmla` accumulates in a
different order; no numbers for large MoEs, which do not fit a free runner's 16 GB; and no AMX
measurement, since the AMX fact is read from upstream source rather than benchmarked.

**Reusable artifacts:**
- Patch series, and a pushed branch ready for upstream review.
- `test_repack_equiv.cpp` — an equivalence gate for repacked kernels. Upstream's `test-backend-ops`
  allocates into the *default* CPU buffer and therefore never exercises the repack path at all,
  which is plausibly why this gap went unnoticed. It also caught a latent upstream null-deref.
- A QEMU cross-build harness reaching dispatch paths no free CI runner offers, including the
  Ampere Altra / Graviton2 `sdot` path — no cloud account, no Arm hardware.
- `agentbench.py` — agent-turn latency (long tool-calling prompt, short JSON reply) rather than
  synthetic throughput.
- `docs/the-gap.md` — the source-level audit, with file:line citations, of what Arm's two fast
  paths actually accept.

## Setup Instructions

**Reproduce the benchmarks on Arm (no account, no spend):** on the public repo, go to
Actions → *A/B - stock vs FastPath64 on Neoverse N2* → **Run workflow**. It clones the pinned
upstream twice, patches one, builds both on the same free Neoverse N2 runner, runs the correctness
gate, and refuses to report timings if that gate fails.

**Reproduce correctness locally on an x86 laptop** — cross-compiles at native speed, executes under
QEMU, covering N2 (`smmla`) and N1 (`sdot`) dispatch paths:

```bash
git clone https://github.com/Marc-Dvci/fastpath64 && cd fastpath64
docker build -t fastpath64-cross tools/qemu/
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git ../upstream-llama.cpp
git -C ../upstream-llama.cpp checkout "$(cat UPSTREAM_SHA)"
for p in patches/*.patch; do git -C ../upstream-llama.cpp apply "$p"; done
docker run --rm -v "$PWD/..:/src" fastpath64-cross bash /src/fastpath64/tools/qemu/run-equiv-test.sh
```

Expect 24 PASS lines: 6 shapes × {iq4_xs, q4_K} × {neoverse-n2, neoverse-n1}.

**Build on a real Arm64 server:**

```bash
cmake -S ../upstream-llama.cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench -j"$(nproc)"
build/bin/llama-bench -m <model>-IQ4_XS.gguf -p 512 -n 128 -t "$(nproc)" -r 5
```

No flags needed: dispatch is automatic from runtime CPU feature detection, gated at DOTPROD.

Repo: https://github.com/Marc-Dvci/fastpath64 · Licence: MIT
