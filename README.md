# FastPath64

**IQ4_XS — the 4-bit format nearly every MoE GGUF ships in — had an accelerated matmul kernel on Intel AMX and none on Arm.**

That was not a claim about benchmarks. It is what upstream `llama.cpp` says about itself:

```
$ grep -c IQ4_XS ggml/src/ggml-cpu/repack.cpp          # Arm's row-interleaved smmla kernels
0
$ grep -n IQ4_XS ggml/src/ggml-cpu/amx/common.h        # Intel AMX tiled kernels
114:        (type == GGML_TYPE_IQ4_XS);
```

So a Qwen3.6-35B-A3B or gemma-4-26B-A4B in IQ4_XS, served on a Graviton / Cobalt / Axion instance,
ran its expert matmuls with **zero I8MM acceleration** — locked out of the repack path (type not in
the table) *and* out of KleidiAI (which supports neither `MUL_MAT_ID` nor IQ types). The `smmla`
unit sat idle while the core did scalar-ish dot products.

There is a loop worth naming: on a cost-efficient Arm instance you want the biggest model in the
least RAM, so you pick the most aggressive 4-bit format — which was exactly the format with no Arm
kernel. **The choice you made to fit the instance was the choice that gave back the performance you
were trying to buy.**

FastPath64 closes it.

## Result

Stock and patched built in the **same CI job on the same physical runner**. Llama-3.2-3B, IQ4_XS:

| | stock | FastPath64 |
|---|---:|---:|
| prefill `pp512` | 24.95 ±0.02 t/s | **52.85 ±0.02 t/s — 2.12x** |
| prefill `pp2048` | 17.72 ±0.01 t/s | **28.31 ±0.01 t/s — 1.60x** |
| Q4_K control | 42.70 ±0.06 t/s | 42.74 ±0.04 t/s — 1.00x |

**IQ4_XS ran prefill at 0.58x of Q4_K. It now runs at 1.24x of Q4_K — while still being the smaller
file.** The format the MoE ecosystem standardised on went from the slowest 4-bit option on Arm to
the fastest. It ends up ahead because IQ4_XS carries no min/`dmin` term: one 6-bit scale covers each
32-element sub-block, so the kernel does strictly less work per byte than Q4_K's.

The Q4_K control lands at **1.00x to two decimals on all four cases**. Q4_K is a format this work
does not touch, so if the patched build were faster for any incidental reason — compiler luck, a
warmer cache, a quieter neighbour — the control would have moved too. It did not.

→ **[results/p1-results.md](results/p1-results.md)** — full tables, MoE numbers, the decode
regression, and the correctness gate ·
[the run](https://github.com/Marc-Dvci/fastpath64/actions/runs/30156929580)

## Status

| # | Work | Status |
|---|---|---|
| P0 | Measure the gap on Neoverse N2 | **done** — mechanism proven, not just observed |
| P1a | Layout, repack, portable reference kernels | **done** — [patch](patches/0001-iq4_xs-repack-reference.patch) |
| P1b | NEON `smmla` GEMM + `sdot` GEMV | **done** — [patch](patches/0002-iq4_xs-arm-neon-smmla.patch) |
| P1c | `sdot` GEMM for pre-I8MM cores | **done** — [patch](patches/0003-iq4_xs-arm-dotprod-gemm.patch) |
| — | Upstream PR | branch [pushed](https://github.com/Marc-Dvci/llama.cpp/tree/iq4-xs-arm-repack), [draft](docs/upstream-pr.md) |
| P2 | Fix the 3% dense-decode regression | in progress |
| P3 | `MUL_MAT_ID` support in KleidiAI | not started |

Three kernels, covering every Arm server CPU in service:

| path | instruction | hardware |
|---|---|---|
| GEMM | `smmla` (I8MM) | Graviton3/4, Cobalt 100, Axion |
| GEMM | `sdot` (DOTPROD) | Graviton2, Ampere Altra |
| GEMV | `sdot` (DOTPROD) | all of the above |

GEMV deliberately stays on DOTPROD: at `nr == 1` there is no second activation row to fill an
`smmla` operand, and regressing batch-1 decode off the existing `vec_dot` path would cost more than
the prefill win is worth.

## What is *not* claimed

- **Decode does not get faster.** It is memory-bandwidth-bound and no matmul kernel changes that.
  Measured 0.97–1.03x. There is a real ~3% regression on dense decode, quantified and explained in
  [results/p1-results.md](results/p1-results.md#decode) rather than buried.
- **Output is not bit-identical.** `smmla` accumulates in a different order than the reference path,
  so results differ by ~1e-6. Some shapes come out exactly equal, most do not. What is gated is
  numerical equivalence against the non-repacked path, not bitwise equality.
- **No numbers for large MoEs.** gemma-4-26B-A4B and Qwen3.6-35B-A3B are the models this work aims
  at, but they do not fit in a free runner's 16 GB. The MoE figure here (1.13x) is from OLMoE-1B-7B,
  whose 1B active parameters make its expert matrices small — a conservative floor, not a ceiling.
- **No AMX measurement.** That Intel has an IQ4_XS path is read from upstream source; the x86 runner
  available here is an AMD EPYC with no AMX.

## How the gap was found and measured

Three builds of one pinned commit, same runner, same GGUF, differing only in whether Arm's fast path
is compiled in:

| build | Q4_K_M | IQ4_XS |
|---|---:|---:|
| stock (repack ON) | **42.78** | **25.08** |
| repack OFF | 27.65 | 24.94 |
| KleidiAI ON | 42.78 | 25.01 |
| **what Arm's fast path was worth** | **1.55x** | **1.01x** |

Turning the fast path off cost Q4_K 35% of its prefill. It cost IQ4_XS **0.6%** — less than the
run-to-run spread. You cannot lose what you never had; that toggle is the difference between
observing a gap and demonstrating its mechanism. Enabling KleidiAI changed nothing for either,
because it accepts only Q4_0 and Q8_0.

→ **[docs/the-gap.md](docs/the-gap.md)** (source evidence, file:line) ·
**[results/phase0.md](results/phase0.md)** (full tables) ·
[the run](https://github.com/Marc-Dvci/fastpath64/actions/runs/30148951999)

## Reproduce it — no Arm hardware, no cloud account, no spend

**Benchmarks**, on GitHub's free Neoverse N2 runners:

> Actions → **A/B - stock vs FastPath64 on Neoverse N2** → Run workflow

**Correctness**, locally on an x86 laptop. Cross-compiles at native speed and executes under QEMU,
which reaches cores no free CI runner offers — `neoverse-n1` is the Ampere Altra / Graviton2
dispatch path:

```bash
docker build -t fastpath64-cross tools/qemu/
docker run --rm -v "$PWD/..:/src" fastpath64-cross bash /src/fastpath64/tools/qemu/run-equiv-test.sh
```

24 combinations: 6 shapes × {iq4_xs, q4_K} × {neoverse-n2, neoverse-n1}. `ggml-cpu` bakes its
`-march` baseline in at compile time, so the harness builds one variant per emulated core —
mispairing them yields SIGILL rather than a result.

QEMU is used for **correctness only**. It does not model microarchitecture and nothing timed under
it is ever reported as a benchmark.

## Correctness gate

[`tools/qemu/test_repack_equiv.cpp`](tools/qemu/test_repack_equiv.cpp) fills a gap in upstream's own
testing: `test-backend-ops` allocates into the *default* CPU buffer, so it never exercises the
repack path at all — plausibly why this area went unnoticed. The test allocates identical weights
into both buffer types and diffs the results.

On real Neoverse N2, run before any timing is reported:

```
type=iq4_xs N=64  K=512  M=1   max_abs=3.815e-06  PASS   (GEMV only)
type=iq4_xs N=64  K=512  M=5   max_abs=3.815e-06  PASS   (both paths + remainder)
type=iq4_xs N=128 K=1024 M=9   max_abs=7.629e-06  PASS
type=iq4_xs N=8   K=256  M=16  max_abs=0.000e+00  PASS   (GEMM only)
```

Worst deviation is below both the portable reference and upstream's own Q4_K repack kernel. The
figures match QEMU to the digit, which is a useful cross-check on the emulation harness itself.

It also caught a latent upstream fault: `init_tensor` leaves `extra == nullptr` for any type no
kernel claims, and `set_tensor` dereferenced it unchecked — a segfault instead of a diagnosable
error. Fixed in [patch 0001](patches/0001-iq4_xs-repack-reference.patch).

## Method

- Upstream pinned in `UPSTREAM_SHA`; both arms of every A/B built from it in one job.
- Timings are gated behind the correctness check — the workflow refuses to report numbers if it
  fails.
- Untouched formats are carried as controls in every run.
- Shared cloud vCPUs are noisy: every figure is reported with its spread.

## Licence

MIT — see [LICENSE](LICENSE). Built for the
[Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/),
Track 2 (Cloud AI).
