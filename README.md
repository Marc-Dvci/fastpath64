# FastPath64

**IQ4_XS — the 4-bit format nearly every MoE GGUF ships in — has an accelerated matmul kernel on Intel AMX and none on Arm.**

That is not a claim about benchmarks. It is what upstream `llama.cpp` says about itself:

```
$ grep -c IQ4_XS ggml/src/ggml-cpu/repack.cpp          # Arm's row-interleaved smmla kernels
0
$ grep -n IQ4_XS ggml/src/ggml-cpu/amx/common.h        # Intel AMX tiled kernels
114:        (type == GGML_TYPE_IQ4_XS);
```

So a Qwen3.6-35B-A3B or gemma-4-26B-A4B in IQ4_XS, served on a Graviton / Cobalt / Axion
instance, runs its expert matmuls with **zero I8MM acceleration** — locked out of the repack
path (type not in the table) *and* out of KleidiAI (which supports neither `MUL_MAT_ID` nor
IQ types). The `smmla` unit sits idle while the core does scalar-ish dot products.

There is a loop worth naming: on a cost-efficient Arm instance you want the biggest model in the
least RAM, so you pick the most aggressive 4-bit format — which is exactly the format with no Arm
kernel. **The choice you make to fit the instance is the choice that gives back the performance
you were trying to buy.**

FastPath64 closes it. Same GGUF, bit-identical output, on the silicon Arm actually ships.

→ **[docs/the-gap.md](docs/the-gap.md)** — the full source-level evidence, with file:line citations.

## Measured on Neoverse N2

Three builds of the same pinned upstream commit, same runner, same GGUF files — differing only in
whether Arm's fast path is compiled in. Llama-3.2-3B, prefill `pp512`, tokens/s:

| build | Q4_K_M | IQ4_XS |
|---|---:|---:|
| stock (repack ON) | **42.78** | **25.08** |
| repack OFF | 27.65 | 24.94 |
| KleidiAI ON | 42.78 | 25.01 |
| **what Arm's fast path is worth** | **1.55x** | **1.01x** |

Turning the fast path off costs Q4_K 35% of its prefill throughput. It costs IQ4_XS **0.6%** —
less than the run-to-run spread. You cannot lose what you never had. Enabling KleidiAI changes
nothing for either, because it only accepts Q4_0 and Q8_0.

Meanwhile the runner's own `lscpu` reports `... sve2 ... svei8mm i8mm bf16`. **The I8MM unit is
right there, and IQ4_XS never issues a single `smmla`.**

→ **[results/phase0.md](results/phase0.md)** — full tables incl. MoE, decode, x86, and the
projected ceiling · [the run that produced them](https://github.com/Marc-Dvci/fastpath64/actions/runs/30148951999)

| # | Work | Status |
|---|---|---|
| P0 | Measure the gap on Neoverse N2 | **done — gap confirmed, mechanism proven** |
| P1a | IQ4_XS layout + repack + reference kernels, validated | **done — [patch](patches/0001-iq4_xs-repack-reference.patch)** |
| P1b | Vectorised `smmla` GEMM / `sdot` GEMV for `iq4_xs_8x8_q8_K` | in progress |
| P2 | Expert-row grouping so MoE `MUL_MAT_ID` reaches M≥8 tiles | not started |
| P3 | `GGML_OP_MUL_MAT_ID` support in KleidiAI | not started |

### P1a — IQ4_XS is on the fast path

`block_iq4_xsx8` mirrors `block_q4_Kx8` while keeping IQ4_XS's own scale encoding and its
16-lo/16-hi nibble grouping, at exactly `8 * sizeof(block_iq4_xs)` — the repack buffer allocates
`ggml_nbytes()`, so there is no room to pre-decode scales.

**This step carries no speedup yet, and is not claimed to.** It registers portable reference
kernels so the layout and dispatch can be proven correct before any intrinsics are written. The
`smmla` kernel is P1b.

Validated with [`tools/qemu/test_repack_equiv.cpp`](tools/qemu/test_repack_equiv.cpp), which fills
a gap in upstream's own testing: `test-backend-ops` allocates into the *default* CPU buffer, so it
never exercises the repack path at all. This test allocates identical weights into both buffer
types and diffs the results.

| emulated core | `-march` | IQ4_XS | Q4_K (control) |
|---|---|---|---|
| Neoverse N2 | `armv8.6-a+i8mm+dotprod` | claimed, `max_abs=1.5e-05` **PASS** | claimed, `max_abs=1.1e-05` PASS |
| Neoverse N1 | `armv8.2-a+dotprod` | correctly **not** claimed (SKIP) | claimed `8x4`, PASS |

IQ4_XS's error profile matches upstream's own Q4_K repack kernel, which is the right bar for a
pure layout change. N1 is left on the existing path by design rather than regressed onto a scalar
reference — a `sdot` variant would be needed to claim it.

Run it yourself with no cloud account and no Arm hardware:

```bash
docker build -t fastpath64-cross tools/qemu/
docker run --rm -v "$PWD/..:/src" fastpath64-cross bash /src/fastpath64/tools/qemu/run-equiv-test.sh
```

QEMU is used for **correctness only** — it does not model microarchitecture, and nothing timed
under it is ever reported here as a benchmark.

While building this, the test caught a latent upstream fault: `init_tensor` leaves `extra ==
nullptr` for any type no kernel claims, and `set_tensor` dereferenced it without a check, turning
a configuration error into a segfault. The patch adds an assert with a diagnostic message.

## Reproduce it yourself, for free

Every measurement runs on **GitHub's free arm64 runners** — Azure Cobalt 100, Neoverse N2,
Armv9 with SVE2 and I8MM. No cloud account, no spend, no trust required:

> Actions → **Phase 0 - measure the Arm gap** → Run workflow

Locally, on any Arm64 Linux box:

```bash
git clone <this repo> && cd fastpath64
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git upstream
git -C upstream checkout "$(cat UPSTREAM_SHA)"
cmake -S upstream -B upstream/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build upstream/build --target llama-bench -j"$(nproc)"

# paired quants of one architecture: any delta is the format's fast path, not the model
while IFS='|' read -r name url; do case "$name" in ''|'#'*) continue;; esac
  curl -fL -o "models/$name" "$url"; done < bench/models.txt

upstream/build/bin/llama-bench -m models/llama-3.2-3b-q4_k_m.gguf -p 512 -n 128 -r 5 -o csv
upstream/build/bin/llama-bench -m models/llama-3.2-3b-iq4_xs.gguf -p 512 -n 128 -r 5 -o csv
```

Prefill (`pp`) is compute-bound — that is where a missing `smmla` kernel shows. Decode (`tg`) is
memory-bandwidth-bound and should stay roughly flat; **no kernel work fixes batch-1 decode, and
this project will not claim otherwise.**

## Method notes

- Upstream is pinned (`UPSTREAM_SHA`) so results stay comparable across runs.
- Benchmarks run on shared cloud vCPUs. Every number is reported with its spread, never as a
  single clean figure.
- Correctness gates land alongside the kernels: `test-backend-ops` for `MUL_MAT`/`MUL_MAT_ID`,
  unchanged perplexity, and a bit-exact token-stream diff against the stock build.

## Licence

MIT — see [LICENSE](LICENSE). Built for the
[Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/),
Track 2 (Cloud AI).
