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

## Status

Phase 0: quantifying the gap on real Neoverse N2 silicon. Nothing here is measured yet — this
README will carry no performance number that isn't linked to the CI run that produced it.

| # | Work | Status |
|---|---|---|
| P0 | Measure the gap on Neoverse N2 | harness written, awaiting first run |
| P1 | `iq4_xs_8x8_q8_K` interleaved `smmla` GEMM + `8x4` `sdot` GEMV | not started |
| P2 | Expert-row grouping so MoE `MUL_MAT_ID` reaches M≥8 tiles | not started |
| P3 | `GGML_OP_MUL_MAT_ID` support in KleidiAI | not started |

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
