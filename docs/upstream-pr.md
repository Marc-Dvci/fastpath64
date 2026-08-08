# Upstream-ready patch series (prepared, not submitted)

The work is packaged so it can go upstream as-is: a rebased branch against pinned upstream, a
commit series that separates layout from kernels, and the description below. **No pull request has
been opened** — that is deliberate, and the decision to submit rests with the author.

Branch: [`Marc-Dvci/llama.cpp:iq4-xs-arm-repack`](https://github.com/Marc-Dvci/llama.cpp/tree/iq4-xs-arm-repack)

---

**Title:** `ggml-cpu: add IQ4_XS to the Arm repack fast path (smmla + sdot kernels)`

## What

IQ4_XS has no entry in `ggml_repack_get_optimal_repack_type()`, so it never reaches the
row-interleaved kernels on any architecture. This adds the interleaved layout, the repack, portable
reference kernels, and Arm NEON kernels for both I8MM (`smmla`) and DOTPROD-only (`sdot`) cores.

## Why this type

IQ4_XS is what most MoE GGUFs are published in, and MoE on CPU is a common way to serve a large
model on a cost-efficient instance without a GPU. Today those weights take the generic `vec_dot`
path on Arm while `Q4_K` gets interleaved `smmla` kernels — so the smaller, more widely distributed
format is the slower one at prefill.

For contrast, `qtype_has_amx_kernels()` in `ggml/src/ggml-cpu/amx/common.h` already lists
`GGML_TYPE_IQ4_XS`, so Intel AMX has a tiled path for this format and Arm has none.

## Approach

IQ4_XS is IQ4_NL's `kvalues_iq4nl` codebook carrying K-quant style super-block scales, so
`block_iq4_xsx8` mirrors `block_q4_Kx8` while keeping IQ4_XS's own scale encoding (6 bits split
4 low / 2 high) and its 16-lo/16-hi nibble grouping. It is exactly `8 * sizeof(block_iq4_xs)`,
since the repack buffer allocates `ggml_nbytes()`.

Unlike Q4_K there is no min term, so no `bsums`/`dmin` correction is needed — a single 6-bit scale
per 32-element sub-block covers both nibble halves. That makes these kernels noticeably simpler
than their Q4_K counterparts.

- **GEMM (I8MM):** a 16-byte weight load (one column pair) meets a 16-byte activation load (one row
  pair) with no shuffling, since the interleaved layout stores 8 bytes per column contiguously.
  `vmmlaq_s32` lanes come out `[w0*a0, w0*a1, w1*a0, w1*a1]`, so lanes 0–1 share the first column's
  scale and 2–3 the second's.
- **GEMM (DOTPROD):** same layout, `sdot` covering 2 columns × 1 row per instruction. Included
  because Neoverse N1 parts (Graviton2, Ampere Altra) are a large share of deployed Arm capacity.
- **GEMV:** stays on DOTPROD. At `nr == 1` there is no second activation row to fill an `smmla`
  operand, and regressing batch-1 decode off the existing `vec_dot` path would cost more than the
  prefill win is worth.

Dispatch is gated at DOTPROD; below that the portable reference would be slower than not repacking
at all, so IQ4_XS is deliberately left alone there.

## Correctness

Validated against the non-repacked path (identical weights allocated into the default CPU buffer
vs the repack buffer type, results diffed) across GEMV-only, GEMM-only and mixed shapes, on
emulated Neoverse N2 and N1 and on real Neoverse N2 hardware in CI.

Worst observed deviation `max_abs = 3.8e-06`, below both the portable reference and the existing
`q4_K` repack kernel — `smmla` accumulates exactly in int32 with fewer float rounding steps.
Several shapes are bit-identical.

Note that `test-backend-ops` cannot cover this: it allocates into the default CPU buffer, so the
repack path is never exercised. The harness used here is described in the linked repo, and I'm
happy to contribute it as a proper test if that's wanted.

## Performance

Measured on GitHub's free `ubuntu-24.04-arm` runners (Azure Cobalt 100, Neoverse N2, 4 vCPU),
building stock and patched **in the same job on the same machine**, so the comparison carries no
machine-to-machine variance. `llama-bench -r 5 -t 4`.

| Llama-3.2-3B-Instruct-IQ4_XS | stock | patched | |
|---|---:|---:|---:|
| `pp512` | 24.95 ±0.02 | **52.85 ±0.02** | **2.12x** |
| `pp2048` | 17.72 ±0.01 | **28.31 ±0.01** | **1.60x** |
| `tg128` | 16.78 ±0.02 | 16.22 ±0.10 | 0.97x |

`Q4_K_M`, which this change does not touch, is carried as a control in every run and lands at
1.00x to two decimal places on all four prefill cases — so the IQ4_XS delta is the kernel and not
the run conditions.

Other models, same harness:

| | `pp512` | `pp2048` | decode |
|---|---:|---:|---:|
| gemma-4-12b-it-IQ4_XS | **2.00x** | 1.61x | 0.92x (`tg64`) |
| OLMoE-1B-7B-Instruct-IQ4_XS (MoE) | **1.13x** | 1.09x | 1.03x |

The MoE gain is smaller because a MoE prefill reads every expert's weights while computing only the
active fraction, so its arithmetic intensity is lower by roughly the sparsity ratio and it sits
closer to the bandwidth-bound regime.

**Decode carries a small regression** (0.97x dense at 3B, 0.92x at 12B). The repacked GEMV performs
per-sub-block scale decoding that the non-repacked `vec_dot` path does not, and at `nr == 1` there
are only 8 output columns to amortise it over. A `vld4_u8`-based fix was written and measured; it
cost ~10% of prefill through a store-to-load forwarding stall and was reverted rather than shipped.
The correct fix keeps the scales in registers and is a larger change to the inner loop; I would
rather land the GEMM win first and treat GEMV separately, but I am happy to hold this until GEMV is
neutral if that is preferred.

On a whole agent turn — 5145-token tool-calling prompt, 96 tokens generated — the end-to-end wall
clock improves 1.30x, and stock and patched emit byte-identical output under greedy decode.

Three independent runs on separate runner instances returned these ratios to two decimal places.

## Drive-by fix

`ggml_backend_cpu_repack_buffer_set_tensor()` dereferences `tensor->extra` without a null check,
but `init_tensor` leaves it null for any type no repack kernel claims. Allocating an unclaimed type
into the repack buffer therefore segfaults instead of reporting the problem. Added an assert with a
diagnostic message. Happy to split this into its own PR.

## Not included

- No SVE-specific path. Neoverse N2 has 128-bit vectors, where the NEON `smmla` path already
  applies; the existing 256-bit SVE `q4_K` kernel is gated on `svcntb() * 8 == 256`.
- No `MUL_MAT_ID` support in KleidiAI (that path accepts only Q4_0/Q8_0 and only `MUL_MAT`), so MoE
  models still get nothing from KleidiAI. Out of scope here.
