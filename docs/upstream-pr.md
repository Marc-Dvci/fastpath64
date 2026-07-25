# Draft PR to ggml-org/llama.cpp

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

<!-- filled from the A/B run before submitting -->

Measured on GitHub's free `ubuntu-24.04-arm` runners (Azure Cobalt 100, Neoverse N2, 4 vCPU),
building stock and patched in the same job on the same machine.

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
