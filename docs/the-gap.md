# The gap, in upstream source

Everything below is read from `ggml-org/llama.cpp` at pinned commit
[`555881e`](https://github.com/ggml-org/llama.cpp/commit/555881ebc8b0fc0402b30e09258a32a7bfd13c52)
(2026-07-24). No inference, no benchmarks yet — just what the code says. Every claim is one
`grep` away from being checked by a reviewer.

## 1. Arm has two accelerated matmul paths in the CPU backend

| Path | File | What it is |
|---|---|---|
| **KleidiAI** | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` | Arm's own microkernel library. The only path with hand-tuned DOTPROD / I8MM / SVE / SME2 kernel selection. |
| **repack** | `ggml/src/ggml-cpu/repack.cpp` + `arch/arm/repack.cpp` | Generic row-interleaved kernels. The 8x8 variants issue `smmla` (I8MM); the 8x4 variants fall back to `sdot` (DOTPROD). |

A weight tensor reaches either path only if its **ggml type** is in that path's table.

## 2. What each path actually accepts

### KleidiAI — ops

```
kleidiai.cpp:536   bool work_size(...)  { if (op->op != GGML_OP_MUL_MAT) { return false; } ... }
kleidiai.cpp:668   if (dst->op == GGML_OP_MUL_MAT) { ... } else if (dst->op == GGML_OP_GET_ROWS) { ... }
```

`GGML_OP_MUL_MAT_ID` — the op that performs **every MoE expert matmul** — appears nowhere in the file.

### KleidiAI — types

```
kleidiai.cpp:669   if (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0)
kleidiai.cpp:974   GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0);
kleidiai.cpp:1337  GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0);
```

Q4_0 and Q8_0 only (plus F32, and F16 for MUL_MAT).

### repack — ops

```
repack.cpp:4775   if (op->op == GGML_OP_MUL_MAT && ...)
repack.cpp:4791   } else if (op->op == GGML_OP_MUL_MAT_ID && ... (ggml_n_dims(op->src[0]) == 3) ...)
```

repack **does** handle MoE. Good — but only for types in its table.

### repack — types

`ggml_repack_get_optimal_repack_type()` at `repack.cpp:4528` instantiates traits for exactly:

`Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `IQ4_NL`, `MXFP4`, `Q8_0`

```
$ grep -c IQ4_XS ggml/src/ggml-cpu/repack.cpp
0
```

**`IQ4_XS` is not there.** Neither is `IQ2_XXS` or any `IQ3_*`.

## 3. The consequence

Take a 2026-typical deployment: a MoE model — Qwen3.6-35B-A3B, gemma-4-26B-A4B,
Nemotron-3-Nano-30B-A3B — quantized to **IQ4_XS**, running on a Graviton / Cobalt / Axion
instance because that is the cost-efficient way to serve it without a GPU.

Its expert matmuls get:

- **no repack kernel** — `IQ4_XS` is not in the type table, so `ggml_repack_get_optimal_repack_type()`
  returns `nullptr` and the tensor never enters the repack buffer type;
- **no KleidiAI kernel** — `MUL_MAT_ID` is not a supported op, and `IQ4_XS` is not a supported type.

Both doors are shut. The work falls back to the generic per-row `vec_dot` path: `sdot` at best,
**`smmla` never issued**, on a core that has an I8MM unit sitting idle.

## 4. The part that should be uncomfortable

```
ggml/src/ggml-cpu/amx/common.h:106
inline bool qtype_has_amx_kernels(const enum ggml_type type) {
    return (type == GGML_TYPE_Q4_0) || (type == GGML_TYPE_Q4_1) || (type == GGML_TYPE_Q8_0) ||
           (type == GGML_TYPE_Q4_K) || (type == GGML_TYPE_Q5_K) || (type == GGML_TYPE_Q6_K) ||
           (type == GGML_TYPE_IQ4_XS);
}
```

**Intel AMX has an IQ4_XS path. Arm has none.**

The same GGUF file that the ecosystem standardised on gets a tiled integer-matmul kernel on x86
and a scalar-ish fallback on Arm.

## 5. Why IQ4_XS specifically

There is a loop worth naming. On a cost-efficient Arm instance you want the largest model in the
least RAM, so you reach for the most aggressive 4-bit format — IQ4_XS, or an "UD" IQ2/IQ3 variant.
That choice is *exactly* the one with no Arm kernel. **The format you pick to fit the instance is
the format that gives back the performance you were trying to buy.**

## 6. What this repo does about it

| # | Work | Opens |
|---|---|---|
| P1 | `iq4_xs_8x8_q8_K` row-interleaved `smmla` GEMM + `8x4` `sdot` GEMV, modelled on the existing `q4_K` case at `repack.cpp:4600` | the repack door |
| P2 | expert-row grouping so MoE `MUL_MAT_ID` reaches M≥8 tiles instead of per-expert GEMV | `smmla` utilisation under real serving |
| P3 | `GGML_OP_MUL_MAT_ID` support in `kleidiai.cpp` | the KleidiAI door |

Measured on free Arm silicon (GitHub `ubuntu-24.04-arm` = Azure Cobalt 100, Neoverse N2 with
SVE2 + I8MM), reproducible by anyone with one click.
