# A vectorised scale decode that made things slower

Batch-1 decode carried a ~3% regression against the non-repacked path. The suspected cause was the
per-sub-block scale decode: `scales_l` is row-major, so the eight bytes a sub-block needs sit at
stride 4, and decoding them scalar costs roughly 40 ALU operations and 16 strided loads per
sub-block — 64 decodes per super-block. GEMM amortises that across 32 outputs, GEMV across only 8,
which fit the observed asymmetry.

`vld4_u8` de-interleaves stride-4 bytes natively, resolving all four sub-block pairs in a single
instruction. The rewrite computed all eight scales in about six vector operations:

```c
static inline void iq4_xs_sub_scales(const uint8x8x4_t & sl, const uint16x8_t & sh,
                                     int sb, int16_t * out) {
    const uint8x8_t  byte = sl.val[sb >> 1];
    const uint8x8_t  lo   = (sb & 1) ? vshr_n_u8(byte, 4) : vand_u8(byte, vdup_n_u8(0x0F));
    const uint16x8_t hi   = vandq_u16(vshlq_u16(sh, vdupq_n_s16(-(int16_t)(2 * sb))), vdupq_n_u16(3));
    const uint16x8_t ls   = vorrq_u16(vmovl_u8(lo), vshlq_n_u16(hi, 4));
    vst1q_s16(out, vsubq_s16(vreinterpretq_s16_u16(ls), vdupq_n_s16(32)));   // <- the problem
}
```

Measured on the same runner, against the same stock build:

| case | patches 0001–0002 | with the rewrite | |
|---|---:|---:|---|
| Llama-3.2-3B `pp512` | **2.12x** | 1.91x | −10% |
| Llama-3.2-3B `pp2048` | **1.60x** | 1.51x | −6% |
| OLMoE-1B-7B `pp512` | **1.13x** | 1.03x | −9% |
| Llama-3.2-3B `tg128` | 0.97x | 0.93x | worse |

It lost throughput everywhere, including on the decode case it was written to fix.

## Why

The final `vst1q_s16` writes 16 bytes, and the consuming kernel immediately reads them back as
eight individual 2-byte scalars. A wide store followed by narrow, overlapping loads is the classic
store-to-load-forwarding failure: the store buffer cannot satisfy a 2-byte load out of a 16-byte
pending store, so each read waits for the store to retire to L1. That penalty lands once per column
per sub-block, which is exactly the innermost loop.

The scalar version it replaced stored and loaded at matching widths, which forwards cleanly. The
arithmetic got cheaper and the memory behaviour got much worse, and the second effect dominated.

The lesson is narrow and worth stating precisely: vectorising a computation is only a win if the
result is also *consumed* in vector form. Producing eight values in one register and then spilling
them to be read one at a time trades ALU work for a pipeline stall.

## Status

Reverted. The shipped patch series is 0001–0003 and the reported figures are the 2.12x set.

The decode regression it targeted is real and remains, documented in
[p1-results.md](p1-results.md#decode). The correct fix keeps the scales in registers and builds the
`[s0, s0, s1, s1]` scale vector each column pair needs with lane operations, never writing them to
memory at all. That is a larger change to the inner loop than the one attempted here, and it is not
included rather than being included unmeasured.
