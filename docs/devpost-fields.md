# Devpost form fields

## Title (60 char max)

```
FastPath64: the missing Arm smmla kernel for IQ4_XS
```
50 characters.

Alternatives, if a different emphasis is wanted:

| option | chars |
|---|---:|
| `FastPath64: 2.12x faster IQ4_XS prefill on Neoverse` | 51 |
| `FastPath64: putting IQ4_XS back on Arm's fast path` | 50 |

## Pitch (200 char max)

```
IQ4_XS, the format most quantized MoE models ship in, had an accelerated matmul kernel on Intel AMX and none on Arm. FastPath64 writes it: 2.12x faster prefill on Neoverse N2, same GGUF, no quality loss.
```
201 characters — trim to 199 by dropping the final period and the comma after "writes it":

```
IQ4_XS, the format most quantized MoE models ship in, had an accelerated matmul kernel on Intel AMX and none on Arm. FastPath64 writes it: 2.12x faster prefill on Neoverse N2, same GGUF, no quality loss
```
200 characters exactly.

## Built with (25 tag max)

Tags are lowercase, comma-separated on Devpost. These 22 are all genuinely used:

```
arm, aarch64, neoverse, arm-neon, smmla, i8mm, sve2, llama.cpp, ggml, kleidiai,
c++, c, neon-intrinsics, quantization, iq4_xs, gguf, cmake, github-actions,
qemu, docker, python, remotion
```

Rationale for the less obvious entries:

- `smmla` / `i8mm` / `sve2` — the instruction and feature names the work targets; these are what an Arm engineer searches for.
- `kleidiai` — audited as part of establishing the gap, even though the fix lands in the repack path.
- `qemu` / `docker` — the cross-build harness that reaches the DOTPROD-only dispatch path.
- `github-actions` — the benchmark and correctness harness runs there, on free Arm runners.
- `remotion` — the demo video renders from the measurement files.

## What is one thing Arm could improve to better support developers like you?

```
Close the gap between "the silicon supports it" and "the framework reaches it", and
make that gap visible.

Neoverse N2 advertises i8mm in lscpu, but a model in IQ4_XS issued no smmla at all,
because the format had no entry in llama.cpp's repack table and KleidiAI accepts only
Q4_0 and Q8_0 on MUL_MAT — not MUL_MAT_ID, which is where every mixture-of-experts
expert matmul goes. Nothing failed and nothing warned; the work simply took the scalar
path. I found it by reading source, not by profiling, because there was no signal to
profile against.

Two concrete things would have helped.

First, coverage as a published matrix rather than something to be reverse-engineered
from a dispatch function: for each quantization type and each op, which Arm path
accelerates it today. That table is what tells a developer whether a format choice
costs them the matrix unit, and it is currently only recoverable by reading
ggml_repack_get_optimal_repack_type and kleidiai.cpp side by side.

Second, a runtime notice when a hot tensor falls back. KleidiAI already logs when it
declines a tensor type; the generic path is silent. One line at load time — "these
weights will not use i8mm" — turns an invisible 2x into something a developer can act
on in a minute.

The deeper point is that KleidiAI's op coverage has not tracked how models are actually
deployed. MoE routing through MUL_MAT_ID is now mainstream, and it is the one op the
library does not implement. Extending Arm's own kernels to MUL_MAT_ID, and to the IQ
formats the ecosystem quantizes to, would reach a large share of real CPU inference
without any developer having to write intrinsics — which is the point of shipping a
microkernel library in the first place.
```

Shorter variant if the field is tight:

```
Close the gap between "the silicon supports it" and "the framework reaches it", and make
it visible. Neoverse N2 advertises i8mm, yet an IQ4_XS model issued no smmla: the format
had no entry in llama.cpp's repack table, and KleidiAI accepts only Q4_0/Q8_0 on MUL_MAT
— not MUL_MAT_ID, where every MoE expert matmul goes. Nothing failed and nothing warned.

Two things would help: publish a coverage matrix of quantization type x op x Arm path, so
a format choice's cost is visible without reading dispatch code; and emit a one-line notice
at load time when hot tensors fall back off the accelerated path. Extending KleidiAI to
MUL_MAT_ID and the IQ formats would reach most real CPU inference without anyone writing
intrinsics.
```
