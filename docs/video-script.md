# Video script — under 3 minutes

Judges are Arm engineers. Lead with the `grep`, not with a logo. No third-party trademarks, no
licensed music (challenge rule). Screen recording throughout; no talking-head needed.

Total: **2:45**.

---

### 0:00–0:25 — The grep

*Screen: a terminal, two commands, nothing else.*

> "This is llama.cpp's Arm fast path. And this is Intel's."

```
$ grep -c IQ4_XS ggml/src/ggml-cpu/repack.cpp
0
$ grep -n IQ4_XS ggml/src/ggml-cpu/amx/common.h
114:        (type == GGML_TYPE_IQ4_XS);
```

> "IQ4_XS is the format nearly every MoE model ships in. Intel AMX has a kernel for it. Arm has
> none."

### 0:25–0:50 — Why that hurts, specifically

*Screen: `ls` of a model folder full of `*-A3B-IQ4_XS.gguf`, then `lscpu` on the runner with
`i8mm` highlighted.*

> "On a cost-efficient Arm instance you want the biggest model in the least RAM — so you pick the
> most aggressive 4-bit format. That's the one with no kernel. The choice you make to fit the
> instance is the choice that gives back the performance you were trying to buy."
>
> "Meanwhile the CPU advertises i8mm. The matrix unit is right there. IQ4_XS never issues a single
> smmla instruction."

### 0:50–1:15 — Proving it's the kernel, not the format

*Screen: the three-build table animating in.*

| build | Q4_K | IQ4_XS |
|---|---:|---:|
| repack ON | 42.78 | 25.08 |
| repack OFF | 27.65 | 24.94 |

> "Same commit, same runner, same files — only the fast path toggled. Turning it off costs Q4_K
> 35% of its prefill. It costs IQ4_XS 0.6%, which is inside the noise. You can't lose what you
> never had."

### 1:15–1:55 — The kernel

*Screen: the interleaved layout diagram, then the smmla inner loop.*

> "IQ4_XS turns out to be IQ4_NL's codebook carrying K-quant super-block scales — and Arm already
> had smmla kernels for both halves separately. The missing kernel was the intersection of two
> kernels that already shipped. That's why nobody wrote it: it fell between two owners."
>
> "A 16-byte weight load meets a 16-byte activation load, and smmla returns the 2×2 tile with no
> shuffling. Three kernels: smmla for Graviton3 and Cobalt, sdot for Graviton2 and Ampere Altra,
> and a dotprod GEMV so batch-1 decode doesn't regress."

### 1:55–2:25 — The result

*Screen: the A/B table, control row highlighted last.*

> "Prefill: 24.95 to 52.85 tokens per second. 2.12x, same GGUF file."
>
> "IQ4_XS was at 0.58x of Q4_K. It's now at 1.24x — the smaller format is now also the faster one."
>
> "And the row that matters most: Q4_K, which this work doesn't touch, sits at 1.00x. If the
> patched build were faster for any incidental reason, the control would have moved too."

### 2:25–2:45 — Check it yourself

*Screen: clicking "Run workflow" in the Actions tab; then the QEMU harness printing PASS lines.*

> "Every number reruns from a button on a public repo, on GitHub's free Neoverse runners. And
> correctness runs on your laptop under emulation — including the Ampere Altra path no free CI
> runner offers."
>
> "The patches are open against upstream llama.cpp. Decode doesn't get faster — it's
> bandwidth-bound, and I don't claim otherwise."

*End card: repo URL, MIT.*

---

## Recording notes

- Record the terminal at 1920×1080, large font. Real output only — no mocked screens.
- The `lscpu` shot must show the real `i8mm` flag from the CI runner (`results/raw-ab/.../hardware.txt`).
- The Actions click-through should be a genuine run, not an edit.
- If the agent-turn numbers land, swap the 1:55 section to a side-by-side wall clock of one
  tool-calling turn — more visceral than a throughput table.
