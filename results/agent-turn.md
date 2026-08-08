# One whole agent turn

**Run:** [actions/runs/30171245415](https://github.com/Marc-Dvci/fastpath64/actions/runs/30171245415) ·
upstream `555881e` + [patches](../patches) · raw artifacts in [`raw-agent/`](raw-agent)

`llama-bench` measures a kernel at synthetic prompt sizes. This measures the thing the instance is
rented for: an agent turn — a large system prompt, tool schemas, conversation history and retrieved
context in; a short structured tool call out.

The prompt is a real tool-calling turn built by [`bench/make_agent_trace.py`](../bench/make_agent_trace.py):
**5145 tokens** in, 96 generated, greedy decode. Both binaries come from the same job on the same
physical Neoverse N2 runner, so nothing but the kernel differs.

| median of 3 | stock | FastPath64 | |
|---|---:|---:|---:|
| time to first token | 463.48 s | 355.13 s | **1.31x** |
| decode | 6.59 s | 6.55 s | 1.01x |
| **whole turn** | **470.09 s** | **362.28 s** | **1.30x** |

**99% of the stock turn is prefill.** That is the whole argument for working on a prefill kernel:
on agent traffic it is not one component among several, it is essentially the entire wall clock.

The three runs per arm bracket the result rather than leaving it to a single median — pairing the
worst patched run against the best stock run still gives 1.29x, the reverse gives 1.31x. Absolute
times are a property of a free 4 vCPU runner and a 3B model; the ratio is the result.

## The gain declines with prompt length, exactly as predicted

| prompt tokens | speedup |
|---:|---:|
| 512 | **2.12x** |
| 2048 | **1.60x** |
| 5145 *(this turn)* | **1.31x** |

This work touches the FFN matmuls and nothing else. Attention grows with sequence length, so its
share of prefill grows too, and the share left for a matmul kernel to improve shrinks. The trend is
monotone across three independent measurements and is the reason no headline number here is quoted
without its prompt length.

## The output does not change

Same prompt, same seed, greedy decode, 96 tokens — the two builds emit **byte-identical** text:

```
## Generated output, greedy decode, same seed

**Byte-identical.** Stock and FastPath64 produced exactly the same 96 tokens.

{"name": "create_ticket", "parameters": {"project": "warehouse_reconciliation",
 "title": "Partition drift detected", "body": "The nightly reconciliation job failed
 again. Partition tenant_a4f2 drifted by 224 units.", "labels": {"status": "open"}}}
```

This is a stronger statement than the equivalence gate, and a weaker one than it looks. Stronger,
because it is what a user cares about: the model produces the same tool call, not merely
numerically close logits. Weaker, because bit-identical output is not guaranteed in general —
`smmla` accumulates in a different order, the kernels differ by ~1e-6, and a token whose top two
candidates sit inside that margin could in principle flip. What is *gated* in CI is numerical
equivalence ([p1-results.md](p1-results.md#correctness-on-the-same-silicon)); this run shows that on
a realistic turn the difference stayed well below the decision boundary at every one of 96 steps.

Both checks run in the same workflow, on every A/B:
[`bench/agentbench.py`](../bench/agentbench.py) and [`bench/diff_generation.sh`](../bench/diff_generation.sh).

## Why this is the number that matters for Track 2

An Arm instance serving agent traffic is, in throughput terms, a prefill machine. Its capacity is
set by how fast it can absorb context, and the format chosen to make the model fit in RAM was the
one with no accelerated kernel to absorb it with. On this measurement that cost 108 seconds of every
470-second turn — 23% of the wall clock, given back for free, on an unmodified GGUF file.
