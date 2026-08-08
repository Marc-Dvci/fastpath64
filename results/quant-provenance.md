# A file named IQ4_XS need not contain IQ4_XS

A benchmark of `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` returned exactly 1.00× on every case — prefill
and decode, both prompt lengths, to two decimal places. A null result that clean is rarely a
performance finding; it usually means the code under test never ran.

`llama-bench` reported the model type as `gemma4 26B.A4B IQ4_XS - 4.25 bpw`, which is the file's
`general.file_type` metadata: the quantisation the file was *produced as*, not an inventory of what
its tensors are. Parsing the tensor table directly gives a different picture:

```
$ python3 bench/gguf_types.py gemma-4-26B-A4B-it-UD-IQ4_XS.gguf

All tensors
type        tensors      GiB    share
IQ3_S            30     6.09    48.8%
IQ4_NL           30     3.99    31.9%
Q8_0            206     2.36    18.9%
F32             392     0.04     0.3%

FFN / expert tensors only (these dominate compute)
IQ3_S            30     6.09    57.2%
IQ4_NL           30     3.99    37.4%
```

**The file contains no IQ4_XS tensors at all.** "Dynamic" quantisations assign types per tensor to
hit a target bitrate, and the name records the target rather than the contents. Benchmarking that
file against a kernel that targets IQ4_XS measures code the patch never touches, and 1.00× is the
correct and expected answer.

## Provenance of the measured models

The same tool, run against the files behind the reported numbers:

| model | IQ4_XS share, all tensors | IQ4_XS share, FFN tensors |
|---|---:|---:|
| Llama-3.2-3B-Instruct-IQ4_XS | 82.2% | **100%** |
| OLMoE-1B-7B-0924-Instruct-IQ4_XS | 97.5% | 97.5% |
| gemma-4-12b-it-IQ4_XS | 91.0% | **100%** |

In each case the compute-dominant FFN tensors are the type under test, so the throughput deltas are
attributable to the kernel. The remaining share is `Q6_K`/`Q4_K` on the token-embedding and output
tensors, which standard quantisation keeps at higher precision and which this work does not touch.

## Consequence for the harness

Quant provenance is now a gate rather than an assumption. Both benchmark workflows parse the GGUF
tensor table before running and refuse to proceed if the FFN tensors of a file under test are not
predominantly the type under test — `.github/workflows/bench-fastpath.yml` (which gates the
`*iq4_xs*` files and prints the histogram for the `q4_k` controls, which are supposed to be `q4_k`)
and `.github/workflows/bench-bigmoe.yml`. A measurement on the wrong file is worse than no
measurement, because it looks like a result.
