## One agent turn — 5145 prompt tokens, 96 generated

| | stock | FastPath64 | change |
|---|---:|---:|---:|
| time to first token | 463.48 s | 355.13 s | **1.31x faster** |
| decode | 6.59 s | 6.55 s | **1.01x faster** |
| **whole turn** | 470.09 s | 362.28 s | **1.30x faster** |

Median of 3 runs. The turn is dominated by prefill (99% of stock wall clock), which is why a prefill kernel moves the number a user actually feels.
