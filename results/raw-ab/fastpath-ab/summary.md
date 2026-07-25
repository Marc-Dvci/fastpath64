## stock vs FastPath64 — same runner, same GGUF

### prefill

| model | quant | case | stock t/s | fastpath t/s | change |
|---|---|---|---:|---:|---:|
| `llama-3.2-3b-iq4_xs` | iq4_xs | pp2048 | 17.72 ±0.01 | 28.31 ±0.01 | **1.60x** **<-** |
| `llama-3.2-3b-iq4_xs` | iq4_xs | pp512 | 24.95 ±0.02 | 52.85 ±0.02 | **2.12x** **<-** |
| `llama-3.2-3b-q4_k_m` | q4_k | pp2048 | 25.11 ±0.01 | 25.09 ±0.05 | **1.00x** |
| `llama-3.2-3b-q4_k_m` | q4_k | pp512 | 42.70 ±0.06 | 42.74 ±0.04 | **1.00x** |
| `olmoe-1b-7b-iq4_xs` | iq4_xs | pp2048 | 46.80 ±0.02 | 51.03 ±0.04 | **1.09x** **<-** |
| `olmoe-1b-7b-iq4_xs` | iq4_xs | pp512 | 66.11 ±0.04 | 74.71 ±0.09 | **1.13x** **<-** |
| `olmoe-1b-7b-q4_k_m` | q4_k | pp2048 | 49.70 ±0.02 | 49.69 ±0.02 | **1.00x** |
| `olmoe-1b-7b-q4_k_m` | q4_k | pp512 | 72.18 ±0.04 | 72.08 ±0.14 | **1.00x** |

### decode

| model | quant | case | stock t/s | fastpath t/s | change |
|---|---|---|---:|---:|---:|
| `llama-3.2-3b-iq4_xs` | iq4_xs | tg128 | 16.78 ±0.02 | 16.22 ±0.10 | **0.97x** |
| `llama-3.2-3b-q4_k_m` | q4_k | tg128 | 16.45 ±0.06 | 16.25 ±0.02 | **0.99x** |
| `olmoe-1b-7b-iq4_xs` | iq4_xs | tg128 | 43.21 ±0.23 | 44.51 ±0.29 | **1.03x** |
| `olmoe-1b-7b-q4_k_m` | q4_k | tg128 | 42.69 ±0.18 | 42.22 ±0.18 | **0.99x** |

---

- best IQ4_XS prefill: **2.12x** on `llama-3.2-3b-iq4_xs`
- worst decode: 0.97x on `llama-3.2-3b-iq4_xs`  :warning: **decode regression**

Rows not marked `<-` are controls: formats FastPath64 does not touch should land at ~1.00x. If they move, the run is measuring something other than the kernel.
