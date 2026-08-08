## stock vs FastPath64 — same runner, same GGUF

### prefill

| model | quant | case | stock t/s | fastpath t/s | change |
|---|---|---|---:|---:|---:|
| `llama-3.2-3b-iq4_xs` | iq4_xs | pp2048 | 17.72 ±0.03 | 28.36 ±0.02 | **1.60x** **<-** |
| `llama-3.2-3b-iq4_xs` | iq4_xs | pp512 | 24.95 ±0.03 | 52.90 ±0.04 | **2.12x** **<-** |
| `llama-3.2-3b-q4_k_m` | q4_k | pp2048 | 25.14 ±0.01 | 25.15 ±0.02 | **1.00x** |
| `llama-3.2-3b-q4_k_m` | q4_k | pp512 | 42.72 ±0.05 | 42.80 ±0.04 | **1.00x** |
| `olmoe-1b-7b-iq4_xs` | iq4_xs | pp2048 | 46.88 ±0.05 | 51.13 ±0.11 | **1.09x** **<-** |
| `olmoe-1b-7b-iq4_xs` | iq4_xs | pp512 | 66.26 ±0.12 | 74.88 ±0.38 | **1.13x** **<-** |
| `olmoe-1b-7b-q4_k_m` | q4_k | pp2048 | 49.78 ±0.04 | 49.77 ±0.02 | **1.00x** |
| `olmoe-1b-7b-q4_k_m` | q4_k | pp512 | 72.30 ±0.02 | 72.23 ±0.14 | **1.00x** |

### decode

| model | quant | case | stock t/s | fastpath t/s | change |
|---|---|---|---:|---:|---:|
| `llama-3.2-3b-iq4_xs` | iq4_xs | tg128 | 17.08 ±0.04 | 16.51 ±0.03 | **0.97x** |
| `llama-3.2-3b-q4_k_m` | q4_k | tg128 | 16.64 ±0.05 | 16.81 ±0.06 | **1.01x** |
| `olmoe-1b-7b-iq4_xs` | iq4_xs | tg128 | 43.22 ±0.11 | 46.04 ±0.17 | **1.07x** |
| `olmoe-1b-7b-q4_k_m` | q4_k | tg128 | 42.96 ±0.50 | 42.94 ±0.43 | **1.00x** |

---

- best IQ4_XS prefill: **2.12x** on `llama-3.2-3b-iq4_xs`
- worst decode: 0.97x on `llama-3.2-3b-iq4_xs`  :warning: **decode regression**

Rows not marked `<-` are controls: formats FastPath64 does not touch should land at ~1.00x. If they move, the run is measuring something other than the kernel.
