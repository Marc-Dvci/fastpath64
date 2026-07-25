#!/usr/bin/env python3
"""Report the tensor-type histogram of a GGUF file, weighted by bytes.

A GGUF whose name says IQ4_XS is not necessarily IQ4_XS throughout: "dynamic" quantisations
mix types per tensor, and the type llama.cpp reports is only the most common one. What matters
for a kernel targeting one type is how many *bytes of hot tensors* actually carry it.

    python3 bench/gguf_types.py model.gguf
"""

import struct
import sys
from collections import defaultdict

# ggml_type -> (name, block_size, type_size)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17),
}


class R:
    def __init__(self, f):
        self.f = f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def i(self, fmt, n):
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u32(), self.u64()
            return [self.value(et) for _ in range(n)]
        sizes = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
                 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
                 12: ("<d", 8)}
        fmt, n = sizes[t]
        return self.i(fmt, n)


def main(path, share_of=None):
    with open(path, "rb") as f:
        r = R(f)
        if f.read(4) != b"GGUF":
            print("not a GGUF file")
            return 1
        r.u32()                       # version
        n_tensors = r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):         # skip metadata
            r.string()
            r.value(r.u32())

        by_type = defaultdict(lambda: [0, 0])       # name -> [count, bytes]
        expert_by_type = defaultdict(lambda: [0, 0])
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            t = r.u32()
            r.u64()                                  # offset
            tname, blk, tsz = TYPES.get(t, (f"?{t}", 1, 1))
            nelem = 1
            for d in dims:
                nelem *= d
            nbytes = nelem // blk * tsz
            by_type[tname][0] += 1
            by_type[tname][1] += nbytes
            # MoE expert weights dominate compute in a sparse model
            if "exp" in name or "ffn" in name:
                expert_by_type[tname][0] += 1
                expert_by_type[tname][1] += nbytes

    # machine-readable mode: percentage of FFN/expert bytes carrying one type, for CI gating
    if share_of:
        total = sum(v[1] for v in expert_by_type.values()) or 1
        print(f"{100 * expert_by_type.get(share_of, [0, 0])[1] / total:.1f}")
        return 0

    def report(title, d):
        total = sum(v[1] for v in d.values()) or 1
        print(f"\n{title}")
        print(f"{'type':<10}{'tensors':>9}{'GiB':>9}{'share':>9}")
        for k, (c, b) in sorted(d.items(), key=lambda kv: -kv[1][1]):
            print(f"{k:<10}{c:>9}{b / 2**30:>9.2f}{100 * b / total:>8.1f}%")

    report(f"All tensors ({path})", by_type)
    report("FFN / expert tensors only (these dominate compute)", expert_by_type)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    want = next((f.split("=", 1)[1] for f in flags if f.startswith("--share=")), None)
    sys.exit(main(args[0] if args else "model.gguf", want))
