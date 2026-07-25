#!/usr/bin/env python3
"""Compare two llama-bench CSVs (stock vs patched) produced on the same machine.

Reports prefill and decode separately, because they answer different questions:
prefill is compute-bound and is where an smmla kernel can help; decode is
memory-bandwidth-bound and is expected to be flat. A decode "win" here would be
noise, and a decode loss is a regression worth failing on.
"""

import argparse
import csv
import sys
from collections import defaultdict


def case_label(row):
    n_prompt = int(row.get("n_prompt") or 0)
    n_gen = int(row.get("n_gen") or 0)
    if n_prompt and not n_gen:
        return f"pp{n_prompt}", "prefill"
    if n_gen and not n_prompt:
        return f"tg{n_gen}", "decode"
    return f"pp{n_prompt}+tg{n_gen}", "mixed"


def quant_of(row):
    hay = f"{row.get('model_type','')} {row.get('model_filename','')}".lower()
    for q in ("iq4_xs", "q4_k", "q4_0", "iq4_nl", "q6_k", "q5_k", "mxfp4"):
        if q in hay:
            return q
    return "?"


def model_of(row):
    name = (row.get("model_filename") or "").rsplit("/", 1)[-1]
    return name.removesuffix(".gguf")


def load(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            case, kind = case_label(r)
            try:
                out[(model_of(r), case)] = (float(r["avg_ts"]), float(r.get("stddev_ts") or 0), kind, quant_of(r))
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("patched")
    args = ap.parse_args()

    a = load(args.stock)
    b = load(args.patched)

    shared = sorted(set(a) & set(b))
    if not shared:
        print("_no comparable rows_")
        return 1

    print("## stock vs FastPath64 — same runner, same GGUF\n")

    by_kind = defaultdict(list)
    for key in shared:
        by_kind[a[key][2]].append(key)

    worst_decode = None
    best_prefill = None

    for kind in ("prefill", "decode", "mixed"):
        keys = by_kind.get(kind)
        if not keys:
            continue
        print(f"### {kind}\n")
        print("| model | quant | case | stock t/s | fastpath t/s | change |")
        print("|---|---|---|---:|---:|---:|")
        for key in keys:
            model, case = key
            s_ts, s_sd, _, quant = a[key]
            p_ts, p_sd, _, _ = b[key]
            ratio = p_ts / s_ts if s_ts else float("nan")
            mark = ""
            if kind == "prefill" and quant == "iq4_xs":
                mark = " **<-**"
                if best_prefill is None or ratio > best_prefill[1]:
                    best_prefill = (model, ratio)
            if kind == "decode":
                if worst_decode is None or ratio < worst_decode[1]:
                    worst_decode = (model, ratio)
            print(
                f"| `{model}` | {quant} | {case} | {s_ts:,.2f} ±{s_sd:.2f} | "
                f"{p_ts:,.2f} ±{p_sd:.2f} | **{ratio:.2f}x**{mark} |"
            )
        print()

    print("---\n")
    if best_prefill:
        print(f"- best IQ4_XS prefill: **{best_prefill[1]:.2f}x** on `{best_prefill[0]}`")
    if worst_decode:
        flag = "" if worst_decode[1] >= 0.97 else "  :warning: **decode regression**"
        print(f"- worst decode: {worst_decode[1]:.2f}x on `{worst_decode[0]}`{flag}")
    print(
        "\nRows not marked `<-` are controls: formats FastPath64 does not touch should land at "
        "~1.00x. If they move, the run is measuring something other than the kernel."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
