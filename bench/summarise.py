#!/usr/bin/env python3
"""Turn llama-bench CSV into a markdown summary aimed at one question:

    how much prefill throughput does IQ4_XS give up versus Q4_K on this machine?

Prefill (pp) is compute-bound and is where a missing smmla kernel shows up.
Decode (tg) is memory-bandwidth-bound and is expected to be roughly flat - if tg
moves a lot, something other than the kernel path changed and the run is suspect.

CSV schema per llama-bench get_fields() (tools/llama-bench/llama-bench.cpp:1570):
one row per case, already averaged over -r repetitions, carrying
model_filename, model_type, n_prompt, n_gen, avg_ts, stddev_ts.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict


def case_label(row):
    """llama-bench emits no 'test' column in CSV - rebuild it from n_prompt/n_gen."""
    n_prompt = int(row.get("n_prompt") or 0)
    n_gen = int(row.get("n_gen") or 0)
    if n_prompt and not n_gen:
        return f"pp{n_prompt}", "prefill"
    if n_gen and not n_prompt:
        return f"tg{n_gen}", "decode"
    return f"pp{n_prompt}+tg{n_gen}", "mixed"


QUANTS = ("iq4_xs", "q4_k", "q4_0", "iq4_nl", "q6_k", "q5_k", "mxfp4")


def quant_of(row):
    """model_type looks like 'Q4_K - Medium' or 'IQ4_XS - 4.25 bpw'."""
    hay = f"{row.get('model_type','')} {row.get('model_filename','')}".lower()
    for q in QUANTS:  # iq4_xs before q4_k: 'iq4_xs' must not match as 'q4_...'
        if q in hay:
            return q
    return "?"


def base_of(row):
    """Strip quant markers off the filename to pair models across formats."""
    name = (row.get("model_filename") or "").lower()
    name = name.rsplit("/", 1)[-1].removesuffix(".gguf")
    name = re.sub(r"[-_.](iq4[-_]?xs|q4[-_]?k([-_]?[ms])?|q4[-_]?0|iq4[-_]?nl)\b", "", name)
    return name.strip("-_. ") or "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    try:
        with open(args.csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        print(f"_no results file at `{args.csv_path}`_")
        return 1
    if not rows:
        print("_result file was empty_")
        return 1

    if args.label:
        print(f"### {args.label}\n")

    cpu = rows[0].get("cpu_info", "").strip()
    threads = rows[0].get("n_threads", "?")
    if cpu:
        print(f"`{cpu}` | {threads} threads | build `{rows[0].get('build_commit','?')}`\n")

    print("| model | quant | case | t/s | stddev |")
    print("|---|---|---|---:|---:|")
    table = {}
    for r in rows:
        case, kind = case_label(r)
        quant = quant_of(r)
        base = base_of(r)
        try:
            ts = float(r.get("avg_ts") or "nan")
            sd = float(r.get("stddev_ts") or 0.0)
        except ValueError:
            continue
        print(
            f"| `{base}` | {quant} | {case} | {ts:,.2f} | ±{sd:,.2f} |"
        )
        table[(base, case, quant)] = (ts, kind)

    # pair iq4_xs against q4_k for the same base model and case
    pairs = defaultdict(dict)
    for (base, case, quant), (ts, kind) in table.items():
        if quant in ("iq4_xs", "q4_k"):
            pairs[(base, case, kind)][quant] = ts

    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    if not complete:
        print("\n_no IQ4_XS/Q4_K pair found in this run - check the model downloads._")
        return 0

    print("\n#### IQ4_XS relative to Q4_K\n")
    print("| model | case | q4_k t/s | iq4_xs t/s | ratio |")
    print("|---|---|---:|---:|---:|")
    flagged = False
    for (base, case, kind), v in sorted(complete.items()):
        ratio = v["iq4_xs"] / v["q4_k"] if v["q4_k"] else float("nan")
        flag = ""
        if kind == "prefill" and ratio < 0.9:
            flag = " :warning:"
            flagged = True
        print(
            f"| `{base}` | {case} | {v['q4_k']:,.2f} | {v['iq4_xs']:,.2f} | **{ratio:.2f}x**{flag} |"
        )

    if flagged:
        print(
            "\n:warning: on a prefill row: IQ4_XS is materially slower than Q4_K despite being "
            "the *smaller* format. That is the signature of the missing repack/smmla fast path "
            "(see [docs/the-gap.md](../docs/the-gap.md))."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
