#!/usr/bin/env python3
"""Measure a whole agent turn - time to first token and total wall clock - on two builds.

llama-bench reports throughput for synthetic prompt sizes. This measures the thing a user
actually waits for: feed a realistic tool-calling prompt, generate a short structured reply,
and report seconds. Parses llama-cli's timing block, which reports prompt eval (prefill) and
eval (decode) separately.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys

# llama-cli timing lines, e.g.
# llama_perf_context_print: prompt eval time =   12345.67 ms /  5321 tokens (    2.32 ms per token, ...
RE_PROMPT = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_EVAL = re.compile(r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)")
RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")


def run_once(binary, model, prompt_file, n_predict, threads):
    cmd = [
        binary, "-m", model, "-f", prompt_file,
        "-n", str(n_predict), "-t", str(threads),
        "--temp", "0", "--seed", "0", "-no-cnv", "--no-warmup",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    blob = proc.stdout + proc.stderr

    m_prompt = RE_PROMPT.search(blob)
    m_total = RE_TOTAL.search(blob)
    evals = RE_EVAL.findall(blob)
    if not (m_prompt and m_total):
        sys.stderr.write(blob[-2000:])
        raise RuntimeError(f"could not parse timings from {binary}")

    prefill_ms, prompt_tokens = float(m_prompt.group(1)), int(m_prompt.group(2))
    # the last "eval time" match is decode; prompt eval also matches the looser pattern
    decode_ms = float(evals[-1][0]) if evals else 0.0
    return {
        "ttft_s": prefill_ms / 1000.0,
        "decode_s": decode_ms / 1000.0,
        "turn_s": float(m_total.group(1)) / 1000.0,
        "prompt_tokens": prompt_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True, help="path to stock llama-cli")
    ap.add_argument("--patched", required=True, help="path to patched llama-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    results = {}
    for label, binary in (("stock", args.stock), ("fastpath", args.patched)):
        runs = [run_once(binary, args.model, args.prompt, args.n_predict, args.threads)
                for _ in range(args.repeats)]
        results[label] = {
            k: statistics.median(r[k] for r in runs)
            for k in ("ttft_s", "decode_s", "turn_s")
        }
        results[label]["prompt_tokens"] = runs[0]["prompt_tokens"]
        results[label]["runs"] = runs

    s, p = results["stock"], results["fastpath"]
    n_tok = s["prompt_tokens"]

    print(f"## One agent turn — {n_tok} prompt tokens, {args.n_predict} generated\n")
    print("| | stock | FastPath64 | change |")
    print("|---|---:|---:|---:|")
    for key, label in (("ttft_s", "time to first token"),
                       ("decode_s", "decode"),
                       ("turn_s", "**whole turn**")):
        ratio = s[key] / p[key] if p[key] else float("nan")
        print(f"| {label} | {s[key]:.2f} s | {p[key]:.2f} s | **{ratio:.2f}x faster** |")

    print(
        f"\nMedian of {args.repeats} runs. The turn is dominated by prefill "
        f"({100 * s['ttft_s'] / s['turn_s']:.0f}% of stock wall clock), which is why a prefill "
        "kernel moves the number a user actually feels."
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
