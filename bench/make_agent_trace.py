#!/usr/bin/env python3
"""Generate a realistic agent-turn prompt.

Agent traffic looks nothing like chat benchmarks. A single tool-calling turn re-sends a large,
mostly-static context - system prompt, tool schemas, conversation history, retrieved documents,
previous tool output - and then emits a few dozen tokens of structured JSON. That makes an agent
turn overwhelmingly *prefill*, which is exactly the compute-bound regime an smmla kernel changes.

Deterministic: same arguments always produce the same prompt, so A/B runs are comparable.
"""

import argparse
import random

TOOLS = [
    ("search_documents", "Search the knowledge base for relevant passages",
     {"query": "string", "top_k": "integer", "filters": "object"}),
    ("read_file", "Read a file from the workspace",
     {"path": "string", "start_line": "integer", "end_line": "integer"}),
    ("write_file", "Write content to a file in the workspace",
     {"path": "string", "content": "string", "create_dirs": "boolean"}),
    ("run_sql", "Execute a read-only SQL query against the analytics warehouse",
     {"query": "string", "timeout_seconds": "integer"}),
    ("send_email", "Send an email on behalf of the user",
     {"to": "string", "subject": "string", "body": "string", "cc": "array"}),
    ("create_ticket", "Open a ticket in the issue tracker",
     {"project": "string", "title": "string", "body": "string", "labels": "array"}),
    ("fetch_url", "Fetch and extract the readable content of a web page",
     {"url": "string", "render_js": "boolean"}),
    ("list_calendar", "List calendar events in a time range",
     {"start": "string", "end": "string", "calendar_id": "string"}),
]

DOC_SENTENCES = [
    "The ingestion pipeline batches records into windows of up to two thousand events.",
    "Retries use exponential backoff with a jitter factor drawn uniformly from zero to one.",
    "Schema migrations are applied ahead of deployment and must remain backward compatible.",
    "Metrics are sampled once per second and aggregated into one-minute buckets downstream.",
    "Access to the warehouse is mediated by short-lived credentials issued per session.",
    "Any request exceeding the latency budget is shed rather than queued indefinitely.",
    "Partition keys are derived from the tenant identifier hashed with a stable seed.",
    "Cold storage tiers are reconciled nightly against the primary index for drift.",
]


def tool_block() -> str:
    out = ["You have access to the following tools. Respond with a JSON tool call.\n"]
    for name, desc, params in TOOLS:
        out.append(f'{{"name": "{name}", "description": "{desc}", "parameters": {{')
        out.append(",".join(f'"{k}": {{"type": "{v}"}}' for k, v in params.items()))
        out.append("}}\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-words", type=int, default=4000,
                    help="approximate prompt size in words (~1.3 tokens/word)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--out", default="-")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    parts = [
        "You are an operations assistant embedded in an internal platform team's tooling. "
        "You plan carefully, call exactly one tool at a time, and never fabricate data. "
        "When a tool result is insufficient, you say so and request the next tool.\n\n",
        tool_block(),
        "\n--- retrieved context ---\n",
    ]

    # retrieved documents: the bulk of the prompt, as in a real RAG-backed agent
    doc_id = 0
    while sum(len(p.split()) for p in parts) < args.target_words:
        doc_id += 1
        parts.append(f"\n[doc-{doc_id:03d}] runbook excerpt\n")
        for _ in range(rng.randint(4, 9)):
            parts.append(rng.choice(DOC_SENTENCES) + " ")

    parts.append(
        "\n\n--- conversation ---\n"
        "user: The nightly reconciliation job failed again. Find out which partition drifted "
        "and open a ticket with the details.\n"
        "assistant: I'll query the warehouse for the reconciliation results first.\n"
        'tool_result: {"rows": [{"partition": "tenant_a4f2", "expected": 88214, "actual": 88190}]}\n'
        "user: Good. Now open the ticket.\n\n"
        "Respond with the next tool call as JSON only.\n"
    )

    text = "".join(parts)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        words = len(text.split())
        print(f"wrote {args.out}: {words} words (~{int(words * 1.3)} tokens)")


if __name__ == "__main__":
    main()
