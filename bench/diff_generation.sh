#!/usr/bin/env bash
# Greedy-decode the same prompt on both builds and diff the generated text.
#
# The equivalence test proves the kernels agree numerically; this proves the thing a user cares
# about - that the model says the same words. Reported, never asserted: the kernels differ in
# float accumulation order by ~1e-6, so an argmax flip is possible in principle. If it happens we
# say so rather than quietly claiming identical output.
set -uo pipefail

STOCK_BIN=${1:?usage: diff_generation.sh <stock llama-cli> <patched llama-cli> <model> <prompt> [n_predict]}
PATCHED_BIN=${2:?}
MODEL=${3:?}
PROMPT=${4:?}
N=${5:-96}

run() {
    "$1" -m "$MODEL" -f "$PROMPT" -n "$N" --temp 0 --seed 0 -no-cnv --no-warmup \
        -t "$(nproc)" 2>/dev/null
}

echo "generating on stock..."
run "$STOCK_BIN" > /tmp/gen-stock.txt
echo "generating on fastpath..."
run "$PATCHED_BIN" > /tmp/gen-patched.txt

echo
echo "## Generated output, greedy decode, same seed"
echo
if diff -q /tmp/gen-stock.txt /tmp/gen-patched.txt > /dev/null; then
    echo "**Byte-identical.** Stock and FastPath64 produced exactly the same $N tokens."
    echo
    echo '```'
    tail -c 600 /tmp/gen-patched.txt
    echo '```'
    exit 0
else
    echo "Outputs differ. This is possible in principle - the kernels differ in float"
    echo "accumulation order - so the divergence is shown rather than hidden:"
    echo
    echo '```diff'
    diff /tmp/gen-stock.txt /tmp/gen-patched.txt | head -40
    echo '```'
    # not a failure: numerical equivalence is gated by test_repack_equiv, not by argmax stability
    exit 0
fi
