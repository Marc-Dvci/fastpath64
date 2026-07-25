#!/usr/bin/env bash
# Cross-build ggml for aarch64, then run the repack equivalence gate under emulated Arm cores.
#
# Run me inside the fastpath64-cross image:
#   docker run --rm -v "<repo parent>:/src" fastpath64-cross bash /src/fastpath64/tools/qemu/run-equiv-test.sh
#
# QEMU is used for correctness only - it does not model microarchitecture, so nothing timed
# here is ever reported as a benchmark.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-/src/upstream-llama.cpp}
FP64_DIR=${FP64_DIR:-/src/fastpath64}
TYPES=${TYPES:-"iq4_xs q4_K"}

# ggml-cpu bakes the -march baseline in at compile time, so each emulated core needs a build
# whose baseline it can actually execute. Pairing them wrongly yields SIGILL, not a test result.
#   <qemu cpu>:<arm arch flag>
TARGETS=${TARGETS:-"neoverse-n2:armv8.6-a+i8mm+dotprod neoverse-n1:armv8.2-a+dotprod"}

status=0

for target in $TARGETS; do
    cpu=${target%%:*}
    arch=${target##*:}
    build="/tmp/b-${cpu}"

    echo
    echo "================ ${cpu}  (-march=${arch}) ================"

    cmake -S "$LLAMA_DIR" -B "$build" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$FP64_DIR/tools/qemu/aarch64-toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$arch" > "/tmp/cfg-${cpu}.log" 2>&1

    cmake --build "$build" --target ggml ggml-base ggml-cpu -j"$(nproc)" > "/tmp/bld-${cpu}.log" 2>&1

    aarch64-linux-gnu-g++ -O2 -std=c++17 \
        -I"$LLAMA_DIR/ggml/include" \
        "$FP64_DIR/tools/qemu/test_repack_equiv.cpp" \
        -L"$build/bin" -lggml -lggml-base -lggml-cpu \
        -Wl,-rpath,"$build/bin" \
        -o "/tmp/test_repack_equiv-${cpu}"

    # N K M - M=1 is gemv only, M=4/8/16 gemm only, M=5/9 exercise both paths plus the
    # remainder handling. K spans one and several QK_K super-blocks.
    SHAPES=${SHAPES:-"64:512:1 64:512:4 64:512:5 64:256:8 128:1024:9 8:256:16"}

    for t in $TYPES; do
        for shape in $SHAPES; do
            N=${shape%%:*}; rest=${shape#*:}; K=${rest%%:*}; M=${rest##*:}
            echo
            echo "---- QEMU_CPU=$cpu  type=$t  N=$N K=$K M=$M ----"
            if ! QEMU_CPU="$cpu" qemu-aarch64 -L /usr/aarch64-linux-gnu \
                    -E LD_LIBRARY_PATH="$build/bin" "/tmp/test_repack_equiv-${cpu}" "$t" "$N" "$K" "$M"; then
                status=1
            fi
        done
    done
done

echo
if [ "$status" = 0 ]; then echo "ALL CHECKS PASSED"; else echo "FAILURES PRESENT"; fi
exit $status
