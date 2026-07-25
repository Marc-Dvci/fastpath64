# Cross-build for aarch64 at native x86 speed, then execute under QEMU user-mode.
#
# Building llama.cpp *inside* emulation takes the better part of an hour; cross-compiling
# on the host arch and only emulating execution takes minutes. QEMU also lets us select the
# emulated core, which is how we reach code paths no free Arm CI runner can reach:
#
#   QEMU_CPU=max           -> SVE2 + I8MM  (what GitHub's Neoverse N2 runners give us)
#   QEMU_CPU=neoverse-n1   -> DOTPROD, no I8MM, no SVE  (Ampere Altra / AWS Graviton2)
#   QEMU_CPU=cortex-a72    -> plain ASIMD, no DOTPROD   (oldest fallback)
#
# Correctness only. QEMU does not model microarchitecture and no timing from it is ever
# reported as a benchmark.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        libc6-dev-arm64-cross \
        qemu-user \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
