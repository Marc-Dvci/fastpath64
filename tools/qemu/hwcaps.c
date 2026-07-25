// Prints the AArch64 hardware capabilities that decide which llama.cpp repack kernel runs.
//
// ggml selects kernels at runtime from exactly these bits (see ggml_cpu_has_matmul_int8 /
// ggml_cpu_has_dotprod / ggml_cpu_has_sve). Running this under `QEMU_CPU=<core>` tells us
// which code path a given emulated core will take, before we spend CI minutes on it.

#include <stdio.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1 << 14)
#endif

int main(void) {
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    int dotprod = (hwcap  & HWCAP_ASIMDDP) != 0;
    int i8mm    = (hwcap2 & HWCAP2_I8MM)   != 0;
    int sve     = (hwcap  & HWCAP_SVE)     != 0;
    int sve2    = (hwcap2 & HWCAP2_SVE2)   != 0;
    int bf16    = (hwcap2 & HWCAP2_BF16)   != 0;

    printf("dotprod=%d i8mm=%d sve=%d sve2=%d bf16=%d  -> ", dotprod, i8mm, sve, sve2, bf16);

    // mirrors the dispatch order in ggml_repack_get_optimal_repack_type()
    if (i8mm) {
        printf("8x8 smmla path\n");
    } else if (dotprod) {
        printf("8x4 sdot path\n");
    } else {
        printf("generic path (no repack)\n");
    }
    return 0;
}
