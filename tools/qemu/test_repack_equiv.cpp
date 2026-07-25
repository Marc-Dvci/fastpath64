// Equivalence gate for repacked weight kernels.
//
// The repack fast path is only reachable when a weight tensor is allocated in the CPU backend's
// "extra" repack buffer type - which is why upstream's test-backend-ops never exercises it: that
// harness allocates into the default CPU buffer. This test allocates the *same* quantized weights
// twice, once in each buffer type, runs MUL_MAT on both, and compares.
//
// A repack kernel is a pure layout optimisation: it must produce the same numbers as the
// reference path. Any difference beyond fp accumulation-order noise is a bug.
//
// Usage: test_repack_equiv [type] [N] [K] [M]
//   type: iq4_xs (default), q4_K, iq4_nl, ...

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static ggml_backend_buffer_type_t find_repack_buft(ggml_backend_dev_t dev) {
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra_bufts =
        (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra_bufts) {
        return nullptr;
    }
    for (auto ** p = get_extra_bufts(dev); p && *p; p++) {
        const char * name = ggml_backend_buft_name(*p);
        if (name && std::string(name).find("REPACK") != std::string::npos) {
            return *p;
        }
    }
    return nullptr;
}

// Runs dst = a * b with the weights `a` placed in `buft`, returning the result rows.
static std::vector<float> run_mul_mat(ggml_backend_t backend,
                                      ggml_backend_buffer_type_t buft,
                                      ggml_type wtype,
                                      int64_t K, int64_t N, int64_t M,
                                      const std::vector<uint8_t> & wdata,
                                      const std::vector<float> & adata,
                                      bool * placed_in_buft) {
    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * wctx = ggml_init(wp);

    ggml_tensor * w = ggml_new_tensor_2d(wctx, wtype, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) {
        ggml_free(wctx);
        return {};
    }
    // init_tensor attaches repack traits in ->extra iff some kernel claims this type on this CPU.
    // Check before set_tensor: writing into the repack buffer with no traits attached is invalid.
    const bool claimed = (w->extra != nullptr);
    if (placed_in_buft) {
        *placed_in_buft = claimed;
    }
    if (!claimed && placed_in_buft) {
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        return {};
    }

    // set_tensor is what triggers the repack conversion for the repack buffer type
    ggml_backend_tensor_set(w, wdata.data(), 0, wdata.size());

    ggml_init_params cp = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * cctx = ggml_init(cp);

    ggml_tensor * a   = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, K, M);
    ggml_tensor * out = ggml_mul_mat(cctx, w, a);

    ggml_cgraph * gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    ggml_backend_tensor_set(a, adata.data(), 0, adata.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }

    std::vector<float> res(N * M);
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));

    ggml_backend_buffer_free(cbuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cctx);
    ggml_free(wctx);
    return res;
}

int main(int argc, char ** argv) {
    const std::string tname = argc > 1 ? argv[1] : "iq4_xs";
    const int64_t N = argc > 2 ? atoll(argv[2]) : 64;    // output rows (must be % 8 == 0)
    const int64_t K = argc > 3 ? atoll(argv[3]) : 512;   // reduction dim (must be % QK_K == 0)
    const int64_t M = argc > 4 ? atoll(argv[4]) : 5;     // activation rows: >1 exercises GEMM

    ggml_type wtype = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const auto * tt = ggml_get_type_traits((ggml_type) t);
        if (tt && tt->type_name && tname == tt->type_name) {
            wtype = (ggml_type) t;
            break;
        }
    }
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown type '%s'\n", tname.c_str());
        return 2;
    }

    printf("type=%s N=%lld K=%lld M=%lld\n", tname.c_str(), (long long) N, (long long) K, (long long) M);

    // deterministic pseudo-random weights and activations
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> wf32(N * K);
    for (auto & v : wf32) v = dist(rng);
    std::vector<float> adata(M * K);
    for (auto & v : adata) v = dist(rng);

    std::vector<uint8_t> wdata(ggml_row_size(wtype, K) * N);
    ggml_quantize_chunk(wtype, wf32.data(), wdata.data(), 0, N, K, nullptr);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);

    ggml_backend_buffer_type_t def_buft    = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t repack_buft = find_repack_buft(dev);

    if (!repack_buft) {
        printf("SKIP: no repack buffer type on this build\n");
        return 0;
    }
    printf("repack buffer type: %s\n", ggml_backend_buft_name(repack_buft));

    bool took_fast_path = false;
    std::vector<float> ref  = run_mul_mat(backend, def_buft,    wtype, K, N, M, wdata, adata, nullptr);
    std::vector<float> fast = run_mul_mat(backend, repack_buft, wtype, K, N, M, wdata, adata, &took_fast_path);

    if (!took_fast_path) {
        printf("SKIP: %s is not claimed by the repack path on this CPU "
               "(no matching kernel for the detected features)\n", tname.c_str());
        return 0;
    }
    if (fast.empty() || ref.empty()) {
        fprintf(stderr, "FAIL: allocation failed\n");
        return 1;
    }

    double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double) ref[i] - (double) fast[i]);
        const double denom = std::fmax(1e-6, std::fabs((double) ref[i]));
        max_abs = std::fmax(max_abs, d);
        max_rel = std::fmax(max_rel, d / denom);
        sum_sq += d * d;
    }
    const double rmse = std::sqrt(sum_sq / ref.size());

    printf("took repack fast path: yes\n");
    printf("max_abs=%.3e  max_rel=%.3e  rmse=%.3e\n", max_abs, max_rel, rmse);

    // Pure layout change: only fp accumulation order may differ.
    const bool ok = (max_rel < 1e-4) || (max_abs < 1e-3);
    printf("%s\n", ok ? "PASS" : "FAIL");

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
