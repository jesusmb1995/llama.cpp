// Sweeps GGML_VK_TBQ_COPY_SG_SIZE = {0, 8, 16, 32, 64} and, for each value,
// runs the f32 -> {TBQ3_0, TBQ4_0, PQ3_0, PQ4_0, *_64} copy-to-quantize
// kernel on the Vulkan backend and compares the resulting quantized bytes
// against the CPU ggml_quantize_chunk reference.
//
// Why this test exists:
//   copy_to_quant.comp's cooperative TBQ/PQ path is a 32-thread workgroup
//   that uses subgroupAdd / subgroupBallot. On hardware with
//   gl_SubgroupSize < 32 (Intel Xe/Arc at 8/16, ARM Mali, Qualcomm Adreno,
//   some AMD configurations) those ops reduce within a subgroup, not the
//   whole workgroup, so the original shader silently produced wrong bytes.
//   The shader is now parameterized on the SG_SIZE spec constant and takes
//   a shared-memory "stitch" path for SG_SIZE < 32. This test exercises
//   the stitch path on devices that only have one native subgroup size,
//   by forcing the pipeline's requiredSubgroupSize + SG_SIZE spec const to
//   8/16/etc.
//
// How it works:
//   Since GGML_VK_TBQ_COPY_SG_SIZE is consumed at Vulkan device init (once
//   per process), we need a separate process per SG value. This binary
//   self-spawns: in "child" mode it runs exactly one (SG, type) combination
//   and prints a machine-readable summary line. In "parent" mode it forks
//   itself for every combination and aggregates the results.
//
// Accuracy metric:
//   The GPU and CPU quantizers compute the same math but in different float
//   orders (horizontal reductions across subgroups vs. scalar sums), so
//   byte-exact equality is not guaranteed -- we report both "bytes match"
//   and a dequantize NMSE against the f32 input, and we also compare GPU
//   dequantize vs. CPU dequantize so SG==32 vs SG==8 is a direct numerical
//   comparison. The fast-path (SG>=32) and stitch-path (SG<32) are required
//   to land within a tight NMSE tolerance of each other: if the stitch
//   implementation is wrong, NMSE blows up to O(1).
//
// Performance:
//   For each (SG, type) we time N repetitions of the copy on the device and
//   report ms/iter and GB/s (input bytes). This is not a rigorous benchmark
//   -- the intent is to catch order-of-magnitude regressions (e.g. a SG=8
//   stitch that barriers too aggressively) rather than to tune performance.

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <process.h>
#    define POPEN  _popen
#    define PCLOSE _pclose
#else
#    include <unistd.h>
#    define POPEN  popen
#    define PCLOSE pclose
#endif

namespace {

// ---------------------------------------------------------------------------
// Shared config
// ---------------------------------------------------------------------------

// Workgroup is 32 threads, block is 128 (or 64 for _64 variants). Use a
// shape that is large enough to amortize launch overhead and to dispatch
// many workgroups, but small enough that the test runs in < 1 s.
struct Shape {
    int64_t      ne0;  // fastest-moving (row length, multiple of block size)
    int64_t      ne1;  // number of rows
    const char * label;
};

// Shapes are shared between the SG sweep (which iterates indices {0, 1}) and
// the WG sweep (which iterates all of {0, 1, 2}). The SG sweep deliberately
// skips "huge" because (a) it doesn't need the per-iter stability the WG
// sweep does, and (b) it already spawns ~48-96 children per leg -- adding
// huge would push each leg past 5 minutes.
static const std::array<Shape, 3> kShapes = {
    {
     // Small: ~0.5 MB f32 input. Good for timing overhead checks.
        { 512, 256, "small" },
     // Medium: ~8 MB f32 input. Dispatches enough workgroups to keep a
        // small iGPU busy, and touches enough blocks to catch cross-subgroup
        // stitch bugs that don't repro on just a couple of blocks.
        { 2048, 1024, "medium" },
     // Huge: ~64 MB f32 input. Used only by --wg-sweep. Chosen so each GPU
        // iter takes >= ~2 ms on a typical iGPU, which makes the timing loop
        // robust against (a) CPU-side dispatch overhead (~50 us/iter doesn't
        // matter when the GPU does 2+ ms of work) and (b) DVFS clock ramp
        // (the GPU stays in its top P-state for the whole timed window
        // instead of transitioning mid-measurement). ne0 is a multiple of
        // both 128 (QUANT_K) and 64 (QUANT_K_64) so the same shape indexes
        // work for both 128- and 64-block quant types.
        { 8192, 2048, "huge" },
     }
};

// Subset of kShapes used by the SG sweep (indices into kShapes).
// Keeps backwards compatibility with the child's "RESULT ... shape=X" line
// parsing when we grow the kShapes array.
static const std::array<size_t, 2> kSgSweepShapes = { 0u, 1u };

// Indices into kShapes used by --wg-sweep. Includes "huge" (index 2) so the
// WG-vs-CPU perf comparison runs on a shape large enough to drown out
// dispatch/DVFS noise; see kShapes[2] for rationale.
static const std::array<size_t, 3> kWgSweepShapeIndices = { 0u, 1u, 2u };

struct QType {
    ggml_type    t;
    const char * name;
    int          blck;
};

// block size here is the SHADER's BK, not ggml_blck_size. For TBQ/PQ
// the shader always processes BK elements per workgroup (BK=128 for
// the regular variants, BK=64 for the _64 variants). ne0 must be a
// multiple of this.
static const std::array<QType, 8> kTypes = {
    {
     { GGML_TYPE_TBQ3_0, "tbq3_0", 128 },
     { GGML_TYPE_TBQ4_0, "tbq4_0", 128 },
     { GGML_TYPE_PQ3_0, "pq3_0", 128 },
     { GGML_TYPE_PQ4_0, "pq4_0", 128 },
     { GGML_TYPE_TBQ3_0_64, "tbq3_0_64", 64 },
     { GGML_TYPE_TBQ4_0_64, "tbq4_0_64", 64 },
     { GGML_TYPE_PQ3_0_64, "pq3_0_64", 64 },
     { GGML_TYPE_PQ4_0_64, "pq4_0_64", 64 },
     }
};

// SG sizes to sweep. 0 means "leave GGML_VK_TBQ_COPY_SG_SIZE unset", i.e.
// let the backend pick its default (the hardware's native SG size on
// size-control devices, or the old SG_SIZE=32 hardcoded path otherwise).
// 4/8/16 exercise the stitch path; 32/64 exercise the fast path. 4 is the
// smallest value the shader's tq_sh_red scratch (sized TQ_WG/4 = 8) can
// accommodate (NSG = 32/4 = 8). Values that the current device does not
// expose in [subgroup_min_size, subgroup_max_size] are rejected host-side
// and the test records them as "skipped".
static const std::array<uint32_t, 6> kSgSizes = {
    { 0, 4, 8, 16, 32, 64 }
};

// Number of warm-up + timed iterations for the perf number. Defaults are
// deliberately small so the SG-sweep (which self-spawns ~48-96 children
// per leg) runs in ~20s per leg; the --wg-sweep parent overrides via the
// env vars below to get tighter numbers on ~40 children per leg.
static constexpr int kWarmupItersDefault = 2;
static constexpr int kTimedItersDefault  = 10;

// Read-once cache of the per-run iteration counts. GGML_TEST_TIMED_ITERS and
// GGML_TEST_WARMUP_ITERS override the defaults in the child. The parent sets
// these before popen() for runs where it wants more stable numbers (e.g. the
// WG sweep, where the GPU kernel is ~100us and 10 iters leaves ~20% noise).
static int read_env_int(const char * name, int fallback) {
    const char * s = getenv(name);
    if (!s || !*s) {
        return fallback;
    }
    int v = std::atoi(s);
    return v > 0 ? v : fallback;
}
static const int kWarmupIters = read_env_int("GGML_TEST_WARMUP_ITERS", kWarmupItersDefault);
static const int kTimedIters  = read_env_int("GGML_TEST_TIMED_ITERS", kTimedItersDefault);

// ---------------------------------------------------------------------------
// Helpers (shared between parent and child)
// ---------------------------------------------------------------------------

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0, denom = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        denom += (double) a[i] * (double) a[i];
    }
    if (denom == 0.0) {
        return 0.0;
    }
    return num / denom;
}

static double max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static size_t byte_mismatch_count(const uint8_t * a, const uint8_t * b, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ++c;
        }
    }
    return c;
}

// Deterministic Gaussian fill (matches the distribution of post-Hadamard
// rotated attention activations, which is what cpy_f32_tbq* actually sees
// in llama-perplexity).
static void fill_normal(std::vector<float> & v, uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto & x : v) {
        x = d(rng);
    }
}

// Pick a non-CPU backend; return nullptr if none. Accept GPU, IGPU, and
// ACCEL device types -- Vulkan on integrated graphics (e.g. AMD gfx1150)
// reports as IGPU, not GPU.
static ggml_backend_t pick_gpu_backend(std::string & name_out) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto         t   = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            t == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) {
                name_out = ggml_backend_dev_name(dev);
                return b;
            }
        }
    }
    return nullptr;
}

// Dequantize via the CPU type traits so we can compare f32 distributions.
static void dequantize(ggml_type t, const void * src, float * dst, int64_t nrows, int64_t ne0) {
    const auto * tt = ggml_get_type_traits(t);
    if (!tt || !tt->to_float) {
        std::fprintf(stderr, "dequantize: no to_float for %s\n", ggml_type_name(t));
        std::abort();
    }
    // to_float takes a count in f32 elements.
    for (int64_t r = 0; r < nrows; ++r) {
        const size_t row_bytes = ggml_row_size(t, ne0);
        tt->to_float((const char *) src + r * row_bytes, dst + r * ne0, ne0);
    }
}

// ---------------------------------------------------------------------------
// Child-process mode: run one (SG, type, shape) and print a result line.
// ---------------------------------------------------------------------------

struct ChildResult {
    bool   ok_run          = false;  // kernel executed without error
    bool   supported       = false;  // backend supports the op for this type
    size_t n_bytes         = 0;      // quantized output size in bytes
    size_t mismatch_bytes  = 0;      // bytes that differ from CPU reference
    double nmse_gpu_vs_cpu = 0.0;    // NMSE(dequant(gpu), dequant(cpu))
    double nmse_gpu_vs_src = 0.0;    // NMSE(dequant(gpu), src_f32)
    double nmse_cpu_vs_src = 0.0;    // NMSE(dequant(cpu), src_f32)  -- sanity
    double max_abs_vs_cpu  = 0.0;
    double ms_per_iter     = 0.0;
    double gb_per_s        = 0.0;
    // CPU reference timing. Only populated when run_one() is called with
    // measure_cpu=true (opt-in because the CPU quantize is 10-100x slower
    // than the GPU kernel and dominates total runtime). 0.0 otherwise.
    double cpu_ms_per_iter = 0.0;
    double cpu_gb_per_s    = 0.0;
};

static ChildResult run_one(ggml_backend_t backend, ggml_type qtype, int64_t ne0, int64_t ne1,
                           bool measure_cpu = false) {
    ChildResult r{};

    const int64_t nrows = ne1;
    const int64_t nels  = ne0 * nrows;

    // ---- Build a minimal graph: f32 input -> ggml_cpy -> qtype output.
    // Using ggml_cpy (not ggml_set_rows) because it's the common path and
    // it's what the host-side pipeline wiring in ggml-vulkan.cpp hits for
    // the regression workload (llama-perplexity KV-cache fill).
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, nrows);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), qtype, ne0, nrows);
    ggml_set_name(src, "src_f32");
    ggml_set_name(dst, "dst_q");
    ggml_tensor * cpy = ggml_cpy(ctx.get(), src, dst);
    ggml_set_name(cpy, "cpy");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        return r;
    }

    if (!ggml_backend_supports_op(backend, cpy)) {
        r.supported = false;
        return r;
    }
    r.supported = true;

    // ---- Generate input.
    std::vector<float> input(nels);
    fill_normal(input, 0xC0FFEEu ^ (uint32_t) qtype ^ (uint32_t) ne0 ^ (uint32_t) ne1);
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));

    // ---- CPU reference: row-by-row quantize_chunk.
    const size_t         row_bytes_q = ggml_row_size(qtype, ne0);
    std::vector<uint8_t> cpu_q(row_bytes_q * nrows);
    const size_t         blck = ggml_blck_size(qtype);
    auto cpu_quantize = [&]() {
        for (int64_t r_i = 0; r_i < nrows; ++r_i) {
            ggml_quantize_chunk(qtype, input.data() + r_i * ne0, cpu_q.data() + r_i * row_bytes_q, 0, ne0 / blck, blck,
                                nullptr);
        }
    };
    // First pass always happens (needed for correctness comparison below).
    cpu_quantize();
    if (measure_cpu) {
        // Time a few iterations to get a stable per-iter number. We reuse
        // the same single-threaded ggml_quantize_chunk loop the production
        // code falls back to when the GPU isn't used; this gives the "old
        // code no parallel" baseline the test asserts against. We cap CPU
        // iters to keep total runtime bounded: each CPU pass is ~30ms on
        // medium, so kTimedIters=200 would mean 6s per child; std-dev is
        // already low at 5 iters for a 30ms baseline, so we don't need
        // more. GPU and CPU iter counts are decoupled on purpose.
        const int cpu_iters = std::min(kTimedIters, 5);
        const auto tc0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cpu_iters; ++i) {
            cpu_quantize();
        }
        const auto tc1   = std::chrono::high_resolution_clock::now();
        const double cs  = std::chrono::duration<double>(tc1 - tc0).count();
        r.cpu_ms_per_iter = 1e3 * cs / cpu_iters;
        r.cpu_gb_per_s    = (double) (nels * sizeof(float)) / (cs / cpu_iters) / 1e9;
    }

    // ---- GPU quantize (warmup + timed).
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, cpy);

    for (int i = 0; i < kWarmupIters; ++i) {
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            return r;
        }
    }
    ggml_backend_synchronize(backend);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimedIters; ++i) {
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            return r;
        }
    }
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::high_resolution_clock::now();

    r.ok_run          = true;
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    r.ms_per_iter     = 1e3 * secs / kTimedIters;
    r.gb_per_s        = (double) (nels * sizeof(float)) / (secs / kTimedIters) / 1e9;

    // ---- Read back GPU output.
    std::vector<uint8_t> gpu_q(row_bytes_q * nrows);
    ggml_backend_tensor_get(dst, gpu_q.data(), 0, gpu_q.size());

    // ---- Accuracy metrics.
    r.n_bytes        = gpu_q.size();
    r.mismatch_bytes = byte_mismatch_count(gpu_q.data(), cpu_q.data(), gpu_q.size());

    std::vector<float> gpu_f32(nels), cpu_f32(nels);
    dequantize(qtype, gpu_q.data(), gpu_f32.data(), nrows, ne0);
    dequantize(qtype, cpu_q.data(), cpu_f32.data(), nrows, ne0);

    r.nmse_gpu_vs_cpu = nmse(cpu_f32.data(), gpu_f32.data(), nels);
    r.nmse_gpu_vs_src = nmse(input.data(), gpu_f32.data(), nels);
    r.nmse_cpu_vs_src = nmse(input.data(), cpu_f32.data(), nels);
    r.max_abs_vs_cpu  = max_abs_diff(cpu_f32.data(), gpu_f32.data(), nels);

    return r;
}

// Child mode entrypoint. Args: <qtype_index> <shape_index>.
// Prints a single line "RESULT ok=... supp=... mism=... nmse_gvsc=... ms=... gbps=..."
// so the parent can parse it trivially.
static int child_main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "child: expected 2 args (qtype_idx, shape_idx)\n");
        return 2;
    }
    const int qi = std::atoi(argv[1]);
    const int si = std::atoi(argv[2]);
    if (qi < 0 || qi >= (int) kTypes.size() || si < 0 || si >= (int) kShapes.size()) {
        std::fprintf(stderr, "child: bad indices qi=%d si=%d\n", qi, si);
        return 2;
    }

    ggml_backend_load_all();
    std::string    backend_name;
    ggml_backend_t backend = pick_gpu_backend(backend_name);
    if (!backend) {
        std::printf("RESULT ok=0 supp=0 skip=no_gpu\n");
        return 0;
    }

    const QType & qt          = kTypes[qi];
    const Shape & sh          = kShapes[si];
    // ne0 must be a multiple of the shader's BK (=ggml_blck_size).
    const int64_t ne0_aligned = (sh.ne0 / qt.blck) * qt.blck;

    // Opt-in CPU timing. The parent sets this env var for at most one child
    // per (type, shape) in the WG sweep (the WG=0 / default child) so we
    // don't pay the CPU quantize cost multiple times for the same input.
    const bool measure_cpu = []() {
        const char * e = getenv("GGML_TEST_MEASURE_CPU");
        return e && *e && e[0] != '0';
    }();

    ChildResult r = run_one(backend, qt.t, ne0_aligned, sh.ne1, measure_cpu);
    std::printf("RESULT backend=%s type=%s shape=%s ne0=%" PRId64 " ne1=%" PRId64
                " ok=%d supp=%d bytes=%zu mism=%zu nmse_gvsc=%.3e nmse_gvss=%.3e "
                "nmse_cvss=%.3e maxabs=%.3e ms=%.3f gbps=%.2f cpu_ms=%.3f cpu_gbps=%.2f\n",
                backend_name.c_str(), qt.name, sh.label, ne0_aligned, sh.ne1, r.ok_run ? 1 : 0, r.supported ? 1 : 0,
                r.n_bytes, r.mismatch_bytes, r.nmse_gpu_vs_cpu, r.nmse_gpu_vs_src, r.nmse_cpu_vs_src, r.max_abs_vs_cpu,
                r.ms_per_iter, r.gb_per_s, r.cpu_ms_per_iter, r.cpu_gb_per_s);

    ggml_backend_free(backend);
    return 0;
}

// ---------------------------------------------------------------------------
// Parent-process mode: spawn children with different SG env values.
// ---------------------------------------------------------------------------

struct ParsedLine {
    bool        present = false;
    bool        ok = false, supp = false;
    size_t      bytes = 0, mism = 0;
    double      nmse_gvsc = 0, nmse_gvss = 0, nmse_cvss = 0, maxabs = 0;
    double      ms = 0, gbps = 0;
    // Optional CPU reference timing (populated only for the child spawned
    // with GGML_TEST_MEASURE_CPU=1, which is at most one child per (type,
    // shape) in the WG sweep). 0 means "not measured".
    double      cpu_ms = 0, cpu_gbps = 0;
    std::string backend;
    // Parsed from "ggml_vulkan: tbq_copy_sg_size_status requested=R applied=A reason=X"
    // in child stderr. override_rejected is true when the child's requested SG
    // differs from what the backend actually applied (applied=0 with a non-zero
    // request, or applied != requested for any other reason). Used by the parent
    // to label these rows SKIPPED instead of OK, since they effectively ran at
    // the default SG and are duplicates of the sg=0 case.
    bool        status_seen         = false;
    uint32_t    status_requested    = 0;
    uint32_t    status_applied      = 0;
    std::string status_reason;
    bool        override_rejected() const {
        return status_seen && status_requested != 0 && status_applied != status_requested;
    }
};

static bool parse_key_double(const std::string & line, const std::string & key, double & out) {
    auto p = line.find(key + "=");
    if (p == std::string::npos) {
        return false;
    }
    out = std::atof(line.c_str() + p + key.size() + 1);
    return true;
}

static bool parse_key_size(const std::string & line, const std::string & key, size_t & out) {
    double d = 0;
    if (!parse_key_double(line, key, d)) {
        return false;
    }
    out = (size_t) d;
    return true;
}

static bool parse_key_int(const std::string & line, const std::string & key, int & out) {
    double d = 0;
    if (!parse_key_double(line, key, d)) {
        return false;
    }
    out = (int) d;
    return true;
}

static bool parse_key_str(const std::string & line, const std::string & key, std::string & out) {
    auto p = line.find(key + "=");
    if (p == std::string::npos) {
        return false;
    }
    p += key.size() + 1;
    auto q = line.find(' ', p);
    out    = line.substr(p, q == std::string::npos ? std::string::npos : (q - p));
    return true;
}

static ParsedLine run_child(const std::string & self_path, uint32_t sg, int qi, int si) {
    ParsedLine pl;

    // Build the command. We re-exec ourselves with --child to force child_main.
    // Env var sg==0 means "don't set the var at all".
    char cmd[2048];
    if (sg == 0) {
        std::snprintf(cmd, sizeof(cmd), "unset GGML_VK_TBQ_COPY_SG_SIZE; \"%s\" --child %d %d 2>&1", self_path.c_str(),
                      qi, si);
    } else {
        std::snprintf(cmd, sizeof(cmd), "GGML_VK_TBQ_COPY_SG_SIZE=%u \"%s\" --child %d %d 2>&1", sg, self_path.c_str(),
                      qi, si);
    }

    FILE * f = POPEN(cmd, "r");
    if (!f) {
        std::fprintf(stderr, "run_child: popen failed for cmd=%s\n", cmd);
        return pl;
    }
    char        buf[4096];
    std::string result_line;
    std::string status_line;  // ggml_vulkan: tbq_copy_sg_size_status ...
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string s = buf;
        // forward child output to parent stderr for debugging; keep only
        // the last RESULT line for parsing.
        std::fprintf(stderr, "[sg=%u %s] %s", sg, kTypes[qi].name, s.c_str());
        if (s.rfind("RESULT", 0) == 0) {
            result_line = s;
        }
        // The backend emits this once per device init when the env var is set.
        // We grep it out of the combined stdout+stderr stream popen() gave us.
        if (s.find("tbq_copy_sg_size_status") != std::string::npos) {
            status_line = s;
        }
    }
    PCLOSE(f);

    if (result_line.empty()) {
        return pl;
    }
    pl.present = true;
    int ok_i = 0, supp_i = 0;
    parse_key_int(result_line, "ok", ok_i);
    parse_key_int(result_line, "supp", supp_i);
    pl.ok   = ok_i != 0;
    pl.supp = supp_i != 0;
    parse_key_size(result_line, "bytes", pl.bytes);
    parse_key_size(result_line, "mism", pl.mism);
    parse_key_double(result_line, "nmse_gvsc", pl.nmse_gvsc);
    parse_key_double(result_line, "nmse_gvss", pl.nmse_gvss);
    parse_key_double(result_line, "nmse_cvss", pl.nmse_cvss);
    parse_key_double(result_line, "maxabs", pl.maxabs);
    parse_key_double(result_line, "ms", pl.ms);
    parse_key_double(result_line, "gbps", pl.gbps);
    parse_key_double(result_line, "cpu_ms", pl.cpu_ms);
    parse_key_double(result_line, "cpu_gbps", pl.cpu_gbps);
    parse_key_str(result_line, "backend", pl.backend);

    if (!status_line.empty()) {
        pl.status_seen = true;
        int req = 0, app = 0;
        parse_key_int(status_line, "requested", req);
        parse_key_int(status_line, "applied", app);
        pl.status_requested = (uint32_t) req;
        pl.status_applied   = (uint32_t) app;
        parse_key_str(status_line, "reason", pl.status_reason);
        // parse_key_str takes everything up to the next whitespace, but the
        // key is at end-of-line so it swallows the trailing '\n'. Strip any
        // trailing CR/LF before we print it into a table.
        while (!pl.status_reason.empty() &&
               (pl.status_reason.back() == '\n' || pl.status_reason.back() == '\r')) {
            pl.status_reason.pop_back();
        }
    }
    return pl;
}

// Resolve a comma-separated list of type names (e.g. "tbq3_0,pq3_0") against
// kTypes, returning the matching indices in their original kTypes order.
// Returns empty on no match (caller decides whether that's a hard error).
// Unknown names are reported on stderr and skipped -- we don't want a typo in
// a test script to silently run zero cases, but we also don't want a mismatch
// between a script and a newly renamed type to fail the whole leg.
static std::vector<size_t> resolve_type_filter(const std::string & csv) {
    std::vector<size_t> out;
    size_t              start = 0;
    while (start <= csv.size()) {
        size_t            end   = csv.find(',', start);
        const std::string token = csv.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty()) {
            bool found = false;
            for (size_t i = 0; i < kTypes.size(); ++i) {
                if (token == kTypes[i].name) {
                    out.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::fprintf(stderr, "warning: --types token '%s' does not match any known type; ignoring\n",
                             token.c_str());
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

static int parent_main(const std::string & self_path, const std::vector<size_t> & type_filter) {
    // Sanity: make sure a GPU backend is actually available at all.
    {
        ggml_backend_load_all();
        std::string    nm;
        ggml_backend_t b = pick_gpu_backend(nm);
        if (!b) {
            std::fprintf(stdout, "no GPU backend available -- skipping\n");
            return 0;
        }
        std::fprintf(stdout, "using backend: %s\n", nm.c_str());
        ggml_backend_free(b);
    }

    // Tolerance for NMSE(gpu_sgN_dequant, cpu_dequant). The quantization math
    // is IEEE-float non-associative, so we don't require bit-identity between
    // GPU orderings. 1e-6 is tight enough to catch a broken stitch (which
    // produces O(1) NMSE) and loose enough to tolerate float reordering.
    const double kNmseTol = 1e-6;

    int n_fail    = 0;
    int n_run     = 0;
    int n_skipped = 0;

    // Build the list of type indices to iterate. Empty filter == all types.
    std::vector<size_t> type_indices;
    if (type_filter.empty()) {
        for (size_t i = 0; i < kTypes.size(); ++i) {
            type_indices.push_back(i);
        }
    } else {
        type_indices = type_filter;
    }
    if (!type_filter.empty()) {
        std::fprintf(stdout, "type filter: ");
        for (size_t i = 0; i < type_indices.size(); ++i) {
            std::fprintf(stdout, "%s%s", i ? "," : "", kTypes[type_indices[i]].name);
        }
        std::fprintf(stdout, "\n");
    }

    // SG sweep only iterates the SG-intended shape subset; huge is reserved
    // for --wg-sweep where per-iter stability matters.
    for (size_t si : kSgSweepShapes) {
        for (size_t qi : type_indices) {
            std::fprintf(stdout, "\n=== %s %s ===\n", kTypes[qi].name, kShapes[si].label);
            // Reference = cpu dequantize: implicit via child's nmse_gvsc.
            // Additionally, within-GPU consistency: nmse_gvsc should be
            // (a) small for every SG, and (b) the same for all SGs modulo
            // reordering. If SG=8 disagrees with SG=32 by more than kNmseTol
            // the stitch path is broken.
            std::vector<ParsedLine> by_sg(kSgSizes.size());
            for (size_t k = 0; k < kSgSizes.size(); ++k) {
                by_sg[k] = run_child(self_path, kSgSizes[k], (int) qi, (int) si);
            }

            // Report. Rows where the child requested a non-default SG but the
            // backend rejected it (e.g. SG=8 on gfx1150 which only exposes
            // [32,64]) are labelled SKIPPED-<reason>: they ran at the device
            // default, so their result is identical to sg=0 and does not
            // provide independent coverage for the requested SG. Including
            // them as OK was misleading, so we now spell it out.
            std::fprintf(stdout, "  %-8s | %-18s | %12s | %12s | %9s | %9s\n", "sg", "status", "nmse(g v c)",
                         "nmse(g v s)", "ms/iter", "GB/s");
            for (size_t k = 0; k < kSgSizes.size(); ++k) {
                const auto & p = by_sg[k];
                char tag[32];
                if (!p.present) {
                    std::snprintf(tag, sizeof(tag), "NOPROC");
                } else if (!p.supp) {
                    std::snprintf(tag, sizeof(tag), "NOSUPP");
                } else if (p.override_rejected()) {
                    // e.g. "SKIPPED-out_of_range", "SKIPPED-unsupported_by_shader",
                    // "SKIPPED-no_size_control". kSgSizes[k] == 0 never triggers
                    // this branch because status_requested == 0 -> not rejected.
                    std::snprintf(tag, sizeof(tag), "SKIP-%s",
                                  p.status_reason.empty() ? "rejected" : p.status_reason.c_str());
                } else if (!p.ok) {
                    std::snprintf(tag, sizeof(tag), "FAIL");
                } else {
                    std::snprintf(tag, sizeof(tag), "OK");
                }
                std::fprintf(stdout, "  sg=%-5u | %-18s | %12.3e | %12.3e | %9.3f | %9.2f\n", kSgSizes[k], tag,
                             p.nmse_gvsc, p.nmse_gvss, p.ms, p.gbps);
            }

            // Decide pass/fail. Pass criteria:
            //   1. Every SG that ran, is supported, and had its override
            //      actually applied must have nmse_gvsc <= kNmseTol.
            //   2. nmse_gvsc across all such SGs must agree to within
            //      kNmseTol*10 (differences come from float reduction order only).
            //   3. At least one SG must have produced a valid result.
            // Rows whose override was rejected are excluded from the pass/fail
            // math (they are effectively duplicates of sg=0) but we DO count
            // them in the skipped tally so the summary is transparent.
            bool   any_applied = false, any_ok = false;
            double min_nmse = 1e300, max_nmse = -1e300;
            int    n_skipped_local = 0;
            for (size_t k = 0; k < kSgSizes.size(); ++k) {
                const auto & p = by_sg[k];
                if (!p.present || !p.supp) {
                    continue;
                }
                if (p.override_rejected()) {
                    ++n_skipped_local;
                    continue;
                }
                any_applied = true;
                if (!p.ok) {
                    std::fprintf(stderr, "  FAIL: sg=%u did not execute\n", kSgSizes[k]);
                    ++n_fail;
                    continue;
                }
                any_ok = true;
                ++n_run;
                if (p.nmse_gvsc > kNmseTol) {
                    std::fprintf(stderr, "  FAIL: sg=%u nmse(gpu vs cpu)=%.3e > %.1e\n", kSgSizes[k], p.nmse_gvsc,
                                 kNmseTol);
                    ++n_fail;
                }
                min_nmse = std::min(min_nmse, p.nmse_gvsc);
                max_nmse = std::max(max_nmse, p.nmse_gvsc);
            }
            n_skipped += n_skipped_local;
            if (any_applied && !any_ok) {
                std::fprintf(stderr, "  FAIL: no SG produced a valid result\n");
                ++n_fail;
            }
            if (any_ok && max_nmse - min_nmse > kNmseTol * 10.0) {
                std::fprintf(stderr, "  FAIL: nmse spread across SG sizes %.3e > %.1e\n", max_nmse - min_nmse,
                             kNmseTol * 10.0);
                ++n_fail;
            }
        }
    }

    // "ran" counts rows whose requested SG was applied and produced an NMSE we
    // actually checked; "skipped" counts rows the backend rejected. On an
    // AMD-RDNA box with [min=32, max=64], you'll typically see sg=4/8/16
    // appearing under skipped and sg=0/32/64 under ran. If everything is
    // skipped except sg=0 the test passes with ran > 0 but gives no
    // stitch-path coverage -- that's honest reporting, not a failure.
    std::fprintf(stdout, "\n%s: ran=%d skipped=%d failed=%d\n", n_fail ? "FAILED" : "PASSED", n_run, n_skipped,
                 n_fail);
    return n_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------
// --wg-sweep mode: opt-in, requires the build to have been configured with
// -DGGML_VULKAN_TEST_SHADERS=ON. For each (type, shape), spawn one child per
// WG size in {0, 2, 4, 8, 16}. WG=0 means "no override", i.e. the production
// pipeline (WG=32). One of the children (the WG=0 one) is additionally asked
// to measure CPU reference timing via GGML_TEST_MEASURE_CPU=1, giving us the
// "single-threaded CPU (the 'old pre-parallel' baseline)" number the other
// assertions check against. We intentionally time CPU just once per (type,
// shape) because ggml_quantize_chunk is ~10-100x slower than the GPU kernel
// and would dominate total runtime if we ran it in every child.
//
// About the per-WG timings: the intent of this sweep is to compare the
// cooperative GPU path against the pre-parallel CPU baseline, NOT to pick
// an optimal WG size. On real hardware, the mapping WG -> time is
// non-monotonic and has cliffs driven by driver-internal choices (wave
// packing, LDS banking, cache layout, scheduling) that are opaque to us.
// Even on the "huge" shape (where per-iter noise is <5%), WG=2 and WG=16
// sometimes beat WG=32, and WG=4 sometimes lands 3x slower than its
// neighbours -- these are repeatable and real, not measurement noise. The
// only claim this sweep makes, and the one the assertions enforce, is:
//   "every tested WG, including the most pessimal (WG=2), beats the
//    single-threaded CPU reference by >= 1.5x on non-trivial shapes."
// That's the "cooperative quantize is worth shipping at any width" story.
// Drawing conclusions about "which WG is optimal" from this data would be
// misleading -- use a GPU profiler (AMD RGA / RGP, NVIDIA Nsight) for that.
// ---------------------------------------------------------------------------

static ParsedLine run_wg_child(const std::string & self_path, uint32_t wg, bool measure_cpu,
                               int qi, int si) {
    ParsedLine pl;
    char       cmd[2048];
    // Compose env prefix: the WG override (0 == unset), plus optional CPU
    // timing flag. Unset any SG override from a containing env so this run
    // always exercises the production spec-constant default.
    //
    // Per-child iteration count, keyed to shape:
    //   small  (0.5 MB):  200 iters x ~0.1ms = ~20ms -- lots of repetitions
    //                     to drown dispatch noise on a tiny kernel.
    //   medium (  8 MB):  200 iters x ~0.3ms = ~60ms -- still cheap and
    //                     keeps per-iter std-dev low.
    //   huge   ( 64 MB):   50 iters x ~2-5ms = ~100-250ms. Each iter is
    //                     already long enough that dispatch/DVFS noise is
    //                     <5% of total; we don't need 200.
    // Keeping total per-child runtime in the 20-250ms range for all shapes
    // keeps the whole sweep under ~2 minutes.
    const char * label = kShapes[si].label;
    int          timed_iters, warmup_iters;
    if (std::strcmp(label, "huge") == 0) {
        timed_iters  = 50;
        warmup_iters = 5;
    } else {
        timed_iters  = 200;
        warmup_iters = 10;
    }
    char iter_env[128];
    std::snprintf(iter_env, sizeof(iter_env),
                  "GGML_TEST_WARMUP_ITERS=%d GGML_TEST_TIMED_ITERS=%d ",
                  warmup_iters, timed_iters);
    std::string env_prefix = std::string("unset GGML_VK_TBQ_COPY_SG_SIZE; ") + iter_env;
    if (wg == 0) {
        env_prefix += "unset GGML_VK_TBQ_COPY_WG_SIZE; ";
    } else {
        char tmp[64];
        std::snprintf(tmp, sizeof(tmp), "GGML_VK_TBQ_COPY_WG_SIZE=%u ", wg);
        env_prefix += tmp;
    }
    if (measure_cpu) {
        env_prefix += "GGML_TEST_MEASURE_CPU=1 ";
    } else {
        env_prefix += "unset GGML_TEST_MEASURE_CPU; ";
    }
    std::snprintf(cmd, sizeof(cmd), "%s \"%s\" --child %d %d 2>&1",
                  env_prefix.c_str(), self_path.c_str(), qi, si);

    FILE * f = POPEN(cmd, "r");
    if (!f) {
        std::fprintf(stderr, "run_wg_child: popen failed for cmd=%s\n", cmd);
        return pl;
    }
    char        buf[4096];
    std::string result_line;
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string s = buf;
        std::fprintf(stderr, "[wg=%u %s] %s", wg, kTypes[qi].name, s.c_str());
        if (s.rfind("RESULT", 0) == 0) {
            result_line = s;
        }
    }
    PCLOSE(f);

    if (result_line.empty()) {
        return pl;
    }
    pl.present = true;
    int ok_i = 0, supp_i = 0;
    parse_key_int(result_line, "ok", ok_i);
    parse_key_int(result_line, "supp", supp_i);
    pl.ok   = ok_i != 0;
    pl.supp = supp_i != 0;
    parse_key_size(result_line, "bytes", pl.bytes);
    parse_key_size(result_line, "mism", pl.mism);
    parse_key_double(result_line, "nmse_gvsc", pl.nmse_gvsc);
    parse_key_double(result_line, "nmse_gvss", pl.nmse_gvss);
    parse_key_double(result_line, "nmse_cvss", pl.nmse_cvss);
    parse_key_double(result_line, "maxabs", pl.maxabs);
    parse_key_double(result_line, "ms", pl.ms);
    parse_key_double(result_line, "gbps", pl.gbps);
    parse_key_double(result_line, "cpu_ms", pl.cpu_ms);
    parse_key_double(result_line, "cpu_gbps", pl.cpu_gbps);
    parse_key_str(result_line, "backend", pl.backend);
    return pl;
}

// Types this mode covers. Must match the WG-sweep SPVs built by vulkan-
// shaders-gen.cpp under #ifdef GGML_VULKAN_TEST_SHADERS. Keep these four
// in sync if that loop changes.
static const std::array<const char *, 4> kWgSweepTypes = {
    "tbq3_0", "pq3_0", "tbq3_0_64", "pq3_0_64",
};

// WG sizes we sweep. 0 means "no override" -> production WG=32 pipeline.
// The rest come from the test SPVs. Keep ascending so the table reads
// naturally (smallest WG first).
static const std::array<uint32_t, 5> kWgSizes = { 0u, 2u, 4u, 8u, 16u };

static int wg_sweep_main(const std::string & self_path) {
    // Sanity: GPU must be available.
    {
        ggml_backend_load_all();
        std::string    nm;
        ggml_backend_t b = pick_gpu_backend(nm);
        if (!b) {
            std::fprintf(stdout, "no GPU backend available -- skipping\n");
            return 0;
        }
        std::fprintf(stdout, "using backend: %s\n", nm.c_str());
        ggml_backend_free(b);
    }

    // Correctness tolerance: same as parent_main. The WG shrink only
    // changes float reduction order, so NMSE should stay within the same
    // ~1e-6 bound we use for the SG sweep.
    const double kNmseTol = 1e-6;

    int n_fail = 0;
    int n_run  = 0;

    for (size_t si : kWgSweepShapeIndices) {
        for (const char * tn : kWgSweepTypes) {
            // Resolve type name -> qi.
            int qi = -1;
            for (size_t i = 0; i < kTypes.size(); ++i) {
                if (std::strcmp(tn, kTypes[i].name) == 0) {
                    qi = (int) i;
                    break;
                }
            }
            if (qi < 0) {
                std::fprintf(stderr, "wg-sweep: type '%s' not in kTypes (harness bug)\n", tn);
                ++n_fail;
                continue;
            }

            std::fprintf(stdout, "\n=== %s %s ===\n", tn, kShapes[si].label);
            std::vector<ParsedLine> by_wg(kWgSizes.size());
            for (size_t k = 0; k < kWgSizes.size(); ++k) {
                // Measure CPU timing in exactly one child (the first, WG=0)
                // per (type, shape). The CPU reference number then attaches
                // to that row in the table but represents the whole
                // (type, shape) case.
                const bool measure_cpu = (k == 0);
                by_wg[k] = run_wg_child(self_path, kWgSizes[k], measure_cpu, qi, (int) si);
            }

            // Table header. Mirrors the SG-sweep layout but with a "wg"
            // column in front instead of "sg" and without the SKIP column
            // because all WG values are always applicable on any GPU
            // (unlike SG, which the device can reject).
            std::fprintf(stdout, "  %-6s | %-6s | %12s | %12s | %9s | %9s\n",
                         "wg", "status", "nmse(g v c)", "nmse(g v s)", "ms/iter", "GB/s");

            double cpu_ms   = 0.0;
            double cpu_gbps = 0.0;
            for (size_t k = 0; k < kWgSizes.size(); ++k) {
                const auto & p = by_wg[k];
                const char * tag = !p.present ? "NOPROC" : !p.supp ? "NOSUPP" : !p.ok ? "FAIL" : "OK";
                const char * wg_lbl = (kWgSizes[k] == 0u) ? "32(prod)" : nullptr;
                char         wg_buf[16];
                if (!wg_lbl) {
                    std::snprintf(wg_buf, sizeof(wg_buf), "%u", kWgSizes[k]);
                    wg_lbl = wg_buf;
                }
                std::fprintf(stdout, "  %-6s | %-6s | %12.3e | %12.3e | %9.3f | %9.2f\n",
                             wg_lbl, tag, p.nmse_gvsc, p.nmse_gvss, p.ms, p.gbps);
                if (p.cpu_ms > 0.0) {
                    cpu_ms   = p.cpu_ms;
                    cpu_gbps = p.cpu_gbps;
                }
            }
            if (cpu_ms > 0.0) {
                std::fprintf(stdout, "  %-6s | %-6s | %12s | %12s | %9.3f | %9.2f\n",
                             "cpu", "REF", "-", "-", cpu_ms, cpu_gbps);
            }

            // Sorted-by-perf summary. Includes the CPU reference row so
            // it's obvious at a glance where CPU falls in the ranking.
            struct Row {
                std::string label;
                double      ms;
                double      gbps;
                double      speedup_vs_cpu;  // cpu_ms / ms, 0 if no CPU ref
            };
            std::vector<Row> rows;
            for (size_t k = 0; k < kWgSizes.size(); ++k) {
                const auto & p = by_wg[k];
                if (!p.present || !p.ok) {
                    continue;
                }
                Row row;
                if (kWgSizes[k] == 0u) {
                    row.label = "wg=32(prod)";
                } else {
                    char b[32];
                    std::snprintf(b, sizeof(b), "wg=%u", kWgSizes[k]);
                    row.label = b;
                }
                row.ms             = p.ms;
                row.gbps           = p.gbps;
                row.speedup_vs_cpu = cpu_ms > 0.0 ? (cpu_ms / p.ms) : 0.0;
                rows.push_back(row);
            }
            if (cpu_ms > 0.0) {
                Row row;
                row.label          = "cpu (ref)";
                row.ms             = cpu_ms;
                row.gbps           = cpu_gbps;
                row.speedup_vs_cpu = 1.0;
                rows.push_back(row);
            }
            std::sort(rows.begin(), rows.end(),
                      [](const Row & a, const Row & b) { return a.ms < b.ms; });
            // Sorted display is for reading convenience only. The per-WG
            // ordering is NOT stable across runs / drivers / GPUs; treat it
            // as informational rather than as a guide to picking a "best"
            // WG. See the header comment above wg_sweep_main for details.
            std::fprintf(stdout, "\n  sorted by ms/iter (informational; see header):\n");
            for (const auto & r : rows) {
                if (r.speedup_vs_cpu > 0.0 && r.label != "cpu (ref)") {
                    std::fprintf(stdout, "    %-14s %9.3f ms %9.2f GB/s  speedup vs CPU = %.2fx\n",
                                 r.label.c_str(), r.ms, r.gbps, r.speedup_vs_cpu);
                } else if (r.label == "cpu (ref)") {
                    std::fprintf(stdout, "    %-14s %9.3f ms %9.2f GB/s  (baseline)\n",
                                 r.label.c_str(), r.ms, r.gbps);
                } else {
                    std::fprintf(stdout, "    %-14s %9.3f ms %9.2f GB/s\n",
                                 r.label.c_str(), r.ms, r.gbps);
                }
            }

            // Assertions.
            // (a) Correctness: every WG must produce near-zero NMSE against the
            //     CPU reference. Same tolerance as the SG sweep.
            for (size_t k = 0; k < kWgSizes.size(); ++k) {
                const auto & p = by_wg[k];
                if (!p.present || !p.ok) {
                    std::fprintf(stderr, "  FAIL: wg=%u did not run\n", kWgSizes[k]);
                    ++n_fail;
                    continue;
                }
                ++n_run;
                if (p.nmse_gvsc > kNmseTol) {
                    std::fprintf(stderr, "  FAIL: wg=%u nmse(gpu vs cpu)=%.3e > %.1e\n",
                                 kWgSizes[k], p.nmse_gvsc, kNmseTol);
                    ++n_fail;
                }
            }
            // (b) Performance: every WG (including WG=2, the most pessimal
            //     cooperative case) must beat the single-threaded CPU
            //     reference. This is the "cooperative GPU beats pre-parallel
            //     CPU" guarantee. Noise margin: require 1.5x speedup so
            //     tiny-shape jitter doesn't trip the assertion. The medium
            //     and huge shapes are the real signal; small shapes are
            //     dispatch-dominated and may legitimately land near 1x.
            if (cpu_ms > 0.0) {
                const double kMinSpeedup = 1.5;
                for (size_t k = 0; k < kWgSizes.size(); ++k) {
                    const auto & p = by_wg[k];
                    if (!p.present || !p.ok) {
                        continue;
                    }
                    const double speedup = cpu_ms / p.ms;
                    // Small-shape cases are dispatch-dominated; soft-warn
                    // instead of hard-fail. Medium is the real check.
                    if (speedup < kMinSpeedup) {
                        const bool is_small = std::strcmp(kShapes[si].label, "small") == 0;
                        if (is_small) {
                            std::fprintf(stderr, "  WARN: wg=%u on %s shape: speedup vs CPU = %.2fx < %.1fx\n",
                                         kWgSizes[k], kShapes[si].label, speedup, kMinSpeedup);
                        } else {
                            std::fprintf(stderr, "  FAIL: wg=%u speedup vs CPU = %.2fx < %.1fx\n",
                                         kWgSizes[k], speedup, kMinSpeedup);
                            ++n_fail;
                        }
                    }
                }
            }
        }
    }

    std::fprintf(stdout, "\n%s: ran=%d failed=%d\n",
                 n_fail ? "FAILED" : "PASSED", n_run, n_fail);
    return n_fail ? 1 : 0;
}

}  // namespace

// Print parent-mode usage. Child mode is an internal ABI and intentionally
// undocumented -- users should never invoke it directly.
static void print_usage(const char * prog) {
    std::fprintf(stderr,
                 "usage: %s [--types t1,t2,...] [--wg-sweep]\n"
                 "\n"
                 "  --types LIST   Comma-separated list of quant type names for the SG sweep.\n"
                 "                 Known: tbq3_0, tbq4_0, pq3_0, pq4_0,\n"
                 "                        tbq3_0_64, tbq4_0_64, pq3_0_64, pq4_0_64.\n"
                 "                 If omitted, every known type is tested (slow).\n"
                 "  --wg-sweep     Run the opt-in workgroup-size sweep instead of the SG sweep.\n"
                 "                 Requires the build to have been configured with\n"
                 "                 -DGGML_VULKAN_TEST_SHADERS=ON. Exercises tbq3_0/pq3_0/*_64\n"
                 "                 at WG in {2,4,8,16,32} and compares against the CPU\n"
                 "                 single-threaded reference. Fails if any cooperative WG\n"
                 "                 fails to beat CPU by 1.5x on the medium shape (soft-warn\n"
                 "                 on small shape where dispatch overhead dominates).\n"
                 "\n"
                 "The test self-spawns one child process per (SG or WG, type, shape)\n"
                 "triple, so restricting the type list linearly reduces runtime.\n",
                 prog);
}

int main(int argc, char ** argv) {
    // Self-spawning: the parent invocation executes parent_main which
    // popen()s this same binary with --child for each (SG, type, shape).
    if (argc >= 2 && std::strcmp(argv[1], "--child") == 0) {
        return child_main(argc - 1, argv + 1);
    }
    const std::string self_path = argv[0];

    // Parse parent-mode CLI. Keep this tiny and hand-rolled: no library
    // dependency needed, and the surface is intentionally small.
    std::vector<size_t> type_filter;
    bool                wg_sweep = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--types") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --types requires a comma-separated argument\n");
                print_usage(argv[0]);
                return 2;
            }
            type_filter = resolve_type_filter(argv[++i]);
            if (type_filter.empty()) {
                std::fprintf(stderr, "error: --types resolved to zero known types\n");
                return 2;
            }
        } else if (a == "--wg-sweep") {
            wg_sweep = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (wg_sweep) {
        if (!type_filter.empty()) {
            std::fprintf(stderr, "warning: --types is ignored in --wg-sweep mode; the sweep "
                                 "always iterates {tbq3_0, pq3_0, tbq3_0_64, pq3_0_64}\n");
        }
        return wg_sweep_main(self_path);
    }
    return parent_main(self_path, type_filter);
}
