// Roofline model: distance calculation latency vs IAA / LZ4 decompression latency.
//
// Benchmarks three operations over varying counts (1,2,4,8,16,32):
//   1. L2 distance calculation to N randomly-accessed nodes (sparse, HNSW-like)
//   2. IAA async decompression+scan_eq (submit all N, wait all N) for varying chunk sizes
//   3. LZ4 software decompression of N blocks (random-access, cold cache) for varying chunk sizes
//
// Outputs:
//   results_dist.csv  — dim, n_nodes, latency_ns_median
//   results_iaa.csv   — dim, chunk_size_bytes, n_jobs, latency_ns_median
//   results_lz4.csv   — dim, chunk_size_bytes, n_jobs, latency_ns_median
//
// Usage: ./profile_roofline [--base-dir <path>] [--out-dir <path>]
//                           [--cpu-core <N>] [--numa-node <N>]

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include <sched.h>           // sched_setaffinity
#include <sys/stat.h>
#include <sys/syscall.h>     // SYS_set_mempolicy
#include <unistd.h>
#include <immintrin.h>       // _mm_clflush, _mm_mfence
#include <lz4.h>
#include <qpl/qpl.h>

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif

// ============================================================
// Timing
// ============================================================

using Clock = std::chrono::steady_clock;

static uint64_t elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

static uint64_t median_ns(std::vector<uint64_t>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ============================================================
// fvecs loading
// ============================================================

struct Fvecs {
    int num = 0;
    int dim = 0;
    std::vector<float> data;  // [num * dim], row-major
};

static Fvecs load_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    f.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    int dim;
    f.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (dim <= 0) throw std::runtime_error("Invalid dim in: " + path);

    const size_t record_bytes = sizeof(int) + static_cast<size_t>(dim) * sizeof(float);
    const int num = static_cast<int>(file_size / record_bytes);

    Fvecs out;
    out.num = num;
    out.dim = dim;
    out.data.resize(static_cast<size_t>(num) * static_cast<size_t>(dim));

    f.seekg(0, std::ios::beg);
    for (int i = 0; i < num; ++i) {
        int d;
        f.read(reinterpret_cast<char*>(&d), sizeof(int));
        f.read(reinterpret_cast<char*>(out.data.data() + i * dim),
               static_cast<std::streamsize>(dim) * sizeof(float));
    }
    return out;
}

// Load a raw binary file as a flat byte array (for FID bitmap data).
static std::vector<uint8_t> load_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open FID file: " + path);
    f.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(sz));
    std::cout << "  loaded " << path << "  size=" << sz << " bytes\n";
    return data;
}

// ============================================================
// L2 squared distance
// ============================================================

static float l2_sq(const float* __restrict__ a, const float* __restrict__ b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Evict all cache lines of a node vector so the next access is a guaranteed
// cold cache miss — models HNSW traversal where each visited neighbor is new.
// Cache line size is 64 bytes on x86; we flush every line covering [ptr, ptr+dim).
static void flush_node(const float* ptr, int dim) {
    const char* p = reinterpret_cast<const char*>(ptr);
    const size_t bytes = static_cast<size_t>(dim) * sizeof(float);
    for (size_t off = 0; off < bytes; off += 64)
        _mm_clflush(p + off);
}

// Evict all cache lines covering a byte range [ptr, ptr+len).
static void flush_buf(const void* ptr, size_t len) {
    const char* p = reinterpret_cast<const char*>(ptr);
    for (size_t off = 0; off < len; off += 64)
        _mm_clflush(p + off);
}

// ============================================================
// QPL RAII job handle
// ============================================================

struct QplJob {
    std::unique_ptr<uint8_t[]> buf;
    qpl_job* job = nullptr;

    QplJob() = default;
    QplJob(const QplJob&) = delete;
    QplJob& operator=(const QplJob&) = delete;

    ~QplJob() {
        if (job) qpl_fini_job(job);
    }

    void init(qpl_path_t path) {
        uint32_t size = 0;
        qpl_status s = qpl_get_job_size(path, &size);
        if (s != QPL_STS_OK)
            throw std::runtime_error("qpl_get_job_size failed: " + std::to_string(s));
        std::cout << "[QPL] qpl_get_job_size = " << size << " bytes\n";
        buf = std::make_unique<uint8_t[]>(size);
        job = reinterpret_cast<qpl_job*>(buf.get());
        s = qpl_init_job(path, job);
        if (s != QPL_STS_OK)
            throw std::runtime_error("qpl_init_job failed: " + std::to_string(s));
    }

    // Reset job state for reuse (must be called before reconfiguring)
    void reinit(qpl_path_t path) {
        qpl_status s = qpl_init_job(path, job);
        if (s != QPL_STS_OK)
            throw std::runtime_error("qpl_init_job (reinit) failed: " + std::to_string(s));
    }
};

// ============================================================
// Compress one chunk with IAA dynamic Huffman (synchronous)
// Returns compressed bytes.
// ============================================================

static std::vector<uint8_t> iaa_compress_chunk(
    const uint8_t* src, size_t src_len, QplJob& cjob)
{
    constexpr uint32_t bound = 2u * 1024u * 1024u;  // 2MB fixed output buffer
    std::vector<uint8_t> out(bound);

    qpl_job* j = cjob.job;
    j->op           = qpl_op_compress;
    j->level        = qpl_default_level;
    j->next_in_ptr  = const_cast<uint8_t*>(src);
    j->next_out_ptr = out.data();
    j->available_in = static_cast<uint32_t>(src_len);
    j->available_out = bound;
    j->flags        = QPL_FLAG_FIRST | QPL_FLAG_LAST
                    | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;

    qpl_status s = qpl_execute_job(j);
    if (s != QPL_STS_OK) {
        std::string msg = "QPL compression failed: status=" + std::to_string(s)
            + " src_len=" + std::to_string(src_len)
            + " total_out=" + std::to_string(j->total_out);
        throw std::runtime_error(msg);
    }

    out.resize(j->total_out);
    cjob.reinit(qpl_path_hardware);  // reset for next use
    return out;
}

// ============================================================
// FID dataset definitions (real filter bitmap files per dimension)
// ============================================================

struct FidDataset {
    int         dim;
    std::string fid_path;
};

static const std::vector<FidDataset> FID_DATASETS = {
    {128,  "/home/jykang5/compass/dataset2/fid_tb/n_filter_100/sift1m/sift1m_synthetic_id_bucket_fid.bin"},
    {512,  "/home/jykang5/compass/dataset2/fid_tb/laion/laion_original_width_fid.bin"},
    {2048, "/home/jykang5/compass/dataset2/fid_tb/hnm/hnm_department_name_fid.bin"},
};

// Extract MAX_N random non-overlapping chunks of chunk_size bytes from fid_data.
// Returns empty vector if fid_data is too small.
static std::vector<std::vector<uint8_t>> extract_chunks(
    const std::vector<uint8_t>& fid_data, size_t chunk_size, int max_n)
{
    if (fid_data.size() < chunk_size) {
        std::cerr << "[WARN] FID data (" << fid_data.size()
                  << " bytes) smaller than chunk_size=" << chunk_size << ", skipping\n";
        return {};
    }
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> pos_dist(0, fid_data.size() - chunk_size);

    std::vector<std::vector<uint8_t>> chunks(max_n, std::vector<uint8_t>(chunk_size));
    for (int i = 0; i < max_n; ++i) {
        const size_t pos = pos_dist(rng);
        std::copy(fid_data.begin() + pos,
                  fid_data.begin() + pos + chunk_size,
                  chunks[i].begin());
    }
    return chunks;
}

// ============================================================
// Benchmark 1: distance calculation
// ============================================================

static void bench_distance(const std::string& base_dir, const std::string& out_dir) {
    struct DataSet { std::string filename; int dim; };
    const std::vector<DataSet> datasets = {
        {"sift1m_base.fvecs",  128},
        {"laion_base.fvecs",   512},
        {"hnm_base.fvecs",    2048},
    };

    const std::vector<int> n_list = {1, 2, 4, 8, 16, 32};
    constexpr int MAX_N   = 32;
    constexpr int WARMUP  = 10;
    constexpr int MEASURE = 50;

    std::ofstream csv(out_dir + "/results_dist.csv");
    csv << "dim,n_nodes,latency_ns_median\n";

    for (const auto& ds : datasets) {
        const std::string path = base_dir + "/" + ds.filename;
        std::cout << "Loading " << path << " ..." << std::flush;
        Fvecs fv = load_fvecs(path);
        std::cout << "  num=" << fv.num << "  dim=" << fv.dim << "\n";

        const int dim      = fv.dim;
        const float* query = fv.data.data();  // first vector = query

        // Pre-select a pool of MAX_N random vectors from the dataset (indices 1..num-1).
        // Each measurement iteration then picks N random indices from this pool,
        // flushes them from cache, and measures distance — mirroring the LZ4 benchmark's
        // pool-based random-access pattern.  The pool is large enough that with N<=32
        // we always have distinct cold-cache targets to choose from.
        std::mt19937 pool_rng(42);
        std::uniform_int_distribution<int> dataset_idx(1, fv.num - 1);
        std::vector<const float*> pool(MAX_N);
        for (int i = 0; i < MAX_N; ++i)
            pool[i] = fv.data.data() + static_cast<size_t>(dataset_idx(pool_rng)) * dim;

        for (int N : n_list) {
            // Pre-generate random pool-index sets (one per iteration)
            const int TOTAL_ITERS = WARMUP + MEASURE;
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> pool_idx(0, MAX_N - 1);

            std::vector<std::vector<int>> iter_indices(TOTAL_ITERS,
                                                       std::vector<int>(N));
            for (int t = 0; t < TOTAL_ITERS; ++t)
                for (int i = 0; i < N; ++i)
                    iter_indices[t][i] = pool_idx(rng);

            volatile float sink = 0.0f;

            // Warmup
            for (int w = 0; w < WARMUP; ++w) {
                const auto& idx = iter_indices[w];
                for (int i = 0; i < N; ++i)
                    sink += l2_sq(query, pool[idx[i]], dim);
            }

            // Measure: flush N random pool vectors then compute distances
            std::vector<uint64_t> times(MEASURE);
            for (int m = 0; m < MEASURE; ++m) {
                const auto& idx = iter_indices[WARMUP + m];

                // Evict node cache lines so every distance computation pays
                // the full DRAM fetch cost (cold random access)
                for (int i = 0; i < N; ++i)
                    flush_node(pool[idx[i]], dim);
                _mm_mfence();

                const auto t0 = Clock::now();
                for (int i = 0; i < N; ++i)
                    sink += l2_sq(query, pool[idx[i]], dim);
                const auto t1 = Clock::now();
                times[m] = elapsed_ns(t0, t1);
            }

            const uint64_t med = median_ns(times);
            std::cout << "  dim=" << dim << "  N=" << N << "  median=" << med << " ns\n";
            csv << dim << "," << N << "," << med << "\n";
        }
    }
}

// ============================================================
// Benchmark 2: IAA decompression + scan_eq  (real FID data)
// ============================================================

static void bench_iaa(const std::string& out_dir) {
    const std::vector<size_t> chunk_sizes = {
          4*1024,   8*1024,  16*1024,  32*1024,  64*1024,
        128*1024, 256*1024, 512*1024,
        1*1024*1024, 2*1024*1024
    };
    const std::vector<int> n_list = {1, 2, 4, 8, 16, 32};
    constexpr int MAX_N   = 32;
    constexpr int WARMUP  = 10;
    constexpr int MEASURE = 50;

    std::ofstream csv(out_dir + "/results_iaa.csv");
    csv << "dim,chunk_size_bytes,n_jobs,latency_ns_median\n";

    // Compressor job (hardware path, synchronous) — reused across all chunk sizes/dims
    QplJob cjob;
    cjob.init(qpl_path_hardware);

    for (const auto& ds : FID_DATASETS) {
        std::cout << "\n[IAA] dim=" << ds.dim << "  loading FID: " << ds.fid_path << "\n";
        const std::vector<uint8_t> fid_data = load_raw(ds.fid_path);

        for (size_t chunk_size : chunk_sizes) {
            const size_t chunk_kb = chunk_size / 1024;
            const size_t chunk_mb = chunk_size / (1024*1024);
            if (chunk_size >= 1024*1024)
                std::cout << "  chunk_size=" << chunk_mb << "MB\n" << std::flush;
            else
                std::cout << "  chunk_size=" << chunk_kb << "KB\n" << std::flush;

            // Extract MAX_N random chunks of chunk_size bytes from the real FID data
            auto raw_chunks = extract_chunks(fid_data, chunk_size, MAX_N);
            if (raw_chunks.empty()) continue;  // file too small for this chunk_size

            // Compress all MAX_N chunks with IAA dynamic Huffman
            std::vector<std::vector<uint8_t>> compressed(MAX_N);
            for (int i = 0; i < MAX_N; ++i)
                compressed[i] = iaa_compress_chunk(raw_chunks[i].data(), chunk_size, cjob);

            // Pre-allocate output bitmap buffers and touch all pages
            // scan_eq with qpl_ow_nom → 1 bit per element → chunk_size/8 bytes
            const size_t out_bytes = (chunk_size + 7) / 8;
            std::vector<std::vector<uint8_t>> out_bufs(MAX_N,
                std::vector<uint8_t>(out_bytes, 0));

            // Initialize MAX_N scan jobs (hardware path)
            std::vector<QplJob> jobs(MAX_N);
            for (int i = 0; i < MAX_N; ++i)
                jobs[i].init(qpl_path_hardware);

            // Configure N jobs for scan_eq (called outside timing)
            auto configure_jobs = [&](int N) {
                for (int i = 0; i < N; ++i) {
                    qpl_job* j         = jobs[i].job;
                    j->op              = qpl_op_scan_eq;
                    j->src1_bit_width  = 8;
                    j->out_bit_width   = qpl_ow_nom;   // 1-bit bitmap output
                    j->param_low       = 1;            // scan predicate: value == 1
                    j->num_input_elements = static_cast<uint32_t>(chunk_size);
                    j->next_in_ptr     = compressed[i].data();
                    j->available_in    = static_cast<uint32_t>(chunk_size);  // original uncompressed size
                    j->next_out_ptr    = out_bufs[i].data();
                    j->available_out   = static_cast<uint32_t>(out_bytes);
                    j->flags           = QPL_FLAG_FIRST | QPL_FLAG_LAST
                                       | QPL_FLAG_DECOMPRESS_ENABLE;
                }
            };

            for (int N : n_list) {
                auto check_scan_jobs = [&](int n, const char* phase) {
                    for (int i = 0; i < n; ++i) {
                        qpl_status s = qpl_wait_job(jobs[i].job);
                        if (s != QPL_STS_OK) {
                            std::string msg = std::string("qpl scan_eq failed (") + phase + ")"
                                + " status=" + std::to_string(s)
                                + " job=" + std::to_string(i)
                                + " chunk_size=" + std::to_string(chunk_size)
                                + " N=" + std::to_string(n)
                                + " compressed_in=" + std::to_string(compressed[i].size())
                                + " num_input_elements=" + std::to_string(chunk_size)
                                + " available_out=" + std::to_string(out_bytes);
                            throw std::runtime_error(msg);
                        }
                    }
                };

                auto submit_jobs = [&](int n, const char* phase) {
                    for (int i = 0; i < n; ++i) {
                        qpl_status s = qpl_submit_job(jobs[i].job);
                        if (s != QPL_STS_OK) {
                            std::string msg = std::string("qpl_submit_job failed (") + phase + ")"
                                + " status=" + std::to_string(s)
                                + " job=" + std::to_string(i)
                                + " chunk_size=" + std::to_string(chunk_size);
                            throw std::runtime_error(msg);
                        }
                    }
                };

                // Warmup
                for (int w = 0; w < WARMUP; ++w) {
                    for (int i = 0; i < N; ++i) jobs[i].reinit(qpl_path_hardware);
                    configure_jobs(N);
                    submit_jobs(N, "warmup");
                    check_scan_jobs(N, "warmup");
                }

                // Measure: submit+wait is timed
                std::vector<uint64_t> times(MEASURE);
                for (int m = 0; m < MEASURE; ++m) {
                    for (int i = 0; i < N; ++i) jobs[i].reinit(qpl_path_hardware);
                    configure_jobs(N);

                    // --- timed region ---
                    const auto t0 = Clock::now();
                    for (int i = 0; i < N; ++i) qpl_submit_job(jobs[i].job);
                    for (int i = 0; i < N; ++i) qpl_wait_job(jobs[i].job);
                    const auto t1 = Clock::now();
                    // --------------------

                    times[m] = elapsed_ns(t0, t1);
                }

                const uint64_t med = median_ns(times);
                const double throughput_gbps =
                    static_cast<double>(N) * static_cast<double>(chunk_size)
                    / static_cast<double>(med);  // bytes / ns = GB/s
                std::cout << "    N=" << N << "  median=" << med << " ns"
                          << "  throughput=" << throughput_gbps << " GB/s\n";
                csv << ds.dim << "," << chunk_size << "," << N << "," << med << "\n";
            }
        }
    }
}

// ============================================================
// Benchmark 3: LZ4 software decompression (N random-access blocks, real FID data)
// Models sparse HNSW traversal: each iteration picks N random blocks
// from the pool of MAX_N, flushes their compressed data from cache,
// then decompresses them — guaranteeing cold DRAM fetch cost.
// ============================================================

static void bench_lz4(const std::string& out_dir) {
    const std::vector<size_t> chunk_sizes = {
          4*1024,   8*1024,  16*1024,  32*1024,  64*1024,
        128*1024, 256*1024, 512*1024,
        1*1024*1024, 2*1024*1024
    };
    const std::vector<int> n_list = {1, 2, 4, 8, 16, 32};
    constexpr int MAX_N   = 32;
    constexpr int WARMUP  = 10;
    constexpr int MEASURE = 50;

    std::ofstream csv(out_dir + "/results_lz4.csv");
    csv << "dim,chunk_size_bytes,n_jobs,latency_ns_median\n";

    for (const auto& ds : FID_DATASETS) {
        std::cout << "\n[LZ4] dim=" << ds.dim << "  loading FID: " << ds.fid_path << "\n";
        const std::vector<uint8_t> fid_data = load_raw(ds.fid_path);

        for (size_t chunk_size : chunk_sizes) {
            const size_t chunk_kb = chunk_size / 1024;
            const size_t chunk_mb = chunk_size / (1024*1024);
            if (chunk_size >= 1024*1024)
                std::cout << "  chunk_size=" << chunk_mb << "MB\n" << std::flush;
            else
                std::cout << "  chunk_size=" << chunk_kb << "KB\n" << std::flush;

            // Extract MAX_N random chunks of chunk_size bytes from the real FID data
            auto raw_chunks = extract_chunks(fid_data, chunk_size, MAX_N);
            if (raw_chunks.empty()) continue;  // file too small for this chunk_size

            // Compress all MAX_N chunks with LZ4
            std::vector<std::vector<char>> compressed(MAX_N);
            for (int i = 0; i < MAX_N; ++i) {
                const int bound = LZ4_compressBound(static_cast<int>(chunk_size));
                compressed[i].resize(static_cast<size_t>(bound));
                const int comp_sz = LZ4_compress_default(
                    reinterpret_cast<const char*>(raw_chunks[i].data()),
                    compressed[i].data(),
                    static_cast<int>(chunk_size),
                    bound);
                if (comp_sz <= 0)
                    throw std::runtime_error("LZ4 compression failed at chunk_size="
                                             + std::to_string(chunk_size));
                compressed[i].resize(static_cast<size_t>(comp_sz));
                std::cout << "  [LZ4] compressed[" << i << "] size=" << comp_sz << " bytes\n";
            }

            // Pre-allocate output buffers and touch all pages
            std::vector<std::vector<char>> out_bufs(MAX_N,
                std::vector<char>(chunk_size, 0));

            for (int N : n_list) {
                // Pre-generate random pool-index sets (one per iteration)
                const int TOTAL_ITERS = WARMUP + MEASURE;
                std::mt19937 rng(42);
                std::uniform_int_distribution<int> idx_dist(0, MAX_N - 1);

                std::vector<std::vector<int>> iter_indices(TOTAL_ITERS,
                                                           std::vector<int>(N));
                for (int t = 0; t < TOTAL_ITERS; ++t)
                    for (int i = 0; i < N; ++i)
                        iter_indices[t][i] = idx_dist(rng);

                // Warmup
                for (int w = 0; w < WARMUP; ++w) {
                    const auto& idx = iter_indices[w];
                    for (int i = 0; i < N; ++i)
                        LZ4_decompress_safe(compressed[idx[i]].data(),
                                            out_bufs[idx[i]].data(),
                                            static_cast<int>(compressed[idx[i]].size()),
                                            static_cast<int>(chunk_size));
                }

                // Measure: flush N random compressed blocks then decompress them
                std::vector<uint64_t> times(MEASURE);
                for (int m = 0; m < MEASURE; ++m) {
                    const auto& idx = iter_indices[WARMUP + m];

                    // Evict compressed blocks from cache so each decompression
                    // pays the full DRAM fetch cost (cold random access)
                    for (int i = 0; i < N; ++i)
                        flush_buf(compressed[idx[i]].data(), compressed[idx[i]].size());
                    _mm_mfence();

                    const auto t0 = Clock::now();
                    for (int i = 0; i < N; ++i)
                        LZ4_decompress_safe(compressed[idx[i]].data(),
                                            out_bufs[idx[i]].data(),
                                            static_cast<int>(compressed[idx[i]].size()),
                                            static_cast<int>(chunk_size));
                    const auto t1 = Clock::now();
                    times[m] = elapsed_ns(t0, t1);
                }

                const uint64_t med = median_ns(times);
                std::cout << "    N=" << N << "  median=" << med << " ns\n";
                csv << ds.dim << "," << chunk_size << "," << N << "," << med << "\n";
            }
        }
    }
}

// ============================================================
// CPU / NUMA binding helpers
// ============================================================

static void pin_cpu_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        std::cerr << "[WARN] sched_setaffinity(core=" << core
                  << ") failed: " << strerror(errno) << "\n";
    else
        std::cout << "[INFO] Pinned to CPU core " << core << "\n";
}

static void bind_numa_node(int node) {
    unsigned long nodemask = (1UL << node);
    // maxnode = highest bit index + 2 (exclusive upper bound per man page)
    long ret = syscall(SYS_set_mempolicy, MPOL_BIND,
                       &nodemask, static_cast<unsigned long>(sizeof(nodemask) * 8));
    if (ret != 0)
        std::cerr << "[WARN] set_mempolicy(node=" << node
                  << ") failed: " << strerror(errno) << "\n";
    else
        std::cout << "[INFO] Bound memory to NUMA node " << node << "\n";
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    std::string base_dir = "/home/jykang5/compass/dataset2/compass_base_query";
    std::string out_dir  = ".";
    int cpu_core  = 8;   // default: core 8
    int numa_node = 0;   // default: NUMA node 0

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--base-dir"  && i + 1 < argc) base_dir  = argv[++i];
        else if (arg == "--out-dir"   && i + 1 < argc) out_dir   = argv[++i];
        else if (arg == "--cpu-core"  && i + 1 < argc) cpu_core  = std::stoi(argv[++i]);
        else if (arg == "--numa-node" && i + 1 < argc) numa_node = std::stoi(argv[++i]);
    }

    // Pin to specific CPU core and NUMA memory node before any allocation
    pin_cpu_core(cpu_core);
    bind_numa_node(numa_node);

    mkdir(out_dir.c_str(), 0755);  // ignore error if already exists

    std::cout << "=== Distance Calculation Benchmark ===\n";
    bench_distance(base_dir, out_dir);

    std::cout << "\n=== IAA Decompression+Scan Benchmark ===\n";
    bench_iaa(out_dir);

    std::cout << "\n=== LZ4 Decompression Benchmark ===\n";
    bench_lz4(out_dir);

    std::cout << "\nDone. Results written to " << out_dir << "/\n";
    return 0;
}
