#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <lz4.h>
#include <qpl/qpl.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif

namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;

static uint64_t elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

static uint64_t median_ns(std::vector<uint64_t>& v) {
    if (v.empty()) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) {
                out.push_back(std::stoi(cur));
                cur.clear();
            }
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(c))) {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(std::stoi(cur));
    }
    return out;
}

static std::vector<size_t> parse_size_list(const std::string& s) {
    std::vector<size_t> out;
    for (int v : parse_int_list(s)) {
        if (v <= 0) {
            throw std::runtime_error("chunk size must be > 0");
        }
        out.push_back(static_cast<size_t>(v));
    }
    return out;
}

static void flush_buf(const void* ptr, size_t len) {
    const char* p = reinterpret_cast<const char*>(ptr);
    for (size_t off = 0; off < len; off += 64) {
        _mm_clflush(p + off);
    }
}

static void pin_cpu_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        std::cerr << "[WARN] sched_setaffinity failed for core=" << core << "\n";
    }
}

static void bind_numa_node(int node) {
    if (node < 0 || node > 63) {
        return;
    }
    unsigned long nodemask = (1UL << static_cast<unsigned long>(node));
    (void)syscall(
        SYS_set_mempolicy,
        MPOL_BIND,
        &nodemask,
        static_cast<unsigned long>(sizeof(nodemask) * 8));
}

struct QplJob {
    std::unique_ptr<uint8_t[]> storage;
    qpl_job* job = nullptr;

    QplJob() = default;
    QplJob(const QplJob&) = delete;
    QplJob& operator=(const QplJob&) = delete;
    QplJob(QplJob&& other) noexcept
        : storage(std::move(other.storage)), job(other.job) {
        other.job = nullptr;
    }

    QplJob& operator=(QplJob&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (job != nullptr) {
            (void)qpl_fini_job(job);
        }
        storage = std::move(other.storage);
        job = other.job;
        other.job = nullptr;
        return *this;
    }

    ~QplJob() {
        if (job != nullptr) {
            (void)qpl_fini_job(job);
        }
    }

    void init(qpl_path_t path) {
        uint32_t size = 0;
        qpl_status s = qpl_get_job_size(path, &size);
        if (s != QPL_STS_OK) {
            throw std::runtime_error("qpl_get_job_size failed");
        }
        storage = std::make_unique<uint8_t[]>(size);
        job = reinterpret_cast<qpl_job*>(storage.get());
        s = qpl_init_job(path, job);
        if (s != QPL_STS_OK) {
            throw std::runtime_error("qpl_init_job failed");
        }
    }

    void reinit(qpl_path_t path) {
        qpl_status s = qpl_init_job(path, job);
        if (s != QPL_STS_OK) {
            throw std::runtime_error("qpl_init_job reinit failed");
        }
    }
};

static std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open: " + path);
    }
    in.seekg(0, std::ios::end);
    const size_t n = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(n);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n));
    return out;
}

static std::vector<uint8_t> tile_to_size(const std::vector<uint8_t>& in, size_t want) {
    if (in.empty()) {
        throw std::runtime_error("cannot tile empty input");
    }
    if (in.size() >= want) {
        return in;
    }
    std::vector<uint8_t> out;
    out.resize(want);
    size_t copied = 0;
    while (copied < want) {
        const size_t take = std::min(in.size(), want - copied);
        std::memcpy(out.data() + copied, in.data(), take);
        copied += take;
    }
    return out;
}

static std::vector<std::vector<uint8_t>> random_chunks(
    const std::vector<uint8_t>& source,
    size_t chunk_size,
    int pool,
    uint32_t seed) {
    if (source.size() < chunk_size) {
        throw std::runtime_error("source smaller than chunk size");
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, source.size() - chunk_size);
    std::vector<std::vector<uint8_t>> chunks(pool, std::vector<uint8_t>(chunk_size));
    for (int i = 0; i < pool; ++i) {
        const size_t pos = dist(rng);
        std::memcpy(chunks[i].data(), source.data() + pos, chunk_size);
    }
    return chunks;
}

static std::vector<uint8_t> iaa_compress(const std::vector<uint8_t>& in, QplJob& cjob) {
    constexpr uint32_t kOutCap = 2u * 1024u * 1024u;
    std::vector<uint8_t> out(kOutCap);
    qpl_job* j = cjob.job;
    j->op = qpl_op_compress;
    j->level = qpl_default_level;
    j->next_in_ptr = const_cast<uint8_t*>(in.data());
    j->available_in = static_cast<uint32_t>(in.size());
    j->next_out_ptr = out.data();
    j->available_out = kOutCap;
    j->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;
    qpl_status s = qpl_execute_job(j);
    if (s != QPL_STS_OK) {
        throw std::runtime_error("qpl compress failed");
    }
    out.resize(j->total_out);
    cjob.reinit(qpl_path_hardware);
    return out;
}

struct Dataset {
    int dim = 0;
    std::string fid_path;
};

struct Args {
    std::string out_csv = "./out/decomp_latency.csv";
    int engine_count = 8;
    bool append = false;
    int warmup = 10;
    int iters = 50;
    int pool_size = 128;
    int cpu_core = 8;
    int numa_node = 0;
    std::vector<size_t> chunk_sizes = {
        4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
    };
    std::vector<int> n_list = {1, 2, 4, 8, 16, 32, 64};
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string cur = argv[i];
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            ++i;
            return argv[i];
        };

        if (cur == "--out-csv") {
            args.out_csv = need(cur);
        } else if (cur == "--engine-count") {
            args.engine_count = std::stoi(need(cur));
        } else if (cur == "--append") {
            args.append = true;
        } else if (cur == "--warmup") {
            args.warmup = std::stoi(need(cur));
        } else if (cur == "--iters") {
            args.iters = std::stoi(need(cur));
        } else if (cur == "--pool-size") {
            args.pool_size = std::stoi(need(cur));
        } else if (cur == "--cpu-core") {
            args.cpu_core = std::stoi(need(cur));
        } else if (cur == "--numa-node") {
            args.numa_node = std::stoi(need(cur));
        } else if (cur == "--chunk-sizes") {
            args.chunk_sizes = parse_size_list(need(cur));
        } else if (cur == "--n-list") {
            args.n_list = parse_int_list(need(cur));
        } else if (cur == "-h" || cur == "--help") {
            std::cout
                << "Usage: " << argv[0] << " [--out-csv PATH] [--engine-count N] [--append]"
                << " [--warmup N] [--iters N] [--pool-size N] [--cpu-core N] [--numa-node N]"
                << " [--chunk-sizes csv] [--n-list csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + cur);
        }
    }
    return args;
}

struct Lz4Pack {
    std::vector<std::vector<char>> compressed;
    std::vector<std::vector<char>> output;
};

struct IaaPack {
    std::vector<std::vector<uint8_t>> compressed;
    std::vector<std::vector<uint8_t>> output;
    std::vector<QplJob> jobs;
};

static Lz4Pack prepare_lz4(
    const std::vector<std::vector<uint8_t>>& chunks,
    size_t chunk_size) {
    Lz4Pack out;
    out.compressed.resize(chunks.size());
    out.output.assign(chunks.size(), std::vector<char>(chunk_size, 0));

    for (size_t i = 0; i < chunks.size(); ++i) {
        const int bound = LZ4_compressBound(static_cast<int>(chunk_size));
        out.compressed[i].resize(static_cast<size_t>(bound));
        const int n = LZ4_compress_default(
            reinterpret_cast<const char*>(chunks[i].data()),
            out.compressed[i].data(),
            static_cast<int>(chunk_size),
            bound);
        if (n <= 0) {
            throw std::runtime_error("LZ4_compress_default failed");
        }
        out.compressed[i].resize(static_cast<size_t>(n));
    }
    return out;
}

static IaaPack prepare_iaa(
    const std::vector<std::vector<uint8_t>>& chunks,
    size_t chunk_size,
    QplJob& compressor,
    int max_jobs) {
    IaaPack out;
    out.compressed.resize(chunks.size());
    const size_t out_bytes = (chunk_size + 7) / 8;
    out.output.assign(chunks.size(), std::vector<uint8_t>(out_bytes, 0));
    out.jobs.resize(static_cast<size_t>(max_jobs));

    for (size_t i = 0; i < chunks.size(); ++i) {
        out.compressed[i] = iaa_compress(chunks[i], compressor);
    }
    for (int i = 0; i < max_jobs; ++i) {
        out.jobs[static_cast<size_t>(i)].init(qpl_path_hardware);
    }
    return out;
}

static void configure_scan_job(
    qpl_job* job,
    uint8_t* in_ptr,
    uint32_t in_size_uncompressed,
    uint32_t compressed_size,
    uint8_t* out_ptr,
    uint32_t out_size) {
    job->op = qpl_op_scan_eq;
    job->src1_bit_width = 8;
    job->out_bit_width = qpl_ow_nom;
    job->param_low = 1;
    job->num_input_elements = in_size_uncompressed;
    job->next_in_ptr = in_ptr;
    job->available_in = compressed_size;
    job->next_out_ptr = out_ptr;
    job->available_out = out_size;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        pin_cpu_core(args.cpu_core);
        bind_numa_node(args.numa_node);

        const std::vector<Dataset> datasets = {
            {128, "/storage/jykang5/fid_tb/n_filter_100/sift1m/sift1m_synthetic_id_bucket_fid.bin"},
            {512, "/storage/jykang5/fid_tb/laion/laion_original_width_fid.bin"},
            {2048, "/storage/jykang5/fid_tb/hnm/hnm_department_name_fid.bin"},
        };

        const size_t max_chunk = *std::max_element(args.chunk_sizes.begin(), args.chunk_sizes.end());

        fs::path out_path(args.out_csv);
        if (!out_path.parent_path().empty()) {
            fs::create_directories(out_path.parent_path());
        }

        const bool write_header = !args.append || !fs::exists(out_path);
        std::ofstream csv(args.out_csv, args.append ? std::ios::app : std::ios::trunc);
        if (!csv.is_open()) {
            throw std::runtime_error("failed to open out csv: " + args.out_csv);
        }
        if (write_header) {
            csv << "dim,engine_count,chunk_size_bytes,n_nodes,iaa_submit_ns_median,iaa_wait_ns_median,iaa_check_ns_median,lz4_decompress_ns_median\n";
        }

        QplJob compressor;
        compressor.init(qpl_path_hardware);

        for (const auto& ds : datasets) {
            std::cout << "[BENCH] dim=" << ds.dim << " file=" << ds.fid_path << "\n";
            std::vector<uint8_t> raw = load_file(ds.fid_path);
            raw = tile_to_size(raw, std::max(max_chunk * 2, raw.size()));

            for (size_t chunk_size : args.chunk_sizes) {
                std::cout << "  chunk_size=" << chunk_size << " bytes\n";
                auto chunks = random_chunks(raw, chunk_size, args.pool_size, static_cast<uint32_t>(42 + ds.dim + chunk_size));
                auto lz4 = prepare_lz4(chunks, chunk_size);
                auto iaa = prepare_iaa(chunks, chunk_size, compressor, *std::max_element(args.n_list.begin(), args.n_list.end()));

                for (int n_jobs : args.n_list) {
                    if (n_jobs <= 0 || n_jobs > static_cast<int>(chunks.size())) {
                        throw std::runtime_error("invalid n_jobs for pool size");
                    }

                    const int total_iters = args.warmup + args.iters;
                    std::mt19937 rng(static_cast<uint32_t>(123 + ds.dim + n_jobs + chunk_size));
                    std::uniform_int_distribution<int> pick(0, args.pool_size - 1);
                    std::vector<std::vector<int>> picks(static_cast<size_t>(total_iters), std::vector<int>(static_cast<size_t>(n_jobs), 0));
                    for (int t = 0; t < total_iters; ++t) {
                        for (int i = 0; i < n_jobs; ++i) {
                            picks[static_cast<size_t>(t)][static_cast<size_t>(i)] = pick(rng);
                        }
                    }

                    auto run_iaa_wait_once = [&](const std::vector<int>& idx, uint64_t& submit_ns, uint64_t& wait_ns) {
                        const size_t out_bytes = (chunk_size + 7) / 8;
                        for (int i = 0; i < n_jobs; ++i) {
                            const int id = idx[static_cast<size_t>(i)];
                            QplJob& jh = iaa.jobs[static_cast<size_t>(i)];
                            jh.reinit(qpl_path_hardware);
                            std::fill(iaa.output[static_cast<size_t>(id)].begin(), iaa.output[static_cast<size_t>(id)].end(), 0);
                            configure_scan_job(
                                jh.job,
                                iaa.compressed[static_cast<size_t>(id)].data(),
                                static_cast<uint32_t>(chunk_size),
                                static_cast<uint32_t>(iaa.compressed[static_cast<size_t>(id)].size()),
                                iaa.output[static_cast<size_t>(id)].data(),
                                static_cast<uint32_t>(out_bytes));
                            flush_buf(iaa.compressed[static_cast<size_t>(id)].data(), iaa.compressed[static_cast<size_t>(id)].size());
                        }
                        _mm_mfence();

                        const auto t0 = Clock::now();
                        for (int i = 0; i < n_jobs; ++i) {
                            qpl_status s = qpl_submit_job(iaa.jobs[static_cast<size_t>(i)].job);
                            if (s != QPL_STS_OK) {
                                throw std::runtime_error("qpl_submit_job failed in wait run");
                            }
                        }
                        const auto t1 = Clock::now();
                        for (int i = 0; i < n_jobs; ++i) {
                            qpl_status s = qpl_wait_job(iaa.jobs[static_cast<size_t>(i)].job);
                            if (s != QPL_STS_OK) {
                                throw std::runtime_error("qpl_wait_job failed in wait run");
                            }
                        }
                        const auto t2 = Clock::now();
                        submit_ns = elapsed_ns(t0, t1);
                        wait_ns = elapsed_ns(t1, t2);
                    };

                    auto run_iaa_check_once = [&](const std::vector<int>& idx, uint64_t& check_ns) {
                        const size_t out_bytes = (chunk_size + 7) / 8;
                        for (int i = 0; i < n_jobs; ++i) {
                            const int id = idx[static_cast<size_t>(i)];
                            QplJob& jh = iaa.jobs[static_cast<size_t>(i)];
                            jh.reinit(qpl_path_hardware);
                            std::fill(iaa.output[static_cast<size_t>(id)].begin(), iaa.output[static_cast<size_t>(id)].end(), 0);
                            configure_scan_job(
                                jh.job,
                                iaa.compressed[static_cast<size_t>(id)].data(),
                                static_cast<uint32_t>(chunk_size),
                                static_cast<uint32_t>(iaa.compressed[static_cast<size_t>(id)].size()),
                                iaa.output[static_cast<size_t>(id)].data(),
                                static_cast<uint32_t>(out_bytes));
                            flush_buf(iaa.compressed[static_cast<size_t>(id)].data(), iaa.compressed[static_cast<size_t>(id)].size());
                        }
                        _mm_mfence();

                        for (int i = 0; i < n_jobs; ++i) {
                            qpl_status s = qpl_submit_job(iaa.jobs[static_cast<size_t>(i)].job);
                            if (s != QPL_STS_OK) {
                                throw std::runtime_error("qpl_submit_job failed in check run");
                            }
                        }

                        std::vector<uint8_t> done(static_cast<size_t>(n_jobs), 0);
                        int remaining = n_jobs;
                        const auto t0 = Clock::now();
                        while (remaining > 0) {
                            for (int i = 0; i < n_jobs; ++i) {
                                if (done[static_cast<size_t>(i)] != 0) {
                                    continue;
                                }
                                qpl_status s = qpl_check_job(iaa.jobs[static_cast<size_t>(i)].job);
                                if (s == QPL_STS_OK) {
                                    done[static_cast<size_t>(i)] = 1;
                                    --remaining;
                                } else if (s == QPL_STS_BEING_PROCESSED) {
                                    continue;
                                } else {
                                    throw std::runtime_error("qpl_check_job failed in check run");
                                }
                            }
                        }
                        const auto t1 = Clock::now();
                        check_ns = elapsed_ns(t0, t1);
                    };

                    auto run_lz4_once = [&](const std::vector<int>& idx, uint64_t& decomp_ns) {
                        for (int i = 0; i < n_jobs; ++i) {
                            const int id = idx[static_cast<size_t>(i)];
                            flush_buf(
                                lz4.compressed[static_cast<size_t>(id)].data(),
                                lz4.compressed[static_cast<size_t>(id)].size());
                            flush_buf(
                                lz4.output[static_cast<size_t>(id)].data(),
                                lz4.output[static_cast<size_t>(id)].size());
                        }
                        _mm_mfence();

                        const auto t0 = Clock::now();
                        for (int i = 0; i < n_jobs; ++i) {
                            const int id = idx[static_cast<size_t>(i)];
                            const int n = LZ4_decompress_safe(
                                lz4.compressed[static_cast<size_t>(id)].data(),
                                lz4.output[static_cast<size_t>(id)].data(),
                                static_cast<int>(lz4.compressed[static_cast<size_t>(id)].size()),
                                static_cast<int>(chunk_size));
                            if (n < 0) {
                                throw std::runtime_error("LZ4_decompress_safe failed");
                            }
                        }
                        const auto t1 = Clock::now();
                        decomp_ns = elapsed_ns(t0, t1);
                    };

                    for (int w = 0; w < args.warmup; ++w) {
                        uint64_t submit_ns = 0;
                        uint64_t wait_ns = 0;
                        uint64_t check_ns = 0;
                        uint64_t lz4_ns = 0;
                        run_iaa_wait_once(picks[static_cast<size_t>(w)], submit_ns, wait_ns);
                        run_iaa_check_once(picks[static_cast<size_t>(w)], check_ns);
                        run_lz4_once(picks[static_cast<size_t>(w)], lz4_ns);
                    }

                    std::vector<uint64_t> submit_vals;
                    std::vector<uint64_t> wait_vals;
                    std::vector<uint64_t> check_vals;
                    std::vector<uint64_t> lz4_vals;
                    submit_vals.reserve(static_cast<size_t>(args.iters));
                    wait_vals.reserve(static_cast<size_t>(args.iters));
                    check_vals.reserve(static_cast<size_t>(args.iters));
                    lz4_vals.reserve(static_cast<size_t>(args.iters));

                    for (int it = 0; it < args.iters; ++it) {
                        const auto& idx = picks[static_cast<size_t>(args.warmup + it)];
                        uint64_t submit_ns = 0;
                        uint64_t wait_ns = 0;
                        uint64_t check_ns = 0;
                        uint64_t lz4_ns = 0;
                        run_iaa_wait_once(idx, submit_ns, wait_ns);
                        run_iaa_check_once(idx, check_ns);
                        run_lz4_once(idx, lz4_ns);
                        submit_vals.push_back(submit_ns);
                        wait_vals.push_back(wait_ns);
                        check_vals.push_back(check_ns);
                        lz4_vals.push_back(lz4_ns);
                    }

                    const uint64_t submit_med = median_ns(submit_vals);
                    const uint64_t wait_med = median_ns(wait_vals);
                    const uint64_t check_med = median_ns(check_vals);
                    const uint64_t lz4_med = median_ns(lz4_vals);

                    csv
                        << ds.dim << ','
                        << args.engine_count << ','
                        << chunk_size << ','
                        << n_jobs << ','
                        << submit_med << ','
                        << wait_med << ','
                        << check_med << ','
                        << lz4_med << '\n';

                    std::cout
                        << "    n=" << n_jobs
                        << " submit=" << submit_med
                        << " wait=" << wait_med
                        << " check=" << check_med
                        << " lz4=" << lz4_med
                        << "\n";
                }
            }
        }

        csv.flush();
        std::cout << "[DONE] wrote " << args.out_csv << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
