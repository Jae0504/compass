#include <qpl/qpl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMaxInFlightJobs = 128;
constexpr uint64_t kOneKB = 1024ULL;
constexpr uint64_t kOneMB = 1024ULL * 1024ULL;

struct Options {
    qpl_path_t path = qpl_path_hardware;
    std::string path_name = "hardware";
    std::vector<uint64_t> sizes_bytes = {
        1ULL * kOneMB,
        2ULL * kOneMB,
        4ULL * kOneMB,
        8ULL * kOneMB,
        16ULL * kOneMB,
        32ULL * kOneMB,
    };
    size_t block_size_kb = 32;
    int iterations = 40;
    int warmup = 5;
    uint64_t seed = 1ULL;
    std::string output_csv = "results/alignment_microbench.csv";
};

struct SummaryStats {
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
};

struct ResultRow {
    std::string layout;
    double size_mb = 0.0;
    uint64_t size_bytes = 0;
    uint64_t raw_bytes = 0;
    uint64_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    int jobs_per_query = 0;
    int iterations = 0;
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double avg_wait_us = 0.0;
    double p50_wait_us = 0.0;
    double p95_wait_us = 0.0;
    double avg_submit_us = 0.0;
    double p50_submit_us = 0.0;
    double p95_submit_us = 0.0;
    double throughput_mb_s = 0.0;
    double slowdown_vs_aligned = 0.0;
};

struct CompressedBlocks {
    size_t block_size = 0;
    std::vector<uint32_t> raw_block_sizes;
    std::vector<std::vector<uint8_t>> compressed_blocks;

    uint64_t compressed_bytes_total() const {
        uint64_t total = 0;
        for (const auto& b : compressed_blocks) {
            total += static_cast<uint64_t>(b.size());
        }
        return total;
    }

    size_t block_count() const {
        return compressed_blocks.size();
    }
};

struct ScanTask {
    const std::vector<uint8_t>* compressed = nullptr;
    uint32_t drop_initial_bytes = 0;
    uint32_t num_input_elements = 0;
    uint32_t needle_u32 = 0;
};

struct AsyncRunMetrics {
    uint64_t matches = 0;
    // Total time for the one-pass submit loop over all tasks in a query.
    uint64_t submit_time_ns = 0;
    // Total time for the one-pass wait/drain loop over all submitted jobs.
    uint64_t wait_time_ns = 0;
};

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

void require_qpl_ok(qpl_status st, const std::string& where) {
    if (st != QPL_STS_OK) {
        fail(where + " failed with qpl_status=" + std::to_string(static_cast<int>(st)));
    }
}

qpl_path_t parse_path(const std::string& text) {
    if (text == "hardware") {
        return qpl_path_hardware;
    }
    if (text == "software") {
        return qpl_path_software;
    }
    if (text == "auto") {
        return qpl_path_auto;
    }
    fail("Invalid --path. Use hardware|software|auto");
    return qpl_path_auto;
}

std::vector<uint64_t> parse_size_list_to_bytes(
    const std::string& text,
    uint64_t unit_bytes,
    const std::string& flag_name) {
    std::vector<uint64_t> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        uint64_t value = 0;
        try {
            value = static_cast<uint64_t>(std::stoull(token));
        } catch (...) {
            fail("Invalid size value in " + flag_name + ": " + token);
        }
        if (value == 0) {
            fail(flag_name + " values must be > 0");
        }
        if (value > (std::numeric_limits<uint64_t>::max() / unit_bytes)) {
            fail(flag_name + " value overflows byte conversion: " + token);
        }
        out.push_back(value * unit_bytes);
    }
    if (out.empty()) {
        fail(flag_name + " produced an empty list");
    }
    return out;
}

double bytes_to_mb(uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(kOneMB);
}

std::string format_size_bytes_human(uint64_t bytes) {
    std::ostringstream oss;
    if (bytes % kOneMB == 0) {
        oss << (bytes / kOneMB) << "MB";
        return oss.str();
    }
    if (bytes % kOneKB == 0) {
        oss << (bytes / kOneKB) << "KB";
        return oss.str();
    }
    oss << bytes << "B";
    return oss.str();
}

Options parse_args(int argc, char** argv) {
    Options opt;
    bool sizes_mb_set = false;
    bool sizes_kb_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--path") {
            opt.path_name = need_value("--path");
            opt.path = parse_path(opt.path_name);
        } else if (arg == "--sizes-mb") {
            sizes_mb_set = true;
            opt.sizes_bytes =
                parse_size_list_to_bytes(need_value("--sizes-mb"), kOneMB, "--sizes-mb");
        } else if (arg == "--sizes-kb") {
            sizes_kb_set = true;
            opt.sizes_bytes =
                parse_size_list_to_bytes(need_value("--sizes-kb"), kOneKB, "--sizes-kb");
        } else if (arg == "--block-size-kb") {
            opt.block_size_kb = static_cast<size_t>(std::stoull(need_value("--block-size-kb")));
        } else if (arg == "--iterations") {
            opt.iterations = std::stoi(need_value("--iterations"));
        } else if (arg == "--warmup") {
            opt.warmup = std::stoi(need_value("--warmup"));
        } else if (arg == "--seed") {
            opt.seed = static_cast<uint64_t>(std::stoull(need_value("--seed")));
        } else if (arg == "--output-csv") {
            opt.output_csv = need_value("--output-csv");
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --path hardware|software|auto   (default: hardware)\n"
                << "  --sizes-mb 1,2,4,8,16,32        (default: 1,2,4,8,16,32)\n"
                << "  --sizes-kb 128,256,512,1024     (optional, mutually exclusive with --sizes-mb)\n"
                << "  --block-size-kb N               (default: 32)\n"
                << "  --iterations N                  (default: 40)\n"
                << "  --warmup N                      (default: 5)\n"
                << "  --seed N                        (default: 123456789)\n"
                << "  --output-csv PATH               (default: results/alignment_microbench.csv)\n";
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (sizes_mb_set && sizes_kb_set) {
        fail("Provide only one of --sizes-mb or --sizes-kb");
    }

    if (opt.iterations <= 0) {
        fail("--iterations must be > 0");
    }
    if (opt.block_size_kb == 0) {
        fail("--block-size-kb must be > 0");
    }
    if (opt.warmup < 0) {
        fail("--warmup must be >= 0");
    }
    return opt;
}

class QplJobHandle {
public:
    explicit QplJobHandle(qpl_path_t path) {
        uint32_t size = 0;
        require_qpl_ok(qpl_get_job_size(path, &size), "qpl_get_job_size");
        storage_.reset(new uint8_t[size]);
        job_ = reinterpret_cast<qpl_job*>(storage_.get());
        require_qpl_ok(qpl_init_job(path, job_), "qpl_init_job");
    }

    ~QplJobHandle() {
        if (job_ != nullptr) {
            (void)qpl_fini_job(job_);
        }
    }

    QplJobHandle(const QplJobHandle&) = delete;
    QplJobHandle& operator=(const QplJobHandle&) = delete;

    qpl_job* job() const {
        return job_;
    }

private:
    std::unique_ptr<uint8_t[]> storage_;
    qpl_job* job_ = nullptr;
};

uint32_t read_u32_le(const std::vector<uint8_t>& bytes, size_t off) {
    if (off + 4 > bytes.size()) {
        fail("read_u32_le out-of-range");
    }
    uint32_t v = 0;
    v |= static_cast<uint32_t>(bytes[off + 0]) << 0U;
    v |= static_cast<uint32_t>(bytes[off + 1]) << 8U;
    v |= static_cast<uint32_t>(bytes[off + 2]) << 16U;
    v |= static_cast<uint32_t>(bytes[off + 3]) << 24U;
    return v;
}

CompressedBlocks compress_deflate_blocks(
    const std::vector<uint8_t>& raw,
    size_t block_size,
    qpl_path_t path) {
    if (block_size == 0) {
        fail("block_size must be > 0");
    }

    CompressedBlocks out;
    out.block_size = block_size;
    if (raw.empty()) {
        return out;
    }

    QplJobHandle compressor(path);
    qpl_job* job = compressor.job();

    const size_t blocks = (raw.size() + block_size - 1U) / block_size;
    out.raw_block_sizes.resize(blocks, 0U);
    out.compressed_blocks.resize(blocks);

    for (size_t bid = 0; bid < blocks; ++bid) {
        const size_t off = bid * block_size;
        const size_t raw_len = std::min(block_size, raw.size() - off);
        if (raw_len > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            fail("Block too large for QPL uint32_t API");
        }
        const uint32_t raw_u32 = static_cast<uint32_t>(raw_len);
        const uint32_t bound = qpl_get_safe_deflate_compression_buffer_size(raw_u32);
        if (bound == 0) {
            fail("qpl_get_safe_deflate_compression_buffer_size returned 0");
        }

        std::vector<uint8_t> compressed(bound, 0);
        job->op = qpl_op_compress;
        job->level = qpl_default_level;
        job->next_in_ptr = const_cast<uint8_t*>(raw.data() + off);
        job->available_in = raw_u32;
        job->next_out_ptr = compressed.data();
        job->available_out = bound;
        job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DYNAMIC_HUFFMAN;

        require_qpl_ok(qpl_execute_job(job), "qpl_execute_job(compress block)");
        if (job->total_out == 0 || job->total_out > bound) {
            fail("Invalid compressed size from QPL block compressor");
        }
        compressed.resize(static_cast<size_t>(job->total_out));
        out.raw_block_sizes[bid] = raw_u32;
        out.compressed_blocks[bid] = std::move(compressed);
    }

    return out;
}

void check_decompress_roundtrip(
    const std::vector<uint8_t>& raw,
    const CompressedBlocks& compressed_blocks,
    qpl_path_t path,
    std::string_view tag) {
    if (raw.empty()) {
        return;
    }
    QplJobHandle decompressor(path);
    qpl_job* job = decompressor.job();

    size_t raw_offset = 0;
    for (size_t bid = 0; bid < compressed_blocks.block_count(); ++bid) {
        const uint32_t raw_len = compressed_blocks.raw_block_sizes[bid];
        std::vector<uint8_t> out(raw_len, 0);

        job->op = qpl_op_decompress;
        job->next_in_ptr = const_cast<uint8_t*>(compressed_blocks.compressed_blocks[bid].data());
        job->available_in = static_cast<uint32_t>(compressed_blocks.compressed_blocks[bid].size());
        job->next_out_ptr = out.data();
        job->available_out = raw_len;
        job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;

        require_qpl_ok(
            qpl_execute_job(job),
            std::string("qpl_execute_job(decompress roundtrip ") + std::string(tag) + " block)");
        if (job->total_out != raw_len) {
            fail("Roundtrip decompression size mismatch for " + std::string(tag));
        }
        if (raw_offset + raw_len > raw.size()) {
            fail("Roundtrip decompression offset overflow");
        }
        if (!std::equal(out.begin(), out.end(), raw.begin() + static_cast<std::ptrdiff_t>(raw_offset))) {
            fail("Roundtrip decompression content mismatch for " + std::string(tag));
        }
        raw_offset += raw_len;
    }
    if (raw_offset != raw.size()) {
        fail("Roundtrip decompression total size mismatch for " + std::string(tag));
    }
}

uint64_t count_index_matches_u32(uint32_t total_out, uint32_t elements) {
    const uint64_t index_count = static_cast<uint64_t>(total_out / sizeof(uint32_t));
    return std::min<uint64_t>(index_count, static_cast<uint64_t>(elements));
}

class AsyncScanExecutor {
public:
    explicit AsyncScanExecutor(qpl_path_t path, size_t queue_depth = kMaxInFlightJobs)
        : path_(path) {
        if (queue_depth == 0) {
            fail("Async queue depth must be > 0");
        }
        if (queue_depth > kMaxInFlightJobs) {
            fail("Async queue depth must be <= " + std::to_string(kMaxInFlightJobs));
        }
        slots_.resize(queue_depth);
        for (size_t i = 0; i < queue_depth; ++i) {
            slots_[i].job = std::make_unique<QplJobHandle>(path_);
            free_slots_.push_back(i);
        }
    }

    void prepare_for_tasks(const std::vector<ScanTask>& tasks) {
        if (!pending_slots_.empty()) {
            fail("AsyncScanExecutor pending queue not empty at prepare");
        }

        size_t valid_task_count = 0;
        for (const ScanTask& task : tasks) {
            if (task.compressed != nullptr && task.num_input_elements > 0) {
                ++valid_task_count;
            }
        }
        if (valid_task_count > slots_.size()) {
            fail(
                "Task count (" + std::to_string(valid_task_count) +
                ") exceeds async queue depth (" + std::to_string(slots_.size()) +
                ") for single submit-then-wait mode");
        }

        preallocate_output_buffers(tasks);
    }

    AsyncRunMetrics run_tasks(const std::vector<ScanTask>& tasks) {
        if (!pending_slots_.empty()) {
            fail("AsyncScanExecutor pending queue not empty at run start");
        }

        AsyncRunMetrics out;
        std::vector<size_t> prepared_slots;
        prepared_slots.reserve(tasks.size());
        for (const ScanTask& task : tasks) {
            if (task.compressed == nullptr || task.num_input_elements == 0) {
                continue;
            }
            if (free_slots_.empty()) {
                fail("No free async slots during prepare phase");
            }
            const size_t slot_id = free_slots_.front();
            free_slots_.pop_front();
            prepare_slot(slot_id, task);
            prepared_slots.push_back(slot_id);
        }

        const auto submit_t0 = Clock::now();
        for (size_t slot_id : prepared_slots) {
            submit_slot(slot_id);
        }
        const auto submit_t1 = Clock::now();
        out.submit_time_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(submit_t1 - submit_t0).count());

        const auto wait_t0 = Clock::now();
        while (!pending_slots_.empty()) {
            wait_oldest_one(&out);
        }
        const auto wait_t1 = Clock::now();
        out.wait_time_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(wait_t1 - wait_t0).count());
        return out;
    }

private:
    struct Slot {
        std::unique_ptr<QplJobHandle> job;
        std::vector<uint8_t> output;
        uint32_t submitted_elements = 0;
    };

    static size_t required_output_bytes(uint32_t num_input_elements) {
        return static_cast<size_t>(num_input_elements) * sizeof(uint32_t) + 64U;
    }

    void preallocate_output_buffers(const std::vector<ScanTask>& tasks) {
        size_t max_required = 0;
        for (const ScanTask& task : tasks) {
            if (task.num_input_elements == 0) {
                continue;
            }
            max_required = std::max(max_required, required_output_bytes(task.num_input_elements));
        }
        if (max_required == 0) {
            return;
        }
        for (Slot& slot : slots_) {
            if (slot.output.size() < max_required) {
                slot.output.assign(max_required, 0);
            }
        }
    }

    void prepare_slot(size_t slot_id, const ScanTask& task) {
        Slot& slot = slots_[slot_id];
        const size_t required_out = required_output_bytes(task.num_input_elements);
        if (slot.output.size() < required_out) {
            fail("Slot output is not preallocated for requested task size");
        }
        std::fill_n(slot.output.begin(), required_out, static_cast<uint8_t>(0));

        const std::vector<uint8_t>& compressed = *task.compressed;
        if (compressed.empty()) {
            fail("Compressed input is empty for async scan");
        }
        if (compressed.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            fail("Compressed input too large for QPL uint32_t API");
        }

        qpl_job* job = slot.job->job();
        job->op = qpl_op_scan_eq;
        job->next_in_ptr = const_cast<uint8_t*>(compressed.data());
        job->available_in = static_cast<uint32_t>(compressed.size());
        job->next_out_ptr = slot.output.data();
        job->available_out = static_cast<uint32_t>(slot.output.size());
        job->src1_bit_width = 32;
        job->out_bit_width = qpl_ow_32;
        job->param_low = task.needle_u32;
        job->num_input_elements = task.num_input_elements;
        job->drop_initial_bytes = task.drop_initial_bytes;
        job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
        slot.submitted_elements = task.num_input_elements;
    }

    void submit_slot(size_t slot_id) {
        Slot& slot = slots_[slot_id];
        qpl_job* job = slot.job->job();
        size_t retry_count = 0;
        constexpr size_t kMaxBusyRetry = 1'000'000;
        for (;;) {
            const qpl_status st = qpl_submit_job(job);
            if (st == QPL_STS_OK) {
                pending_slots_.push_back(slot_id);
                if (pending_slots_.size() > kMaxInFlightJobs) {
                    fail("In-flight job count exceeded hard limit " + std::to_string(kMaxInFlightJobs));
                }
                return;
            }
            if (st == QPL_STS_QUEUES_ARE_BUSY_ERR) {
                ++retry_count;
                if (retry_count > kMaxBusyRetry) {
                    fail(
                        "qpl_submit_job(scan_eq+decompress) remained busy after retries for slot " +
                        std::to_string(slot_id));
                }
                continue;
            }
            fail("qpl_submit_job(scan_eq+decompress) failed with qpl_status=" +
                 std::to_string(static_cast<int>(st)));
        }
    }

    void wait_oldest_one(AsyncRunMetrics* out) {
        if (pending_slots_.empty()) {
            return;
        }
        const size_t slot_id = pending_slots_.front();
        pending_slots_.pop_front();

        Slot& slot = slots_[slot_id];
        qpl_job* job = slot.job->job();
        const qpl_status st = qpl_wait_job(job);
        if (st != QPL_STS_OK && st != QPL_STS_JOB_NOT_SUBMITTED) {
            fail("qpl_wait_job(scan_eq+decompress) failed with qpl_status=" +
                 std::to_string(static_cast<int>(st)));
        }
        if (out != nullptr) {
            out->matches += count_index_matches_u32(job->total_out, slot.submitted_elements);
        }
        free_slots_.push_back(slot_id);
    }

    qpl_path_t path_ = qpl_path_auto;
    std::vector<Slot> slots_;
    std::deque<size_t> free_slots_;
    std::deque<size_t> pending_slots_;
};

std::vector<ScanTask> build_aligned_scan_tasks(
    const CompressedBlocks& blocks,
    uint32_t needle_u32) {
    std::vector<ScanTask> tasks;
    tasks.reserve(blocks.block_count());
    for (size_t bid = 0; bid < blocks.block_count(); ++bid) {
        const uint32_t raw_len = blocks.raw_block_sizes[bid];
        const uint32_t elems = raw_len / 4U;
        if (elems == 0) {
            continue;
        }
        tasks.push_back(ScanTask{&blocks.compressed_blocks[bid], 0U, elems, needle_u32});
    }
    return tasks;
}

std::vector<ScanTask> build_legacy_scan_tasks(
    const CompressedBlocks& blocks,
    uint32_t needle_u32) {
    std::vector<ScanTask> tasks;
    tasks.reserve(blocks.block_count() * 4U);
    for (size_t bid = 0; bid < blocks.block_count(); ++bid) {
        const uint32_t raw_len = blocks.raw_block_sizes[bid];
        for (uint32_t shift = 0; shift < 4; ++shift) {
            if (raw_len <= shift) {
                continue;
            }
            const uint32_t elems = (raw_len - shift) / 4U;
            if (elems == 0) {
                continue;
            }
            tasks.push_back(ScanTask{&blocks.compressed_blocks[bid], shift, elems, needle_u32});
        }
    }
    return tasks;
}

std::vector<uint8_t> make_aligned_u32_payload(size_t raw_bytes, std::mt19937_64* rng, uint32_t* needle) {
    if (rng == nullptr || needle == nullptr) {
        fail("Null pointer in make_aligned_u32_payload");
    }
    if (raw_bytes < 4) {
        fail("Aligned payload size must be >= 4 bytes");
    }
    const size_t elems = raw_bytes / 4U;
    if (elems == 0) {
        fail("Aligned payload has zero 32-bit elements");
    }
    std::vector<uint32_t> values(elems, 0);
    for (size_t i = 0; i < elems; ++i) {
        values[i] = static_cast<uint32_t>((*rng)());
    }
    *needle = values[elems / 2U];

    std::vector<uint8_t> bytes(elems * 4U, 0);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<uint8_t> make_legacy_byte_payload(size_t raw_bytes, std::mt19937_64* rng, uint32_t* needle) {
    if (rng == nullptr || needle == nullptr) {
        fail("Null pointer in make_legacy_byte_payload");
    }
    if (raw_bytes < 16) {
        fail("Legacy payload size must be >= 16 bytes");
    }

    std::vector<uint8_t> bytes(raw_bytes, 0);
    for (size_t i = 0; i < raw_bytes; ++i) {
        bytes[i] = static_cast<uint8_t>((*rng)() & 0xFFU);
    }

    size_t off = raw_bytes / 2U;
    if (off + 4U > raw_bytes) {
        off = raw_bytes - 4U;
    }
    if ((off % 4U) == 0U && off + 5U < raw_bytes) {
        ++off;  // Force an unaligned target token.
    }
    *needle = read_u32_le(bytes, off);
    return bytes;
}

SummaryStats summarize_us(std::vector<double> samples_us) {
    if (samples_us.empty()) {
        return {};
    }
    const double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    std::sort(samples_us.begin(), samples_us.end());
    auto pick_pct = [&](double pct) -> double {
        if (samples_us.size() == 1) {
            return samples_us.front();
        }
        const double pos = pct * static_cast<double>(samples_us.size() - 1);
        const size_t lo = static_cast<size_t>(pos);
        const size_t hi = std::min(samples_us.size() - 1, lo + 1);
        const double alpha = pos - static_cast<double>(lo);
        return (1.0 - alpha) * samples_us[lo] + alpha * samples_us[hi];
    };

    SummaryStats s;
    s.avg_us = sum / static_cast<double>(samples_us.size());
    s.p50_us = pick_pct(0.50);
    s.p95_us = pick_pct(0.95);
    return s;
}

double logical_throughput_mb_s(uint64_t raw_bytes, double avg_us) {
    if (avg_us <= 0.0) {
        return 0.0;
    }
    const double sec = avg_us / 1e6;
    const double mb = static_cast<double>(raw_bytes) / (1024.0 * 1024.0);
    return mb / sec;
}

std::array<ResultRow, 2> run_one_size(const Options& opt, uint64_t size_bytes_u64, std::mt19937_64* rng) {
    if (rng == nullptr) {
        fail("RNG pointer is null");
    }

    if (size_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        fail("Size too large for current benchmark: " + format_size_bytes_human(size_bytes_u64));
    }
    const size_t raw_bytes = static_cast<size_t>(size_bytes_u64);

    uint32_t aligned_needle = 0;
    uint32_t legacy_needle = 0;
    const std::vector<uint8_t> aligned_raw = make_aligned_u32_payload(raw_bytes, rng, &aligned_needle);
    const std::vector<uint8_t> legacy_raw = make_legacy_byte_payload(raw_bytes, rng, &legacy_needle);

    const size_t block_size = opt.block_size_kb * 1024ULL;
    // Keep compressed source generation deterministic and robust across runs.
    // Scan/decompress execution path is still controlled by --path.
    constexpr qpl_path_t kCompressionPath = qpl_path_software;
    const CompressedBlocks aligned_compressed =
        compress_deflate_blocks(aligned_raw, block_size, kCompressionPath);
    const CompressedBlocks legacy_compressed =
        compress_deflate_blocks(legacy_raw, block_size, kCompressionPath);
    check_decompress_roundtrip(aligned_raw, aligned_compressed, kCompressionPath, "aligned");
    check_decompress_roundtrip(legacy_raw, legacy_compressed, kCompressionPath, "legacy");

    const std::vector<ScanTask> aligned_tasks = build_aligned_scan_tasks(aligned_compressed, aligned_needle);
    const std::vector<ScanTask> legacy_tasks = build_legacy_scan_tasks(legacy_compressed, legacy_needle);

    AsyncScanExecutor async_exec(opt.path, kMaxInFlightJobs);
    async_exec.prepare_for_tasks(aligned_tasks);
    async_exec.prepare_for_tasks(legacy_tasks);
    volatile uint64_t match_sink = 0;

    for (int i = 0; i < opt.warmup; ++i) {
        match_sink += async_exec.run_tasks(aligned_tasks).matches;
        match_sink += async_exec.run_tasks(legacy_tasks).matches;
    }

    std::vector<double> aligned_us;
    std::vector<double> legacy_us;
    std::vector<double> aligned_wait_us;
    std::vector<double> legacy_wait_us;
    std::vector<double> aligned_submit_us;
    std::vector<double> legacy_submit_us;
    aligned_us.reserve(static_cast<size_t>(opt.iterations));
    legacy_us.reserve(static_cast<size_t>(opt.iterations));
    aligned_wait_us.reserve(static_cast<size_t>(opt.iterations));
    legacy_wait_us.reserve(static_cast<size_t>(opt.iterations));
    aligned_submit_us.reserve(static_cast<size_t>(opt.iterations));
    legacy_submit_us.reserve(static_cast<size_t>(opt.iterations));

    for (int it = 0; it < opt.iterations; ++it) {
        {
            const auto t0 = Clock::now();
            const AsyncRunMetrics m = async_exec.run_tasks(aligned_tasks);
            match_sink += m.matches;
            const auto t1 = Clock::now();
            aligned_us.push_back(
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / 1e3);
            aligned_wait_us.push_back(static_cast<double>(m.wait_time_ns) / 1e3);
            aligned_submit_us.push_back(static_cast<double>(m.submit_time_ns) / 1e3);
        }
        {
            const auto t0 = Clock::now();
            const AsyncRunMetrics m = async_exec.run_tasks(legacy_tasks);
            match_sink += m.matches;
            const auto t1 = Clock::now();
            legacy_us.push_back(
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / 1e3);
            legacy_wait_us.push_back(static_cast<double>(m.wait_time_ns) / 1e3);
            legacy_submit_us.push_back(static_cast<double>(m.submit_time_ns) / 1e3);
        }
    }

    (void)match_sink;

    const SummaryStats aligned_stats = summarize_us(aligned_us);
    const SummaryStats legacy_stats = summarize_us(legacy_us);
    const SummaryStats aligned_wait_stats = summarize_us(aligned_wait_us);
    const SummaryStats legacy_wait_stats = summarize_us(legacy_wait_us);
    const SummaryStats aligned_submit_stats = summarize_us(aligned_submit_us);
    const SummaryStats legacy_submit_stats = summarize_us(legacy_submit_us);
    const double aligned_slowdown_base = (aligned_stats.avg_us > 0.0) ? aligned_stats.avg_us : 1.0;

    ResultRow aligned_row;
    aligned_row.layout = "aligned_4byte";
    aligned_row.size_mb = bytes_to_mb(size_bytes_u64);
    aligned_row.size_bytes = size_bytes_u64;
    aligned_row.raw_bytes = static_cast<uint64_t>(aligned_raw.size());
    aligned_row.compressed_bytes = aligned_compressed.compressed_bytes_total();
    aligned_row.compression_ratio =
        (aligned_row.compressed_bytes > 0)
            ? static_cast<double>(aligned_row.raw_bytes) / static_cast<double>(aligned_row.compressed_bytes)
            : 0.0;
    aligned_row.jobs_per_query = static_cast<int>(aligned_tasks.size());
    aligned_row.iterations = opt.iterations;
    aligned_row.avg_us = aligned_stats.avg_us;
    aligned_row.p50_us = aligned_stats.p50_us;
    aligned_row.p95_us = aligned_stats.p95_us;
    aligned_row.avg_wait_us = aligned_wait_stats.avg_us;
    aligned_row.p50_wait_us = aligned_wait_stats.p50_us;
    aligned_row.p95_wait_us = aligned_wait_stats.p95_us;
    aligned_row.avg_submit_us = aligned_submit_stats.avg_us;
    aligned_row.p50_submit_us = aligned_submit_stats.p50_us;
    aligned_row.p95_submit_us = aligned_submit_stats.p95_us;
    aligned_row.throughput_mb_s = logical_throughput_mb_s(aligned_row.raw_bytes, aligned_row.avg_us);
    aligned_row.slowdown_vs_aligned = 1.0;

    ResultRow legacy_row;
    legacy_row.layout = "legacy_unaligned_4shift";
    legacy_row.size_mb = bytes_to_mb(size_bytes_u64);
    legacy_row.size_bytes = size_bytes_u64;
    legacy_row.raw_bytes = static_cast<uint64_t>(legacy_raw.size());
    legacy_row.compressed_bytes = legacy_compressed.compressed_bytes_total();
    legacy_row.compression_ratio =
        (legacy_row.compressed_bytes > 0)
            ? static_cast<double>(legacy_row.raw_bytes) / static_cast<double>(legacy_row.compressed_bytes)
            : 0.0;
    legacy_row.jobs_per_query = static_cast<int>(legacy_tasks.size());
    legacy_row.iterations = opt.iterations;
    legacy_row.avg_us = legacy_stats.avg_us;
    legacy_row.p50_us = legacy_stats.p50_us;
    legacy_row.p95_us = legacy_stats.p95_us;
    legacy_row.avg_wait_us = legacy_wait_stats.avg_us;
    legacy_row.p50_wait_us = legacy_wait_stats.p50_us;
    legacy_row.p95_wait_us = legacy_wait_stats.p95_us;
    legacy_row.avg_submit_us = legacy_submit_stats.avg_us;
    legacy_row.p50_submit_us = legacy_submit_stats.p50_us;
    legacy_row.p95_submit_us = legacy_submit_stats.p95_us;
    legacy_row.throughput_mb_s = logical_throughput_mb_s(legacy_row.raw_bytes, legacy_row.avg_us);
    legacy_row.slowdown_vs_aligned =
        (aligned_slowdown_base > 0.0) ? (legacy_row.avg_us / aligned_slowdown_base) : 0.0;

    return {aligned_row, legacy_row};
}

void write_csv(const std::string& path, const std::string& path_name, const std::vector<ResultRow>& rows) {
    fs::path p(path);
    if (!p.parent_path().empty()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        fail("Failed to open output CSV: " + path);
    }

    out << "path,layout,size_mb,size_bytes,raw_bytes,compressed_bytes,compression_ratio,jobs_per_query,"
           "iterations,avg_us,p50_us,p95_us,avg_wait_us,p50_wait_us,p95_wait_us,avg_submit_us,p50_submit_us,p95_submit_us,throughput_mb_s,slowdown_vs_aligned\n";
    out << std::fixed << std::setprecision(6);
    for (const ResultRow& r : rows) {
        out << path_name << ","
            << r.layout << ","
            << r.size_mb << ","
            << r.size_bytes << ","
            << r.raw_bytes << ","
            << r.compressed_bytes << ","
            << r.compression_ratio << ","
            << r.jobs_per_query << ","
            << r.iterations << ","
            << r.avg_us << ","
            << r.p50_us << ","
            << r.p95_us << ","
            << r.avg_wait_us << ","
            << r.p50_wait_us << ","
            << r.p95_wait_us << ","
            << r.avg_submit_us << ","
            << r.p50_submit_us << ","
            << r.p95_submit_us << ","
            << r.throughput_mb_s << ","
            << r.slowdown_vs_aligned
            << "\n";
    }
}

void print_table(const std::vector<ResultRow>& rows) {
    std::cout
        << "\nResults (latency in us, throughput in MB/s)\n"
        << "-------------------------------------------------------------------------------\n"
        << "layout                      size   jobs  avg_us  wait_us submit_us  p95_us    thrpt    slowdown\n"
        << "-------------------------------------------------------------------------------\n";
    for (const ResultRow& r : rows) {
        const std::string size_label = format_size_bytes_human(r.size_bytes);
        std::cout
            << std::left << std::setw(25) << r.layout
            << std::right << std::setw(7) << size_label
            << std::setw(6) << r.jobs_per_query
            << std::setw(10) << std::fixed << std::setprecision(2) << r.avg_us
            << std::setw(9) << r.avg_wait_us
            << std::setw(10) << r.avg_submit_us
            << std::setw(10) << r.p95_us
            << std::setw(10) << std::setprecision(1) << r.throughput_mb_s
            << std::setw(10) << std::setprecision(2) << r.slowdown_vs_aligned
            << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        std::mt19937_64 rng(opt.seed);

        std::cout << "QPL version: " << qpl_get_library_version() << "\n";
        std::cout << "Path: " << opt.path_name << "\n";
        std::cout << "Sizes: ";
        for (size_t i = 0; i < opt.sizes_bytes.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << format_size_bytes_human(opt.sizes_bytes[i]);
        }
        std::cout << "\nBlock size: " << opt.block_size_kb << "KB";
        std::cout << "\nIterations: " << opt.iterations << ", warmup: " << opt.warmup << "\n";

        std::vector<ResultRow> rows;
        rows.reserve(opt.sizes_bytes.size() * 2U);

        for (uint64_t size_bytes : opt.sizes_bytes) {
            std::cout << "\n[run] size=" << format_size_bytes_human(size_bytes) << " ...\n";
            const auto pair = run_one_size(opt, size_bytes, &rng);
            rows.push_back(pair[0]);
            rows.push_back(pair[1]);
        }

        write_csv(opt.output_csv, opt.path_name, rows);
        print_table(rows);

        std::cout << "\nWrote CSV: " << opt.output_csv << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
