#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <x86intrin.h>

#include <lz4.h>
#include <nlohmann/json.hpp>
#include <qpl/c_api/dictionary.h>
#include <qpl/c_api/huffman_table.h>
#include <qpl/qpl.h>
#include <zlib.h>

using json = nlohmann::json;

namespace {

constexpr const char* kDefaultJsonDir = "/home/jykang5/compass/json_dataset";
constexpr const char* kDefaultMetadataProfile =
    "/home/jykang5/compass/profile/metadata_compressability/profiling.json";

constexpr std::size_t kBlockSize8KB = 8ULL * 1024ULL;
constexpr std::size_t kBlockSize16KB = 16ULL * 1024ULL;
constexpr std::size_t kBlockSize64KB = 64ULL * 1024ULL;
constexpr std::size_t kBlockSize128KB = 128ULL * 1024ULL;
constexpr std::size_t kBlockSize256KB = 256ULL * 1024ULL;
constexpr std::size_t kBlockSize512KB = 512ULL * 1024ULL;
constexpr std::size_t kBlockSize1MB = 1ULL * 1024ULL * 1024ULL;
constexpr std::size_t kBlockSize2MB = 2ULL * 1024ULL * 1024ULL;

struct Config {
    std::string json_dir = kDefaultJsonDir;
    std::string metadata_profile = kDefaultMetadataProfile;
    std::vector<std::string> datasets = {"hnm", "laion"};
    std::size_t max_rows = 0;  // 0: all rows
};

struct DatasetBytes {
    std::size_t rows = 0;
    std::vector<std::string> columns;
    std::vector<uint8_t> bytes;
};

struct ChunkedCompressed {
    std::vector<std::vector<uint8_t>> chunks;
    std::vector<std::uint32_t> original_chunk_sizes;
    std::vector<uint8_t> dictionary_raw;
    std::size_t compressed_bytes = 0;
};

struct BenchmarkResult {
    std::string dataset;
    std::string algorithm;
    std::size_t block_size = 0;
    std::size_t original_bytes = 0;
    std::size_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    std::uint64_t compression_cycles = 0;
    long long decompression_latency_ns = 0;
    bool verified = false;
    std::string error;
};

std::string trim_copy(const std::string& in) {
    std::size_t first = 0;
    while (first < in.size() &&
           std::isspace(static_cast<unsigned char>(in[first])) != 0) {
        ++first;
    }
    std::size_t last = in.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(in[last - 1])) != 0) {
        --last;
    }
    return in.substr(first, last - first);
}

bool parse_int64_str(const std::string& s, long long& out) {
    const std::string trimmed = trim_copy(s);
    if (trimmed.empty()) {
        return false;
    }
    try {
        std::size_t pos = 0;
        long long value = std::stoll(trimmed, &pos);
        if (pos != trimmed.size()) {
            return false;
        }
        out = value;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_float64_str(const std::string& s, double& out) {
    const std::string trimmed = trim_copy(s);
    if (trimmed.empty()) {
        return false;
    }
    try {
        std::size_t pos = 0;
        double value = std::stod(trimmed, &pos);
        if (pos != trimmed.size()) {
            return false;
        }
        out = value;
        return true;
    } catch (...) {
        return false;
    }
}

std::string json_value_to_string(const json& v) {
    if (v.is_null()) {
        return "";
    }
    if (v.is_string()) {
        return v.get<std::string>();
    }
    return v.dump();
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--json-dir <path>] [--metadata-profile <path>]"
                 " [--dataset <hnm|laion|all>] [--max-rows <N>]\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
    std::set<std::string> dataset_set(cfg.datasets.begin(), cfg.datasets.end());

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return argv[++i];
        };

        if (arg == "--json-dir") {
            cfg.json_dir = need_value(arg);
        } else if (arg == "--metadata-profile") {
            cfg.metadata_profile = need_value(arg);
        } else if (arg == "--dataset") {
            const std::string value = need_value(arg);
            if (value == "all") {
                dataset_set = {"hnm", "laion"};
            } else if (value == "hnm" || value == "laion") {
                dataset_set.clear();
                dataset_set.insert(value);
            } else {
                throw std::runtime_error(
                    "Invalid dataset. Use hnm, laion, or all.");
            }
        } else if (arg == "--max-rows") {
            const std::string value = need_value(arg);
            try {
                cfg.max_rows = static_cast<std::size_t>(std::stoull(value));
            } catch (...) {
                throw std::runtime_error("Invalid --max-rows value: " + value);
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    cfg.datasets.assign(dataset_set.begin(), dataset_set.end());
    return true;
}

std::unordered_map<std::string, std::vector<std::string>> load_included_columns(
    const std::string& metadata_profile_path) {
    std::ifstream in(metadata_profile_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open metadata profile: " +
                                 metadata_profile_path);
    }

    json profile = json::parse(in, nullptr, true, true);
    if (!profile.contains("datasets") || !profile["datasets"].is_array()) {
        throw std::runtime_error(
            "Invalid metadata profile format: missing datasets array");
    }

    std::unordered_map<std::string, std::vector<std::string>> result;
    for (const auto& ds : profile["datasets"]) {
        if (!ds.contains("name") || !ds.contains("metadata_columns")) {
            continue;
        }
        const std::string name = ds["name"].get<std::string>();
        const auto& metadata_columns = ds["metadata_columns"];
        if (!metadata_columns.contains("included_names") ||
            !metadata_columns["included_names"].is_array()) {
            continue;
        }
        result[name] =
            metadata_columns["included_names"].get<std::vector<std::string>>();
    }

    return result;
}

template <typename T>
void append_scalar_bytes(std::vector<uint8_t>& out, const T& value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

void append_raw_column_bytes(const std::string& column_name,
                             const std::vector<std::string>& values,
                             std::vector<uint8_t>& out) {
    (void)column_name;
    bool all_int = true;
    bool all_numeric = true;

    for (const auto& s : values) {
        const std::string trimmed = trim_copy(s);
        if (trimmed.empty()) {
            continue;
        }
        long long iv = 0;
        if (parse_int64_str(trimmed, iv)) {
            continue;
        }
        double dv = 0.0;
        if (parse_float64_str(trimmed, dv)) {
            all_int = false;
            continue;
        }
        all_int = false;
        all_numeric = false;
        break;
    }

    if (all_numeric && all_int) {
        for (const auto& s : values) {
            long long iv = 0;
            if (!parse_int64_str(s, iv)) {
                iv = 0;
            }
            if (iv < static_cast<long long>(std::numeric_limits<int32_t>::min())) {
                iv = std::numeric_limits<int32_t>::min();
            }
            if (iv > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
                iv = std::numeric_limits<int32_t>::max();
            }
            const int32_t v = static_cast<int32_t>(iv);
            append_scalar_bytes(out, v);
        }
        return;
    }

    if (all_numeric && !all_int) {
        for (const auto& s : values) {
            double dv = 0.0;
            if (!parse_float64_str(s, dv)) {
                dv = 0.0;
            }
            const float fv = static_cast<float>(dv);
            append_scalar_bytes(out, fv);
        }
        return;
    }

    for (const auto& s : values) {
        out.insert(out.end(), s.begin(), s.end());
        out.push_back('\n');
    }
}

DatasetBytes build_dataset_bytes(const std::string& jsonl_path,
                                 const std::vector<std::string>& selected_columns,
                                 std::size_t max_rows) {
    std::ifstream in(jsonl_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open dataset file: " + jsonl_path);
    }

    std::unordered_map<std::string, std::vector<std::string>> per_column_values;
    for (const auto& col : selected_columns) {
        per_column_values[col] = {};
        if (max_rows > 0) {
            per_column_values[col].reserve(max_rows);
        }
    }

    std::string line;
    std::size_t row_count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        json row = json::parse(line, nullptr, false, true);
        if (row.is_discarded() || !row.is_object()) {
            throw std::runtime_error("Invalid JSONL row in file: " + jsonl_path);
        }

        for (const auto& col : selected_columns) {
            auto it = row.find(col);
            if (it == row.end() || it->is_null()) {
                per_column_values[col].push_back("");
            } else {
                per_column_values[col].push_back(json_value_to_string(*it));
            }
        }

        ++row_count;
        if (max_rows > 0 && row_count >= max_rows) {
            break;
        }
    }

    if (row_count == 0) {
        throw std::runtime_error("No rows parsed from file: " + jsonl_path);
    }

    DatasetBytes out;
    out.rows = row_count;
    out.columns = selected_columns;
    out.bytes.reserve(row_count * selected_columns.size() * sizeof(int32_t));

    for (const auto& col : selected_columns) {
        append_raw_column_bytes(col, per_column_values[col], out.bytes);
    }

    return out;
}

std::size_t total_original_bytes(const ChunkedCompressed& c) {
    std::size_t total = 0;
    for (const auto& s : c.original_chunk_sizes) {
        total += static_cast<std::size_t>(s);
    }
    return total;
}

using RawBlocks = std::vector<std::vector<uint8_t>>;

RawBlocks split_into_raw_blocks(const std::vector<uint8_t>& data,
                                std::size_t block_size) {
    RawBlocks blocks;
    if (block_size == 0 || data.empty()) {
        return blocks;
    }
    const std::size_t num_blocks = (data.size() + block_size - 1ULL) / block_size;
    blocks.reserve(num_blocks);
    for (std::size_t offset = 0; offset < data.size(); offset += block_size) {
        const std::size_t chunk_size = std::min(block_size, data.size() - offset);
        blocks.emplace_back(data.begin() + static_cast<std::ptrdiff_t>(offset),
                            data.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
    }
    return blocks;
}

std::vector<uint8_t> collect_dictionary_raw(const RawBlocks& raw_blocks,
                                            std::size_t max_bytes) {
    std::vector<uint8_t> dictionary_raw;
    if (max_bytes == 0) {
        return dictionary_raw;
    }

    std::size_t available_bytes = 0;
    for (const auto& block : raw_blocks) {
        available_bytes += block.size();
        if (available_bytes >= max_bytes) {
            break;
        }
    }
    dictionary_raw.reserve(std::min(max_bytes, available_bytes));

    for (const auto& block : raw_blocks) {
        if (dictionary_raw.size() >= max_bytes) {
            break;
        }
        const std::size_t remaining = max_bytes - dictionary_raw.size();
        const std::size_t to_copy = std::min(remaining, block.size());
        dictionary_raw.insert(dictionary_raw.end(), block.begin(),
                              block.begin() + static_cast<std::ptrdiff_t>(to_copy));
    }
    return dictionary_raw;
}

ChunkedCompressed compress_lz4(const RawBlocks& raw_blocks,
                               std::uint64_t& cycles) {
    ChunkedCompressed out;
    out.chunks.resize(raw_blocks.size());
    out.original_chunk_sizes.resize(raw_blocks.size(), 0);

    const std::uint64_t t0 = __rdtsc();
    for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
        const auto& raw_block = raw_blocks[i];
        const std::size_t chunk_size = raw_block.size();
        const int max_compressed = LZ4_compressBound(
            static_cast<int>(chunk_size));
        if (max_compressed <= 0) {
            throw std::runtime_error("LZ4_compressBound failed.");
        }
        out.chunks[i].resize(static_cast<std::size_t>(max_compressed));
        const int compressed = LZ4_compress_default(
            reinterpret_cast<const char*>(raw_block.data()),
            reinterpret_cast<char*>(out.chunks[i].data()),
            static_cast<int>(chunk_size),
            max_compressed);
        if (compressed <= 0) {
            throw std::runtime_error("LZ4 compression failed.");
        }
        out.chunks[i].resize(static_cast<std::size_t>(compressed));
        out.compressed_bytes += out.chunks[i].size();
        out.original_chunk_sizes[i] = static_cast<std::uint32_t>(chunk_size);
    }
    const std::uint64_t t1 = __rdtsc();
    cycles = t1 - t0;
    return out;
}

bool decompress_lz4(const ChunkedCompressed& compressed,
                    std::vector<uint8_t>& output,
                    long long& latency_ns,
                    std::string& error) {
    const std::size_t total_size = total_original_bytes(compressed);
    output.assign(total_size, 0);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::size_t offset = 0;
    for (std::size_t i = 0; i < compressed.chunks.size(); ++i) {
        const auto& chunk = compressed.chunks[i];
        const int expected = static_cast<int>(compressed.original_chunk_sizes[i]);
        const int decompressed = LZ4_decompress_safe(
            reinterpret_cast<const char*>(chunk.data()),
            reinterpret_cast<char*>(output.data() + offset),
            static_cast<int>(chunk.size()),
            expected);
        if (decompressed != expected) {
            std::ostringstream oss;
            oss << "LZ4 decompression failed at chunk " << i
                << " (expected " << expected << ", got " << decompressed << ")";
            error = oss.str();
            return false;
        }
        offset += static_cast<std::size_t>(expected);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    latency_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return true;
}

ChunkedCompressed compress_deflate(const RawBlocks& raw_blocks,
                                   std::uint64_t& cycles) {
    ChunkedCompressed out;
    out.chunks.resize(raw_blocks.size());
    out.original_chunk_sizes.resize(raw_blocks.size(), 0);

    const std::uint64_t t0 = __rdtsc();
    for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
        const auto& raw_block = raw_blocks[i];
        const std::size_t chunk_size = raw_block.size();
        uLongf dst_cap = compressBound(static_cast<uLong>(chunk_size));
        out.chunks[i].resize(static_cast<std::size_t>(dst_cap));
        int ret = compress2(out.chunks[i].data(), &dst_cap, raw_block.data(),
                            static_cast<uLong>(chunk_size), 1);
        if (ret != Z_OK) {
            throw std::runtime_error("Deflate compression failed.");
        }
        out.chunks[i].resize(static_cast<std::size_t>(dst_cap));
        out.compressed_bytes += out.chunks[i].size();
        out.original_chunk_sizes[i] = static_cast<std::uint32_t>(chunk_size);
    }
    const std::uint64_t t1 = __rdtsc();
    cycles = t1 - t0;
    return out;
}

bool decompress_deflate(const ChunkedCompressed& compressed,
                        std::vector<uint8_t>& output,
                        long long& latency_ns,
                        std::string& error) {
    const std::size_t total_size = total_original_bytes(compressed);
    output.assign(total_size, 0);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::size_t offset = 0;
    for (std::size_t i = 0; i < compressed.chunks.size(); ++i) {
        const auto& chunk = compressed.chunks[i];
        uLongf out_size = compressed.original_chunk_sizes[i];
        int ret = uncompress(output.data() + offset, &out_size, chunk.data(),
                             static_cast<uLong>(chunk.size()));
        if (ret != Z_OK ||
            out_size != compressed.original_chunk_sizes[i]) {
            std::ostringstream oss;
            oss << "Deflate decompression failed at chunk " << i;
            error = oss.str();
            return false;
        }
        offset += static_cast<std::size_t>(out_size);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    latency_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return true;
}

class IaaJob {
public:
    IaaJob() {
        qpl_status status = qpl_get_job_size(qpl_path_hardware, &job_size_);
        if (status != QPL_STS_OK) {
            throw std::runtime_error("qpl_get_job_size failed with status " +
                                     std::to_string(status));
        }

        buffer_ = std::make_unique<uint8_t[]>(job_size_);
        job_ = reinterpret_cast<qpl_job*>(buffer_.get());

        status = qpl_init_job(qpl_path_hardware, job_);
        if (status != QPL_STS_OK) {
            throw std::runtime_error("qpl_init_job(hardware) failed with status " +
                                     std::to_string(status));
        }
    }

    ~IaaJob() {
        if (job_ != nullptr) {
            (void)qpl_fini_job(job_);
        }
    }

    qpl_job* get() { return job_; }

private:
    uint32_t job_size_ = 0;
    std::unique_ptr<uint8_t[]> buffer_;
    qpl_job* job_ = nullptr;
};

class IaaJobPool {
public:
    explicit IaaJobPool(std::size_t queue_size) : queue_size_(queue_size) {
        if (queue_size_ == 0) {
            throw std::runtime_error("IAA job pool queue size must be > 0");
        }

        qpl_status status = qpl_get_job_size(qpl_path_hardware, &job_size_);
        if (status != QPL_STS_OK) {
            throw std::runtime_error("qpl_get_job_size failed with status " +
                                     std::to_string(status));
        }

        buffers_.resize(queue_size_);
        jobs_.resize(queue_size_, nullptr);

        for (std::size_t i = 0; i < queue_size_; ++i) {
            buffers_[i] = std::make_unique<uint8_t[]>(job_size_);
            jobs_[i] = reinterpret_cast<qpl_job*>(buffers_[i].get());
            status = qpl_init_job(qpl_path_hardware, jobs_[i]);
            if (status != QPL_STS_OK) {
                throw std::runtime_error(
                    "qpl_init_job(hardware) failed at slot " + std::to_string(i) +
                    " with status " + std::to_string(status));
            }
        }
    }

    ~IaaJobPool() {
        for (qpl_job* job : jobs_) {
            if (job != nullptr) {
                (void)qpl_fini_job(job);
            }
        }
    }

    qpl_job* get(std::size_t slot) { return jobs_.at(slot); }
    std::size_t size() const { return queue_size_; }

private:
    std::size_t queue_size_ = 0;
    uint32_t job_size_ = 0;
    std::vector<std::unique_ptr<uint8_t[]>> buffers_;
    std::vector<qpl_job*> jobs_;
};

ChunkedCompressed compress_iaa(const RawBlocks& raw_blocks,
                               std::uint64_t& cycles) {
    ChunkedCompressed out;
    const std::size_t chunk_count = raw_blocks.size();
    out.chunks.resize(chunk_count);
    out.original_chunk_sizes.resize(chunk_count, 0);

    if (chunk_count == 0) {
        cycles = 0;
        return out;
    }

    constexpr std::size_t kQueueSize = 128;
    constexpr std::size_t kWaitBatch = 64;
    constexpr std::size_t kIaaOutBufferSize = kBlockSize2MB;
    constexpr std::size_t kDictionaryRawSize = 4ULL * 1024ULL;
    const qpl_path_t execution_path = qpl_path_hardware;
    const std::size_t queue_size = std::min<std::size_t>(kQueueSize, chunk_count);
    IaaJobPool pool(queue_size);

    for (std::size_t i = 0; i < chunk_count; ++i) {
        const std::size_t raw_size = raw_blocks[i].size();
        if (raw_size > kBlockSize2MB) {
            throw std::runtime_error(
                "Raw block size exceeds 2MB while 2MB-only compression is required.");
        }
        out.original_chunk_sizes[i] = static_cast<std::uint32_t>(raw_size);
    }

    std::vector<std::vector<uint8_t>> slot_buffers(queue_size);

    struct HuffmanTableGuard {
        qpl_huffman_table_t table = nullptr;
        ~HuffmanTableGuard() {
            if (table != nullptr) {
                (void)qpl_huffman_table_destroy(table);
            }
        }
    } huffman_guard;

    struct DictionaryGuard {
        std::unique_ptr<uint8_t[]> buffer;
        qpl_dictionary* dictionary = nullptr;
    } dictionary_guard;

    allocator_t default_allocator_c = {malloc, free};
    qpl_status status = qpl_deflate_huffman_table_create(
        compression_table_type, execution_path, default_allocator_c, &huffman_guard.table);
    if (status != QPL_STS_OK) {
        throw std::runtime_error(
            "IAA static Huffman table creation failed with status " +
            std::to_string(status));
    }

    out.dictionary_raw = collect_dictionary_raw(raw_blocks, kDictionaryRawSize);
    if (!out.dictionary_raw.empty()) {
        const sw_compression_level sw_compr_level = SW_NONE;
        const hw_compression_level hw_compr_level = HW_LEVEL_1;
        const std::size_t dictionary_buffer_size = qpl_get_dictionary_size(
            sw_compr_level, hw_compr_level, out.dictionary_raw.size());
        if (dictionary_buffer_size == 0) {
            throw std::runtime_error("IAA dictionary buffer size is zero.");
        }

        dictionary_guard.buffer = std::make_unique<uint8_t[]>(dictionary_buffer_size);
        dictionary_guard.dictionary =
            reinterpret_cast<qpl_dictionary*>(dictionary_guard.buffer.get());
        status = qpl_build_dictionary(dictionary_guard.dictionary,
                                      sw_compr_level,
                                      hw_compr_level,
                                      out.dictionary_raw.data(),
                                      out.dictionary_raw.size());
        if (status != QPL_STS_OK) {
            throw std::runtime_error("IAA dictionary build failed with status " +
                                     std::to_string(status));
        }
    }
    bool use_dictionary = (dictionary_guard.dictionary != nullptr);

    auto submit_compress_job = [&](std::size_t chunk_index) -> void {
        const std::size_t slot = chunk_index % pool.size();
        qpl_job* job = pool.get(slot);

        const auto& raw_block = raw_blocks[chunk_index];
        const uint32_t raw_size = out.original_chunk_sizes[chunk_index];

        qpl_histogram deflate_histogram {};
        status = qpl_gather_deflate_statistics(
            const_cast<uint8_t*>(raw_block.data()), raw_size, &deflate_histogram,
            qpl_default_level, execution_path);
        if (status != QPL_STS_OK) {
            throw std::runtime_error(
                "IAA gather deflate statistics failed at chunk " +
                std::to_string(chunk_index) + " with status " + std::to_string(status));
        }

        status = qpl_huffman_table_init_with_histogram(
            huffman_guard.table, &deflate_histogram);
        if (status != QPL_STS_OK) {
            throw std::runtime_error(
                "IAA Huffman table init failed at chunk " +
                std::to_string(chunk_index) + " with status " + std::to_string(status));
        }

        // Excluded from cycle accounting by request. Keep output buffer fixed at 2MB,
        // including for the last partial input block.
        slot_buffers[slot].resize(kIaaOutBufferSize);

        job->op = qpl_op_compress;
        job->level = qpl_default_level;
        job->next_in_ptr = const_cast<uint8_t*>(raw_block.data());
        job->available_in = raw_size;
        job->next_out_ptr = slot_buffers[slot].data();
        job->available_out = static_cast<uint32_t>(kIaaOutBufferSize);
        job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_OMIT_VERIFY;
        job->dictionary = use_dictionary ? dictionary_guard.dictionary : nullptr;
        job->huffman_table = huffman_guard.table;

        const std::uint64_t t0 = __rdtsc();
        status = qpl_submit_job(job);
        const std::uint64_t t1 = __rdtsc();
        cycles += (t1 - t0);
        if (status == QPL_STS_NOT_SUPPORTED_MODE_ERR && use_dictionary) {
            // Some Intel IAA devices do not support dictionary mode on hardware.
            use_dictionary = false;
            out.dictionary_raw.clear();
            job->dictionary = nullptr;

            const std::uint64_t t2 = __rdtsc();
            status = qpl_submit_job(job);
            const std::uint64_t t3 = __rdtsc();
            cycles += (t3 - t2);
        }
        if (status != QPL_STS_OK) {
            throw std::runtime_error("IAA compression submit failed at chunk " +
                                     std::to_string(chunk_index) + " with status " +
                                     std::to_string(status));
        }
    };

    auto wait_compress_job = [&](std::size_t chunk_index) -> void {
        const std::size_t slot = chunk_index % pool.size();
        qpl_job* job = pool.get(slot);

        const std::uint64_t t0 = __rdtsc();
        qpl_status status = qpl_wait_job(job);
        const std::uint64_t t1 = __rdtsc();
        cycles += (t1 - t0);
        if (status != QPL_STS_OK) {
            throw std::runtime_error("IAA compression wait failed at chunk " +
                                     std::to_string(chunk_index) + " with status " +
                                     std::to_string(status));
        }

        const std::size_t produced = static_cast<std::size_t>(job->total_out);
        if (produced > slot_buffers[slot].size()) {
            throw std::runtime_error("IAA compression produced invalid size at chunk " +
                                     std::to_string(chunk_index));
        }

        // Excluded from cycle accounting by request.
        out.chunks[chunk_index].assign(slot_buffers[slot].begin(),
                                       slot_buffers[slot].begin() + produced);
        out.compressed_bytes += produced;
    };

    cycles = 0;

    std::size_t next_submit = 0;
    std::size_t next_wait = 0;

    const std::size_t initial_submit =
        std::min<std::size_t>(pool.size(), chunk_count);
    for (; next_submit < initial_submit; ++next_submit) {
        submit_compress_job(next_submit);
    }

    while (next_wait < chunk_count) {
        const std::size_t wait_count =
            std::min<std::size_t>(kWaitBatch, chunk_count - next_wait);
        for (std::size_t i = 0; i < wait_count; ++i) {
            wait_compress_job(next_wait + i);
        }
        next_wait += wait_count;

        const std::size_t submit_count =
            std::min<std::size_t>(wait_count, chunk_count - next_submit);
        for (std::size_t i = 0; i < submit_count; ++i) {
            submit_compress_job(next_submit + i);
        }
        next_submit += submit_count;
    }

    return out;
}

bool decompress_iaa(const ChunkedCompressed& compressed,
                    std::vector<uint8_t>& output,
                    long long& latency_ns,
                    std::string& error) {
    const std::size_t total_size = total_original_bytes(compressed);
    output.assign(total_size, 0);

    if (compressed.chunks.empty()) {
        latency_ns = 0;
        return true;
    }

    constexpr std::size_t kQueueSize = 128;
    constexpr std::size_t kWaitBatch = 64;

    const std::size_t queue_size = std::min<std::size_t>(kQueueSize, compressed.chunks.size());
    IaaJobPool pool(queue_size);

    struct HuffmanTableGuard {
        qpl_huffman_table_t table = nullptr;
        ~HuffmanTableGuard() {
            if (table != nullptr) {
                (void)qpl_huffman_table_destroy(table);
            }
        }
    } huffman_guard;

    allocator_t default_allocator_c = {malloc, free};
    qpl_status status = qpl_deflate_huffman_table_create(
        decompression_table_type, qpl_path_hardware, default_allocator_c,
        &huffman_guard.table);
    if (status != QPL_STS_OK) {
        std::ostringstream oss;
        oss << "IAA decompression Huffman table creation failed with status "
            << status;
        error = oss.str();
        return false;
    }

    struct DictionaryGuard {
        std::unique_ptr<uint8_t[]> buffer;
        qpl_dictionary* dictionary = nullptr;
    } dictionary_guard;

    if (!compressed.dictionary_raw.empty()) {
        const sw_compression_level sw_compr_level = SW_NONE;
        const hw_compression_level hw_compr_level = HW_LEVEL_1;
        const std::size_t dictionary_buffer_size = qpl_get_dictionary_size(
            sw_compr_level, hw_compr_level, compressed.dictionary_raw.size());
        if (dictionary_buffer_size == 0) {
            error = "IAA dictionary buffer size is zero during decompression.";
            return false;
        }

        dictionary_guard.buffer = std::make_unique<uint8_t[]>(dictionary_buffer_size);
        dictionary_guard.dictionary =
            reinterpret_cast<qpl_dictionary*>(dictionary_guard.buffer.get());
        qpl_status status = qpl_build_dictionary(dictionary_guard.dictionary,
                                                 sw_compr_level,
                                                 hw_compr_level,
                                                 compressed.dictionary_raw.data(),
                                                 compressed.dictionary_raw.size());
        if (status != QPL_STS_OK) {
            std::ostringstream oss;
            oss << "IAA dictionary build failed during decompression with status "
                << status;
            error = oss.str();
            return false;
        }
    }

    std::vector<std::size_t> output_offsets(compressed.chunks.size(), 0);
    for (std::size_t i = 1; i < output_offsets.size(); ++i) {
        output_offsets[i] =
            output_offsets[i - 1] + compressed.original_chunk_sizes[i - 1];
    }

    auto submit_decompress_job = [&](std::size_t chunk_index) -> bool {
        const std::size_t slot = chunk_index % pool.size();
        qpl_job* job = pool.get(slot);
        const auto& chunk = compressed.chunks[chunk_index];
        const uint32_t expected = compressed.original_chunk_sizes[chunk_index];

        job->op = qpl_op_decompress;
        job->next_in_ptr = const_cast<uint8_t*>(chunk.data());
        job->available_in = static_cast<uint32_t>(chunk.size());
        job->next_out_ptr = output.data() + output_offsets[chunk_index];
        job->available_out = expected;
        job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;
        job->dictionary = dictionary_guard.dictionary;
        job->huffman_table = huffman_guard.table;

        qpl_status status = qpl_submit_job(job);
        if (status != QPL_STS_OK) {
            std::ostringstream oss;
            oss << "IAA decompression submit failed at chunk " << chunk_index
                << " with status " << status;
            error = oss.str();
            return false;
        }
        return true;
    };

    auto wait_decompress_job = [&](std::size_t chunk_index) -> bool {
        const std::size_t slot = chunk_index % pool.size();
        qpl_job* job = pool.get(slot);
        const uint32_t expected = compressed.original_chunk_sizes[chunk_index];

        qpl_status status = qpl_wait_job(job);
        if (status != QPL_STS_OK ||
            static_cast<uint32_t>(job->total_out) != expected) {
            std::ostringstream oss;
            oss << "IAA decompression wait failed at chunk " << chunk_index
                << " with status " << status
                << " (expected " << expected
                << ", got " << static_cast<uint32_t>(job->total_out) << ")";
            error = oss.str();
            return false;
        }
        return true;
    };

    const auto t0 = std::chrono::high_resolution_clock::now();

    std::size_t next_submit = 0;
    std::size_t next_wait = 0;

    const std::size_t initial_submit =
        std::min<std::size_t>(pool.size(), compressed.chunks.size());
    for (; next_submit < initial_submit; ++next_submit) {
        if (!submit_decompress_job(next_submit)) {
            return false;
        }
    }

    while (next_wait < compressed.chunks.size()) {
        const std::size_t wait_count = std::min<std::size_t>(
            kWaitBatch, compressed.chunks.size() - next_wait);
        for (std::size_t i = 0; i < wait_count; ++i) {
            if (!wait_decompress_job(next_wait + i)) {
                return false;
            }
        }
        next_wait += wait_count;

        const std::size_t submit_count = std::min<std::size_t>(
            wait_count, compressed.chunks.size() - next_submit);
        for (std::size_t i = 0; i < submit_count; ++i) {
            if (!submit_decompress_job(next_submit + i)) {
                return false;
            }
        }
        next_submit += submit_count;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    latency_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return true;
}

BenchmarkResult run_lz4(const std::string& dataset,
                        const std::vector<uint8_t>& data,
                        const RawBlocks& raw_blocks,
                        std::size_t block_size) {
    BenchmarkResult result;
    result.dataset = dataset;
    result.algorithm = "LZ4";
    result.block_size = block_size;
    result.original_bytes = data.size();

    try {
        ChunkedCompressed compressed;
        compressed = compress_lz4(raw_blocks, result.compression_cycles);
        result.compressed_bytes = compressed.compressed_bytes;
        result.compression_ratio =
            static_cast<double>(result.compressed_bytes) /
            static_cast<double>(result.original_bytes);

        std::vector<uint8_t> decompressed;
        std::string err;
        if (!decompress_lz4(compressed, decompressed,
                            result.decompression_latency_ns, err)) {
            result.error = err;
            return result;
        }
        result.verified = (decompressed == data);
        if (!result.verified) {
            result.error = "Decompressed data mismatch.";
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }

    return result;
}

BenchmarkResult run_deflate(const std::string& dataset,
                            const std::vector<uint8_t>& data,
                            const RawBlocks& raw_blocks,
                            std::size_t block_size) {
    BenchmarkResult result;
    result.dataset = dataset;
    result.algorithm = "Deflate";
    result.block_size = block_size;
    result.original_bytes = data.size();

    try {
        ChunkedCompressed compressed;
        compressed = compress_deflate(raw_blocks, result.compression_cycles);
        result.compressed_bytes = compressed.compressed_bytes;
        result.compression_ratio =
            static_cast<double>(result.compressed_bytes) /
            static_cast<double>(result.original_bytes);

        std::vector<uint8_t> decompressed;
        std::string err;
        if (!decompress_deflate(compressed, decompressed,
                                result.decompression_latency_ns, err)) {
            result.error = err;
            return result;
        }
        result.verified = (decompressed == data);
        if (!result.verified) {
            result.error = "Decompressed data mismatch.";
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }

    return result;
}

BenchmarkResult run_iaa(const std::string& dataset,
                        const std::vector<uint8_t>& data,
                        const RawBlocks& requested_blocks,
                        std::size_t block_size) {
    BenchmarkResult result;
    result.dataset = dataset;
    result.algorithm = "IAA(HW)";
    result.block_size = block_size;
    result.original_bytes = data.size();

    try {
        ChunkedCompressed compressed;
        compressed = compress_iaa(requested_blocks, result.compression_cycles);

        result.compressed_bytes = compressed.compressed_bytes;
        result.compression_ratio =
            static_cast<double>(result.compressed_bytes) /
            static_cast<double>(result.original_bytes);

        std::vector<uint8_t> decompressed;
        std::string err;
        if (!decompress_iaa(compressed, decompressed,
                            result.decompression_latency_ns, err)) {
            result.error = err;
            return result;
        }
        result.verified = (decompressed == data);
        if (!result.verified) {
            result.error = "Decompressed data mismatch.";
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }

    return result;
}

std::string block_label(std::size_t block_size) {
    if (block_size == kBlockSize8KB) {
        return "8KB";
    }
    if (block_size == kBlockSize16KB) {
        return "16KB";
    }
    if (block_size == kBlockSize64KB) {
        return "64KB";
    }
    if (block_size == kBlockSize128KB) {
        return "128KB";
    }
    if (block_size == kBlockSize256KB) {
        return "256KB";
    }
    if (block_size == kBlockSize512KB) {
        return "512KB";
    }
    if (block_size == kBlockSize1MB) {
        return "1MB";
    }
    if (block_size == kBlockSize2MB) {
        return "2MB";
    }
    return std::to_string(block_size) + "B";
}

void print_dataset_header(const std::string& dataset,
                          std::size_t rows,
                          const std::vector<std::string>& columns,
                          std::size_t bytes) {
    std::cout << "\n===============================================================\n";
    std::cout << "Dataset: " << dataset << "\n";
    std::cout << "Rows used: " << rows << "\n";
    std::cout << "Selected attributes (" << columns.size() << "): ";
    for (std::size_t i = 0; i < columns.size(); ++i) {
        std::cout << columns[i];
        if (i + 1 < columns.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";
    std::cout << "Raw uint8 bytes: " << bytes << "\n";
    std::cout << "===============================================================\n";
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(10) << r.algorithm
              << std::setw(8) << block_label(r.block_size)
              << std::right << std::setw(12) << r.original_bytes
              << std::setw(14) << r.compressed_bytes
              << std::setw(12) << std::fixed << std::setprecision(6)
              << r.compression_ratio
              << std::setw(18) << r.compression_cycles
              << std::setw(18) << r.decompression_latency_ns
              << std::setw(10) << (r.verified ? "OK" : "FAIL")
              << "\n";
    if (!r.error.empty()) {
        std::cout << "  error: " << r.error << "\n";
    }
}

std::string jsonl_path_for_dataset(const std::string& json_dir,
                                   const std::string& dataset) {
    return json_dir + "/" + dataset + "_payloads.jsonl";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg;
        if (!parse_args(argc, argv, cfg)) {
            return 0;
        }

        auto included = load_included_columns(cfg.metadata_profile);

        std::cout << "JSON directory: " << cfg.json_dir << "\n";
        std::cout << "Metadata profile: " << cfg.metadata_profile << "\n";
        std::cout << "Max rows: "
                  << (cfg.max_rows == 0 ? std::string("all")
                                        : std::to_string(cfg.max_rows))
                  << "\n";

        for (const auto& dataset : cfg.datasets) {
            auto it = included.find(dataset);
            if (it == included.end() || it->second.empty()) {
                throw std::runtime_error(
                    "No included attribute list for dataset '" + dataset +
                    "' in metadata profile.");
            }

            const std::string jsonl_path =
                jsonl_path_for_dataset(cfg.json_dir, dataset);
            DatasetBytes data =
                build_dataset_bytes(jsonl_path, it->second, cfg.max_rows);

            print_dataset_header(dataset, data.rows, data.columns,
                                 data.bytes.size());
            std::cout << std::left << std::setw(10) << "Algorithm"
                      << std::setw(8) << "Block"
                      << std::right << std::setw(12) << "Original"
                      << std::setw(14) << "Compressed"
                      << std::setw(12) << "CR"
                      << std::setw(18) << "CompCycles"
                      << std::setw(18) << "DecompLat(ns)"
                      << std::setw(10) << "Verify"
                      << "\n";

            const std::vector<std::size_t> block_sizes = {
                kBlockSize8KB,
                kBlockSize16KB,
                kBlockSize64KB,
                kBlockSize128KB,
                kBlockSize256KB,
                kBlockSize512KB,
                kBlockSize1MB,
                kBlockSize2MB,
            };
            for (std::size_t block : block_sizes) {
                RawBlocks raw_blocks = split_into_raw_blocks(data.bytes, block);
                print_result(run_lz4(dataset, data.bytes, raw_blocks, block));
                print_result(run_deflate(dataset, data.bytes, raw_blocks, block));
                print_result(run_iaa(dataset, data.bytes, raw_blocks, block));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
