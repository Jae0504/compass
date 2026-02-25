#pragma once

#include "../build_fid_tb/json.hpp"
#include "../../cpp_examples/filter_search_examples/filter_expr.h"

#include <qpl/qpl.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace compass_iaa_filter {

using json = nlohmann::json;

constexpr size_t kDefaultFidBlockSizeBytes = 8192;
constexpr size_t kDefaultTbBlockSizeBytes = 8192;
constexpr size_t kMaxBuckets = 256;
constexpr size_t kTbBytesPerNode = 32;

struct QueryDecompressionMetrics {
    uint64_t iaa_decompress_time_ns = 0;

    uint64_t fid_blocks_decompressed = 0;
    uint64_t tb_blocks_decompressed = 0;

    uint64_t fid_cache_hits = 0;
    uint64_t tb_cache_hits = 0;

    uint64_t fid_bytes_decompressed = 0;
    uint64_t tb_bytes_decompressed = 0;
};

struct QueryBlockCache {
    std::vector<std::unordered_map<size_t, std::vector<uint8_t>>> fid_blocks;
    std::vector<std::unordered_map<size_t, std::vector<uint8_t>>> tb_blocks;

    explicit QueryBlockCache(size_t attr_count = 0)
        : fid_blocks(attr_count), tb_blocks(attr_count) {}

    void reset(size_t attr_count) {
        fid_blocks.assign(attr_count, {});
        tb_blocks.assign(attr_count, {});
    }
};

struct ManifestAttribute {
    std::string key;
    std::string encoding;
    bool numeric = false;

    int used_bins = 0;
    double min_value = 0.0;
    double max_value = 0.0;

    std::unordered_map<std::string, int> category_map;

    std::string fid_file;
    std::string tb_file;
};

struct ManifestData {
    std::string manifest_path;
    std::string dataset_type;
    std::string benchmark;

    size_t n_elements = 0;
    int nfilters = 256;

    std::vector<ManifestAttribute> attributes;
};

namespace detail {

inline bool compare_numeric(double lhs, double rhs, filter_expr::CompareOp op) {
    switch (op) {
        case filter_expr::CompareOp::Eq:
            return lhs == rhs;
        case filter_expr::CompareOp::Ne:
            return lhs != rhs;
        case filter_expr::CompareOp::Gt:
            return lhs > rhs;
        case filter_expr::CompareOp::Ge:
            return lhs >= rhs;
        case filter_expr::CompareOp::Lt:
            return lhs < rhs;
        case filter_expr::CompareOp::Le:
            return lhs <= rhs;
    }
    return false;
}

inline bool compare_string(const std::string& lhs, const std::string& rhs, filter_expr::CompareOp op) {
    switch (op) {
        case filter_expr::CompareOp::Eq:
            return lhs == rhs;
        case filter_expr::CompareOp::Ne:
            return lhs != rhs;
        case filter_expr::CompareOp::Gt:
            return lhs > rhs;
        case filter_expr::CompareOp::Ge:
            return lhs >= rhs;
        case filter_expr::CompareOp::Lt:
            return lhs < rhs;
        case filter_expr::CompareOp::Le:
            return lhs <= rhs;
    }
    return false;
}

inline bool parse_numeric_literal(const filter_expr::Literal& lit, double* out) {
    if (lit.is_number) {
        if (out != nullptr) {
            *out = lit.number;
        }
        return true;
    }
    return filter_expr::detail::try_parse_double(lit.text, out);
}

inline std::vector<uint8_t> read_payload_with_size_header(
    const std::string& path,
    size_t* item_count,
    size_t element_size_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload: " + path);
    }

    size_t count = 0;
    if (!in.read(reinterpret_cast<char*>(&count), sizeof(size_t))) {
        throw std::runtime_error("Failed to read size header from payload: " + path);
    }

    if (item_count != nullptr) {
        *item_count = count;
    }

    const size_t total_bytes = count * element_size_bytes;
    std::vector<uint8_t> bytes(total_bytes, 0);
    if (total_bytes > 0 && !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(total_bytes))) {
        throw std::runtime_error("Failed to read payload body from: " + path);
    }

    return bytes;
}

inline std::array<uint8_t, kTbBytesPerNode> to_mask_bytes(const std::bitset<kMaxBuckets>& bits) {
    std::array<uint8_t, kTbBytesPerNode> out{};
    for (size_t bit = 0; bit < kMaxBuckets; ++bit) {
        if (bits.test(bit)) {
            out[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
        }
    }
    return out;
}

struct IaaBlockStorage {
    size_t block_size = kDefaultFidBlockSizeBytes;
    size_t raw_size = 0;
    std::vector<uint32_t> raw_block_sizes;
    std::vector<std::vector<uint8_t>> compressed_blocks;

    size_t block_count() const {
        return compressed_blocks.size();
    }
};

inline std::string qpl_status_string(qpl_status status) {
    return "qpl_status=" + std::to_string(static_cast<int>(status));
}

inline void require_qpl_ok(qpl_status status, const std::string& context) {
    if (status != QPL_STS_OK) {
        throw std::runtime_error(context + " (" + qpl_status_string(status) + ")");
    }
}

class QplJobHandle {
public:
    explicit QplJobHandle(qpl_path_t path, const std::string& purpose) {
        uint32_t size = 0;
        qpl_status status = qpl_get_job_size(path, &size);
        require_qpl_ok(status, purpose + ": qpl_get_job_size failed");

        buffer_ = std::make_unique<uint8_t[]>(size);
        job_ = reinterpret_cast<qpl_job*>(buffer_.get());

        status = qpl_init_job(path, job_);
        require_qpl_ok(status, purpose + ": qpl_init_job failed");
    }

    ~QplJobHandle() {
        if (job_ != nullptr) {
            (void)qpl_fini_job(job_);
        }
    }

    QplJobHandle(const QplJobHandle&) = delete;
    QplJobHandle& operator=(const QplJobHandle&) = delete;
    QplJobHandle(QplJobHandle&&) = delete;
    QplJobHandle& operator=(QplJobHandle&&) = delete;

    qpl_job* job() const {
        return job_;
    }

private:
    std::unique_ptr<uint8_t[]> buffer_;
    qpl_job* job_ = nullptr;
};

inline std::unique_ptr<QplJobHandle> make_hardware_job(const std::string& purpose) {
    return std::make_unique<QplJobHandle>(qpl_path_hardware, purpose);
}

inline IaaBlockStorage compress_to_iaa_blocks(
    const std::vector<uint8_t>& raw,
    size_t block_size,
    QplJobHandle* compressor) {
    if (block_size == 0) {
        throw std::runtime_error("Block size must be positive");
    }
    if (compressor == nullptr || compressor->job() == nullptr) {
        throw std::runtime_error("QPL compressor job is not initialized");
    }

    IaaBlockStorage out;
    out.block_size = block_size;
    out.raw_size = raw.size();

    if (raw.empty()) {
        return out;
    }

    const size_t nblocks = (raw.size() + block_size - 1) / block_size;
    out.raw_block_sizes.resize(nblocks, 0);
    out.compressed_blocks.resize(nblocks);

    for (size_t bid = 0; bid < nblocks; ++bid) {
        const size_t start = bid * block_size;
        const size_t raw_len = std::min(block_size, raw.size() - start);
        if (raw_len > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("Block size exceeds uint32_t range");
        }

        const uint32_t raw_len_u32 = static_cast<uint32_t>(raw_len);
        const uint32_t bound = qpl_get_safe_deflate_compression_buffer_size(raw_len_u32);
        if (bound == 0) {
            throw std::runtime_error(
                "QPL safe compression buffer size is zero at block " + std::to_string(bid));
        }

        out.compressed_blocks[bid].assign(static_cast<size_t>(bound), 0);

        qpl_job* job = compressor->job();
        job->op = qpl_op_compress;
        job->level = qpl_default_level;
        job->next_in_ptr = const_cast<uint8_t*>(raw.data() + start);
        job->next_out_ptr = out.compressed_blocks[bid].data();
        job->available_in = raw_len_u32;
        job->available_out = bound;
        job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;

        const qpl_status status = qpl_execute_job(job);
        require_qpl_ok(status, "QPL compression failed at block " + std::to_string(bid));

        if (job->total_out == 0 || job->total_out > bound) {
            throw std::runtime_error("QPL compression produced invalid output size at block " + std::to_string(bid));
        }

        out.compressed_blocks[bid].resize(static_cast<size_t>(job->total_out));
        out.raw_block_sizes[bid] = raw_len_u32;
    }

    return out;
}

inline const std::vector<uint8_t>& get_or_decompress_block(
    const IaaBlockStorage& storage,
    size_t block_id,
    bool is_fid,
    std::unordered_map<size_t, std::vector<uint8_t>>* cache,
    QueryDecompressionMetrics* metrics,
    QplJobHandle* decompressor) {
    if (block_id >= storage.block_count()) {
        throw std::runtime_error("Block id out of range during decompression");
    }
    if (cache == nullptr) {
        throw std::runtime_error("Block cache must not be null");
    }
    if (decompressor == nullptr || decompressor->job() == nullptr) {
        throw std::runtime_error("QPL decompressor job is not initialized");
    }

    auto it = cache->find(block_id);
    if (it != cache->end()) {
        if (metrics != nullptr) {
            if (is_fid) {
                ++metrics->fid_cache_hits;
            } else {
                ++metrics->tb_cache_hits;
            }
        }
        return it->second;
    }

    const size_t raw_len = static_cast<size_t>(storage.raw_block_sizes[block_id]);
    std::vector<uint8_t> out(raw_len, 0);

    qpl_job* job = decompressor->job();
    job->op = qpl_op_decompress;
    job->next_in_ptr = const_cast<uint8_t*>(storage.compressed_blocks[block_id].data());
    job->next_out_ptr = out.data();
    job->available_in = static_cast<uint32_t>(storage.compressed_blocks[block_id].size());
    job->available_out = static_cast<uint32_t>(raw_len);
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;

    const auto t0 = std::chrono::steady_clock::now();
    const qpl_status status = qpl_execute_job(job);
    const auto t1 = std::chrono::steady_clock::now();
    require_qpl_ok(status, "QPL decompression failed at block " + std::to_string(block_id));

    if (job->total_out != raw_len) {
        throw std::runtime_error("QPL decompression size mismatch at block " + std::to_string(block_id));
    }

    if (metrics != nullptr) {
        metrics->iaa_decompress_time_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        if (is_fid) {
            ++metrics->fid_blocks_decompressed;
            metrics->fid_bytes_decompressed += static_cast<uint64_t>(raw_len);
        } else {
            ++metrics->tb_blocks_decompressed;
            metrics->tb_bytes_decompressed += static_cast<uint64_t>(raw_len);
        }
    }

    auto inserted = cache->emplace(block_id, std::move(out));
    return inserted.first->second;
}

inline uint8_t read_raw_byte(
    const IaaBlockStorage& storage,
    size_t offset,
    bool is_fid,
    std::unordered_map<size_t, std::vector<uint8_t>>* cache,
    QueryDecompressionMetrics* metrics,
    QplJobHandle* decompressor) {
    if (offset >= storage.raw_size) {
        throw std::runtime_error("Byte offset out of range in compressed storage");
    }

    const size_t block_id = offset / storage.block_size;
    const size_t in_block = offset % storage.block_size;
    const std::vector<uint8_t>& block = get_or_decompress_block(
        storage,
        block_id,
        is_fid,
        cache,
        metrics,
        decompressor);

    if (in_block >= block.size()) {
        throw std::runtime_error("In-block offset out of range after decompression");
    }
    return block[in_block];
}

}  // namespace detail

class CompassIaaFilterEngine {
public:
    static CompassIaaFilterEngine Build(
        const std::string& manifest_path,
        const std::string& filter_expression,
        const std::unordered_set<std::string>& referenced_fields,
        size_t fid_block_size = kDefaultFidBlockSizeBytes,
        size_t tb_block_size = kDefaultTbBlockSizeBytes) {
        if (manifest_path.empty()) {
            throw std::runtime_error("--fidtb-manifest is required");
        }
        if (filter_expression.empty()) {
            throw std::runtime_error("Filter expression is empty");
        }
        if (referenced_fields.empty()) {
            throw std::runtime_error("Filter expression did not reference any fields");
        }
        if (fid_block_size == 0 || tb_block_size == 0) {
            throw std::runtime_error("FID/TB block sizes must be positive");
        }

        CompassIaaFilterEngine engine;
        engine.fid_block_size_ = fid_block_size;
        engine.tb_block_size_ = tb_block_size;
        engine.filter_expression_ = filter_expression;
        engine.manifest_ = load_manifest(manifest_path);
        engine.decompressor_job_ = detail::make_hardware_job("IAA filter engine decompressor");
        std::unique_ptr<detail::QplJobHandle> compressor_job =
            detail::make_hardware_job("IAA filter engine compressor");

        if (engine.manifest_.nfilters <= 0 || engine.manifest_.nfilters > static_cast<int>(kMaxBuckets)) {
            throw std::runtime_error("Manifest nfilters must be in [1, 256]");
        }
        if (engine.manifest_.n_elements == 0) {
            throw std::runtime_error("Manifest n_elements must be positive");
        }

        std::unordered_map<std::string, ManifestAttribute> attr_by_key;
        for (const ManifestAttribute& attr : engine.manifest_.attributes) {
            attr_by_key[attr.key] = attr;
        }

        engine.attributes_.reserve(referenced_fields.size());
        for (const std::string& field : referenced_fields) {
            auto it = attr_by_key.find(field);
            if (it == attr_by_key.end()) {
                throw std::runtime_error("Referenced field not found in manifest: " + field);
            }
            engine.attributes_.push_back(load_attribute(
                engine.manifest_,
                it->second,
                engine.fid_block_size_,
                engine.tb_block_size_,
                compressor_job.get()));
            engine.key_to_attr_index_[field] = engine.attributes_.size() - 1;
        }

        std::vector<filter_expr::detail::Token> tokens = filter_expr::detail::tokenize(engine.filter_expression_);
        filter_expr::detail::Parser parser(std::move(tokens));
        std::unique_ptr<filter_expr::Node> root = parser.parse();
        engine.compiled_root_ = engine.compile_node(root.get());

        return engine;
    }

    bool allow_result(
        size_t node_id,
        QueryBlockCache* cache,
        QueryDecompressionMetrics* metrics) const {
        if (node_id >= manifest_.n_elements) {
            return false;
        }
        if (compiled_root_ == nullptr) {
            return false;
        }
        return evaluate_node(compiled_root_.get(), node_id, false, cache, metrics);
    }

    bool allow_traversal(
        size_t node_id,
        QueryBlockCache* cache,
        QueryDecompressionMetrics* metrics) const {
        if (node_id >= manifest_.n_elements) {
            return false;
        }
        if (compiled_root_ == nullptr) {
            return false;
        }
        return evaluate_node(compiled_root_.get(), node_id, true, cache, metrics);
    }

    std::vector<size_t> collect_result_candidates() const {
        QueryBlockCache cache(attributes_.size());
        QueryDecompressionMetrics metrics;

        std::vector<size_t> out;
        out.reserve(manifest_.n_elements);
        for (size_t node_id = 0; node_id < manifest_.n_elements; ++node_id) {
            if (allow_result(node_id, &cache, &metrics)) {
                out.push_back(node_id);
            }
        }
        return out;
    }

    const ManifestData& manifest() const {
        return manifest_;
    }

    size_t num_elements() const {
        return manifest_.n_elements;
    }

    size_t attribute_count() const {
        return attributes_.size();
    }

    const std::string& filter_expression() const {
        return filter_expression_;
    }

private:
    struct AttributeStorage {
        std::string key;
        std::string encoding;
        bool numeric = false;

        int nfilters = 256;
        int usable_bins = 255;

        double min_value = 0.0;
        double max_value = 0.0;

        std::unordered_map<std::string, int> category_map;

        detail::IaaBlockStorage fid_blocks;
        detail::IaaBlockStorage tb_blocks;
    };

    struct CompiledLeaf {
        size_t attr_index = 0;
        std::bitset<kMaxBuckets> allowed_buckets;
        std::array<uint8_t, kTbBytesPerNode> allowed_bucket_mask{};
    };

    struct CompiledNode {
        enum class Kind {
            Logical,
            Leaf
        };

        Kind kind = Kind::Leaf;
        filter_expr::LogicalOp logical_op = filter_expr::LogicalOp::And;

        CompiledLeaf leaf;
        std::unique_ptr<CompiledNode> left;
        std::unique_ptr<CompiledNode> right;
    };

    static ManifestData load_manifest(const std::string& manifest_path) {
        std::ifstream in(manifest_path);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open manifest: " + manifest_path);
        }

        json j;
        try {
            in >> j;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse manifest JSON: " + std::string(e.what()));
        }

        ManifestData out;
        out.manifest_path = manifest_path;
        out.dataset_type = j.value("dataset_type", "");
        out.benchmark = j.value("benchmark", "");
        out.n_elements = j.value("n_elements", static_cast<size_t>(0));
        out.nfilters = j.value("nfilters", 256);

        if (!j.contains("attributes") || !j["attributes"].is_array()) {
            throw std::runtime_error("Manifest is missing attributes array");
        }

        const std::filesystem::path manifest_dir = std::filesystem::absolute(std::filesystem::path(manifest_path)).parent_path();

        for (const auto& item : j["attributes"]) {
            if (!item.is_object()) {
                continue;
            }

            ManifestAttribute attr;
            attr.key = item.value("key", "");
            attr.encoding = item.value("encoding", "");
            attr.numeric = item.value("numeric", false);
            attr.used_bins = item.value("used_bins", 0);
            attr.min_value = item.value("min_value", 0.0);
            attr.max_value = item.value("max_value", 0.0);
            attr.fid_file = item.value("fid_file", "");
            attr.tb_file = item.value("tb_file", "");

            if (attr.key.empty()) {
                throw std::runtime_error("Manifest attribute is missing key");
            }
            if (attr.fid_file.empty() || attr.tb_file.empty()) {
                throw std::runtime_error("Manifest attribute is missing fid_file or tb_file: " + attr.key);
            }

            if (item.contains("category_map") && item["category_map"].is_object()) {
                for (auto it = item["category_map"].begin(); it != item["category_map"].end(); ++it) {
                    if (!it.value().is_number_integer()) {
                        continue;
                    }
                    attr.category_map[it.key()] = it.value().get<int>();
                }
            }

            std::filesystem::path fid_path(attr.fid_file);
            std::filesystem::path tb_path(attr.tb_file);
            if (fid_path.is_relative()) {
                fid_path = manifest_dir / fid_path;
            }
            if (tb_path.is_relative()) {
                tb_path = manifest_dir / tb_path;
            }
            attr.fid_file = fid_path.lexically_normal().string();
            attr.tb_file = tb_path.lexically_normal().string();

            out.attributes.push_back(std::move(attr));
        }

        if (out.attributes.empty()) {
            throw std::runtime_error("Manifest contains no attributes");
        }

        return out;
    }

    static AttributeStorage load_attribute(
        const ManifestData& manifest,
        const ManifestAttribute& attr,
        size_t fid_block_size,
        size_t tb_block_size,
        detail::QplJobHandle* compressor_job) {
        AttributeStorage out;
        out.key = attr.key;
        out.encoding = attr.encoding;
        out.numeric = attr.numeric;
        out.nfilters = manifest.nfilters;
        out.usable_bins = attr.used_bins;
        out.min_value = attr.min_value;
        out.max_value = attr.max_value;
        out.category_map = attr.category_map;

        if (out.usable_bins <= 0) {
            out.usable_bins = std::max(1, out.nfilters - 1);
        }
        out.usable_bins = std::min(out.usable_bins, std::max(1, out.nfilters - 1));

        if (out.numeric) {
            if (out.encoding != "numeric_minmax_quantized") {
                throw std::runtime_error(
                    "Numeric attribute has unsupported encoding for strict FID/TB translation: " +
                    attr.key + " (" + out.encoding + ")");
            }
        } else {
            if (out.category_map.empty()) {
                throw std::runtime_error(
                    "Categorical attribute is missing category_map in manifest: " + attr.key);
            }
        }

        size_t fid_count = 0;
        std::vector<uint8_t> fid_raw = detail::read_payload_with_size_header(attr.fid_file, &fid_count, sizeof(uint8_t));
        if (fid_count != manifest.n_elements) {
            throw std::runtime_error(
                "FID element count mismatch for attribute " + attr.key +
                ": manifest=" + std::to_string(manifest.n_elements) +
                ", fid=" + std::to_string(fid_count));
        }

        size_t tb_count = 0;
        std::vector<uint8_t> tb_raw = detail::read_payload_with_size_header(attr.tb_file, &tb_count, kTbBytesPerNode);
        if (tb_count != manifest.n_elements) {
            throw std::runtime_error(
                "TB element count mismatch for attribute " + attr.key +
                ": manifest=" + std::to_string(manifest.n_elements) +
                ", tb=" + std::to_string(tb_count));
        }

        out.fid_blocks = detail::compress_to_iaa_blocks(fid_raw, fid_block_size, compressor_job);
        out.tb_blocks = detail::compress_to_iaa_blocks(tb_raw, tb_block_size, compressor_job);
        return out;
    }

    int bucket_from_numeric(const AttributeStorage& attr, double value) const {
        if (attr.usable_bins <= 1 || attr.max_value <= attr.min_value) {
            return 0;
        }

        const double scale = static_cast<double>(attr.usable_bins - 1) / (attr.max_value - attr.min_value);
        int bucket = static_cast<int>(std::floor((value - attr.min_value) * scale));
        if (bucket < 0) {
            bucket = 0;
        }
        if (bucket >= attr.usable_bins) {
            bucket = attr.usable_bins - 1;
        }
        return bucket;
    }

    double bucket_representative(const AttributeStorage& attr, int bucket) const {
        if (attr.usable_bins <= 1 || attr.max_value <= attr.min_value) {
            return attr.min_value;
        }
        const double ratio = static_cast<double>(bucket) / static_cast<double>(attr.usable_bins - 1);
        return attr.min_value + ratio * (attr.max_value - attr.min_value);
    }

    std::bitset<kMaxBuckets> all_data_buckets(const AttributeStorage& attr) const {
        std::bitset<kMaxBuckets> out;
        const int limit = std::min(attr.usable_bins, static_cast<int>(kMaxBuckets));
        for (int b = 0; b < limit; ++b) {
            out.set(static_cast<size_t>(b));
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_compare_numeric(
        const AttributeStorage& attr,
        filter_expr::CompareOp op,
        const filter_expr::Literal& lit) const {
        double rhs = 0.0;
        if (!detail::parse_numeric_literal(lit, &rhs)) {
            throw std::runtime_error(
                "Numeric predicate has non-numeric literal for field '" + attr.key + "': " + lit.text);
        }

        std::bitset<kMaxBuckets> out;
        if (op == filter_expr::CompareOp::Eq) {
            out.set(static_cast<size_t>(bucket_from_numeric(attr, rhs)));
            return out;
        }
        if (op == filter_expr::CompareOp::Ne) {
            out = all_data_buckets(attr);
            out.reset(static_cast<size_t>(bucket_from_numeric(attr, rhs)));
            return out;
        }

        const int limit = std::min(attr.usable_bins, static_cast<int>(kMaxBuckets));
        for (int b = 0; b < limit; ++b) {
            const double rep = bucket_representative(attr, b);
            if (detail::compare_numeric(rep, rhs, op)) {
                out.set(static_cast<size_t>(b));
            }
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_between_numeric(
        const AttributeStorage& attr,
        const filter_expr::Literal& lo,
        const filter_expr::Literal& hi) const {
        double lo_v = 0.0;
        double hi_v = 0.0;
        if (!detail::parse_numeric_literal(lo, &lo_v) || !detail::parse_numeric_literal(hi, &hi_v)) {
            throw std::runtime_error("Numeric BETWEEN has non-numeric bounds for field '" + attr.key + "'");
        }

        if (lo_v > hi_v) {
            return std::bitset<kMaxBuckets>();
        }

        std::bitset<kMaxBuckets> out;
        const int limit = std::min(attr.usable_bins, static_cast<int>(kMaxBuckets));
        for (int b = 0; b < limit; ++b) {
            const double rep = bucket_representative(attr, b);
            if (rep >= lo_v && rep <= hi_v) {
                out.set(static_cast<size_t>(b));
            }
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_in_numeric(
        const AttributeStorage& attr,
        const std::vector<filter_expr::Literal>& list) const {
        std::bitset<kMaxBuckets> out;
        for (const filter_expr::Literal& lit : list) {
            double v = 0.0;
            if (!detail::parse_numeric_literal(lit, &v)) {
                throw std::runtime_error(
                    "Numeric IN has non-numeric literal for field '" + attr.key + "': " + lit.text);
            }
            out.set(static_cast<size_t>(bucket_from_numeric(attr, v)));
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_compare_categorical(
        const AttributeStorage& attr,
        filter_expr::CompareOp op,
        const filter_expr::Literal& lit) const {
        std::bitset<kMaxBuckets> out;

        auto bucket_it = attr.category_map.find(lit.text);
        if (op == filter_expr::CompareOp::Eq) {
            if (bucket_it != attr.category_map.end() && bucket_it->second >= 0 && bucket_it->second < attr.usable_bins) {
                out.set(static_cast<size_t>(bucket_it->second));
            }
            return out;
        }

        if (op == filter_expr::CompareOp::Ne) {
            out = all_data_buckets(attr);
            if (bucket_it != attr.category_map.end() && bucket_it->second >= 0 && bucket_it->second < attr.usable_bins) {
                out.reset(static_cast<size_t>(bucket_it->second));
            }
            return out;
        }

        for (const auto& kv : attr.category_map) {
            const std::string& key = kv.first;
            const int bucket = kv.second;
            if (bucket < 0 || bucket >= attr.usable_bins || bucket >= static_cast<int>(kMaxBuckets)) {
                continue;
            }
            if (detail::compare_string(key, lit.text, op)) {
                out.set(static_cast<size_t>(bucket));
            }
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_between_categorical(
        const AttributeStorage& attr,
        const filter_expr::Literal& lo,
        const filter_expr::Literal& hi) const {
        std::bitset<kMaxBuckets> out;
        if (lo.text > hi.text) {
            return out;
        }

        for (const auto& kv : attr.category_map) {
            const std::string& key = kv.first;
            const int bucket = kv.second;
            if (bucket < 0 || bucket >= attr.usable_bins || bucket >= static_cast<int>(kMaxBuckets)) {
                continue;
            }
            if (key >= lo.text && key <= hi.text) {
                out.set(static_cast<size_t>(bucket));
            }
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_in_categorical(
        const AttributeStorage& attr,
        const std::vector<filter_expr::Literal>& list) const {
        std::bitset<kMaxBuckets> out;
        for (const filter_expr::Literal& lit : list) {
            auto it = attr.category_map.find(lit.text);
            if (it == attr.category_map.end()) {
                continue;
            }
            if (it->second < 0 || it->second >= attr.usable_bins || it->second >= static_cast<int>(kMaxBuckets)) {
                continue;
            }
            out.set(static_cast<size_t>(it->second));
        }
        return out;
    }

    std::bitset<kMaxBuckets> compile_leaf_buckets(
        const AttributeStorage& attr,
        const filter_expr::Node* node) const {
        if (node == nullptr) {
            throw std::runtime_error("Internal error: null expression node");
        }

        if (node->kind == filter_expr::Node::Kind::Compare) {
            if (attr.numeric) {
                return compile_compare_numeric(attr, node->compare_op, node->literal);
            }
            return compile_compare_categorical(attr, node->compare_op, node->literal);
        }

        if (node->kind == filter_expr::Node::Kind::Between) {
            if (attr.numeric) {
                return compile_between_numeric(attr, node->lower, node->upper);
            }
            return compile_between_categorical(attr, node->lower, node->upper);
        }

        if (node->kind == filter_expr::Node::Kind::In) {
            if (attr.numeric) {
                return compile_in_numeric(attr, node->list);
            }
            return compile_in_categorical(attr, node->list);
        }

        throw std::runtime_error("Unexpected logical node passed to compile_leaf_buckets");
    }

    std::unique_ptr<CompiledNode> compile_node(const filter_expr::Node* node) const {
        if (node == nullptr) {
            throw std::runtime_error("Parsed filter expression node is null");
        }

        auto out = std::make_unique<CompiledNode>();
        if (node->kind == filter_expr::Node::Kind::Logical) {
            out->kind = CompiledNode::Kind::Logical;
            out->logical_op = node->logical_op;
            out->left = compile_node(node->left.get());
            out->right = compile_node(node->right.get());
            return out;
        }

        auto it = key_to_attr_index_.find(node->field);
        if (it == key_to_attr_index_.end()) {
            throw std::runtime_error("Field was not loaded from manifest: " + node->field);
        }

        out->kind = CompiledNode::Kind::Leaf;
        out->leaf.attr_index = it->second;
        out->leaf.allowed_buckets = compile_leaf_buckets(attributes_[it->second], node);
        out->leaf.allowed_bucket_mask = detail::to_mask_bytes(out->leaf.allowed_buckets);
        return out;
    }

    uint8_t read_fid_bucket(
        const AttributeStorage& attr,
        size_t attr_index,
        size_t node_id,
        QueryBlockCache* cache,
        QueryDecompressionMetrics* metrics) const {
        if (cache == nullptr) {
            throw std::runtime_error("QueryBlockCache must not be null");
        }
        if (attr_index >= cache->fid_blocks.size()) {
            throw std::runtime_error("FID cache index out of range");
        }

        const size_t offset = node_id;
        return detail::read_raw_byte(
            attr.fid_blocks,
            offset,
            true,
            &cache->fid_blocks[attr_index],
            metrics,
            decompressor_job_.get());
    }

    bool tb_matches_any_bucket(
        const AttributeStorage& attr,
        size_t attr_index,
        size_t node_id,
        const std::array<uint8_t, kTbBytesPerNode>& mask,
        QueryBlockCache* cache,
        QueryDecompressionMetrics* metrics) const {
        if (cache == nullptr) {
            throw std::runtime_error("QueryBlockCache must not be null");
        }
        if (attr_index >= cache->tb_blocks.size()) {
            throw std::runtime_error("TB cache index out of range");
        }

        const size_t base = node_id * kTbBytesPerNode;
        for (size_t i = 0; i < kTbBytesPerNode; ++i) {
            const uint8_t tb_byte = detail::read_raw_byte(
                attr.tb_blocks,
                base + i,
                false,
                &cache->tb_blocks[attr_index],
                metrics,
                decompressor_job_.get());
            if ((tb_byte & mask[i]) != 0) {
                return true;
            }
        }
        return false;
    }

    bool evaluate_leaf(
        const CompiledLeaf& leaf,
        size_t node_id,
        bool traversal_mode,
        QueryBlockCache* cache,
        QueryDecompressionMetrics* metrics) const {
        const AttributeStorage& attr = attributes_[leaf.attr_index];
        const uint8_t fid_bucket = read_fid_bucket(attr, leaf.attr_index, node_id, cache, metrics);

        const bool fid_match =
            (static_cast<size_t>(fid_bucket) < kMaxBuckets) &&
            leaf.allowed_buckets.test(static_cast<size_t>(fid_bucket));
        if (!traversal_mode) {
            return fid_match;
        }
        if (fid_match) {
            return true;
        }

        return tb_matches_any_bucket(attr, leaf.attr_index, node_id, leaf.allowed_bucket_mask, cache, metrics);
    }

    bool evaluate_node(
        const CompiledNode* node,
        size_t node_id,
        bool traversal_mode,
        QueryBlockCache* cache,
        QueryDecompressionMetrics* metrics) const {
        if (node == nullptr) {
            return false;
        }

        if (node->kind == CompiledNode::Kind::Logical) {
            if (node->logical_op == filter_expr::LogicalOp::And) {
                return evaluate_node(node->left.get(), node_id, traversal_mode, cache, metrics) &&
                       evaluate_node(node->right.get(), node_id, traversal_mode, cache, metrics);
            }
            return evaluate_node(node->left.get(), node_id, traversal_mode, cache, metrics) ||
                   evaluate_node(node->right.get(), node_id, traversal_mode, cache, metrics);
        }

        return evaluate_leaf(node->leaf, node_id, traversal_mode, cache, metrics);
    }

    ManifestData manifest_;
    std::vector<AttributeStorage> attributes_;
    std::unordered_map<std::string, size_t> key_to_attr_index_;

    std::string filter_expression_;
    size_t fid_block_size_ = kDefaultFidBlockSizeBytes;
    size_t tb_block_size_ = kDefaultTbBlockSizeBytes;
    std::unique_ptr<detail::QplJobHandle> decompressor_job_;

    std::unique_ptr<CompiledNode> compiled_root_;
};

}  // namespace compass_iaa_filter
