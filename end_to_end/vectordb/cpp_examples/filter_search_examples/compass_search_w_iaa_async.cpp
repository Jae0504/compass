#include "../../hnswlib/filter_search_hnswlib/hnswlib.h"
#include "../../hnswlib/filter_search_iaa/compass_iaa_filter.h"

#include "filter_expr.h"
#include "io_utils.h"

#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using filter_search_io::DenseVectors;
using filter_search_io::IndexFileInfo;
using filter_search_io::MetadataTable;
using filter_search_io::VecFileInfo;

namespace {

struct Args {
    std::string dataset_type;
    std::string graph_path;
    std::string query_path;
    std::string filter_expression;
    std::string fidtb_manifest;

    int k = 0;
    int ef = -1;
    double break_factor = -1.0;
    bool ef_explicit = false;
    bool break_factor_explicit = false;
    int num_queries = -1;
    size_t fid_block_size_bytes = compass_iaa_filter::kDefaultFidBlockSizeBytes;
    size_t tb_block_size_bytes = compass_iaa_filter::kDefaultTbBlockSizeBytes;

    // Backward-compatible optional flags. Not used in the IAA FID/TB path.
    std::string payload_jsonl;
    std::string metadata_csv;
    std::string id_column = "id";

    std::string topk_out;
    std::string per_query_out;
    std::string summary_out;
};

struct FilteredCandidate {
    hnswlib::tableint internal_id = 0;
    hnswlib::labeltype label = 0;
};

struct QueryMetrics {
    size_t qid = 0;
    double recall_at_k = 0.0;
    size_t enns_size = 0;
    size_t anns_size = 0;
    uint64_t filter_time_ns = 0;
    uint64_t search_time_ns = 0;

    uint64_t iaa_decompress_time_ns = 0;
    uint64_t fid_blocks_decompressed = 0;
    uint64_t tb_blocks_decompressed = 0;
    uint64_t fid_cache_hits = 0;
    uint64_t tb_cache_hits = 0;
    uint64_t fid_tb_mismatch_count = 0;
    uint64_t fid_tb_mismatch_log_capped = 0;
};

struct RunStats {
    size_t query_count = 0;

    uint64_t search_loop_time_ns = 0;
    uint64_t filter_eval_calls = 0;
    uint64_t filter_eval_time_ns = 0;

    uint64_t filter_time_total_ns = 0;
    uint64_t search_time_total_ns = 0;

    uint64_t iaa_decompress_time_ns = 0;
    uint64_t fid_blocks_decompressed = 0;
    uint64_t tb_blocks_decompressed = 0;
    uint64_t fid_cache_hits = 0;
    uint64_t tb_cache_hits = 0;
    uint64_t fid_bytes_decompressed = 0;
    uint64_t tb_bytes_decompressed = 0;
    uint64_t fid_tb_mismatch_count = 0;
    uint64_t fid_tb_mismatch_log_capped = 0;

    size_t returned_results = 0;
    size_t selectivity_count = 0;
    size_t total_elements = 0;
    double selectivity_ratio = 0.0;

    double recall_sum = 0.0;
    double average_recall_at_k = 0.0;
    size_t queries_with_enns_lt_k = 0;
    std::string execution_mode = "async_eq_hardware";
    size_t resolved_ef = 0;
    double effective_break_factor = 0.0;

    std::vector<QueryMetrics> per_query_metrics;
};

struct SearchCallStats {
    uint64_t filter_eval_calls = 0;
    uint64_t filter_eval_time_ns = 0;
    compass_iaa_filter::QueryDecompressionMetrics decomp;
    uint64_t fid_tb_mismatch_count = 0;
    uint64_t fid_tb_mismatch_log_capped = 0;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --dataset-type <sift|sift1m|sift1b|laion|hnm>"
        << " --graph <path>"
        << " --query <path(.fvecs|.bvecs)>"
        << " --k <int>"
        << " --filter \"<expression>\""
        << " --fidtb-manifest <path>"
        << " [--ef <int>]"
        << " [--break-factor <float>]"
        << " [--payload-jsonl <path>]"
        << " [--metadata-csv <path>]"
        << " [--id-column id]"
        << " [--num-queries <int>]"
        << " [--max-queries <int>]"  // Backward-compatible alias.
        << " [--fid-block-size-bytes <int>]"
        << " [--tb-block-size-bytes <int>]"
        << " [--topk-out <path>]"
        << " [--per-query-out <path>]"
        << " [--summary-out <path>]\n";
}

void ensure_readable_file(const std::string& path, const std::string& flag_name) {
    if (path.empty()) {
        throw std::runtime_error("Missing required argument: " + flag_name);
    }
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
        throw std::runtime_error("File does not exist: " + path);
    }
}

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string cur = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            ++i;
            return argv[i];
        };

        if (cur == "--dataset-type") {
            args.dataset_type = require_value(cur);
        } else if (cur == "--graph") {
            args.graph_path = require_value(cur);
        } else if (cur == "--query") {
            args.query_path = require_value(cur);
        } else if (cur == "--k") {
            args.k = std::stoi(require_value(cur));
        } else if (cur == "--ef") {
            args.ef = std::stoi(require_value(cur));
            args.ef_explicit = true;
        } else if (cur == "--break-factor") {
            args.break_factor = std::stod(require_value(cur));
            args.break_factor_explicit = true;
        } else if (cur == "--filter") {
            args.filter_expression = require_value(cur);
        } else if (cur == "--fidtb-manifest") {
            args.fidtb_manifest = require_value(cur);
        } else if (cur == "--payload-jsonl") {
            args.payload_jsonl = require_value(cur);
        } else if (cur == "--metadata-csv") {
            args.metadata_csv = require_value(cur);
        } else if (cur == "--id-column") {
            args.id_column = require_value(cur);
        } else if (cur == "--num-queries" || cur == "--max-queries") {
            args.num_queries = std::stoi(require_value(cur));
        } else if (cur == "--fid-block-size-bytes") {
            args.fid_block_size_bytes = static_cast<size_t>(std::stoull(require_value(cur)));
        } else if (cur == "--tb-block-size-bytes") {
            args.tb_block_size_bytes = static_cast<size_t>(std::stoull(require_value(cur)));
        } else if (cur == "--topk-out") {
            args.topk_out = require_value(cur);
        } else if (cur == "--per-query-out") {
            args.per_query_out = require_value(cur);
        } else if (cur == "--summary-out") {
            args.summary_out = require_value(cur);
        } else if (cur == "-h" || cur == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + cur);
        }
    }

    if (args.dataset_type != "sift" &&
        args.dataset_type != "sift1m" &&
        args.dataset_type != "sift1b" &&
        args.dataset_type != "laion" &&
        args.dataset_type != "hnm") {
        throw std::runtime_error("--dataset-type must be one of: sift, sift1m, sift1b, laion, hnm");
    }

    ensure_readable_file(args.graph_path, "--graph");
    ensure_readable_file(args.query_path, "--query");
    ensure_readable_file(args.fidtb_manifest, "--fidtb-manifest");

    const bool graph_bin = filter_search_io::ends_with(args.graph_path, ".bin");
    const bool graph_index = filter_search_io::ends_with(args.graph_path, ".index");
    if (!graph_bin && !graph_index) {
        throw std::runtime_error("--graph must end with .bin or .index");
    }

    if (args.k <= 0) {
        throw std::runtime_error("--k must be > 0");
    }
    if (args.filter_expression.empty()) {
        throw std::runtime_error("--filter is required");
    }
    if (args.num_queries == 0) {
        throw std::runtime_error("--num-queries/--max-queries must be > 0 when provided");
    }
    if (args.fid_block_size_bytes == 0 || args.tb_block_size_bytes == 0) {
        throw std::runtime_error("--fid-block-size-bytes and --tb-block-size-bytes must be > 0");
    }

    const bool query_fvec = filter_search_io::ends_with(args.query_path, ".fvecs");
    const bool query_bvec = filter_search_io::ends_with(args.query_path, ".bvecs");
    if (!query_fvec && !query_bvec) {
        throw std::runtime_error("--query must end with .fvecs or .bvecs");
    }

    if (args.ef_explicit && args.ef <= 0) {
        throw std::runtime_error("--ef must be > 0 when provided");
    }
    if (args.break_factor_explicit && args.break_factor <= 0.0) {
        throw std::runtime_error("--break-factor must be > 0 when provided");
    }
    if (args.id_column.empty()) {
        args.id_column = "id";
    }

    return args;
}

size_t resolve_ef(const Args& args) {
    if (args.ef_explicit && args.ef > 0) {
        return static_cast<size_t>(args.ef);
    }

    if (args.break_factor_explicit && args.break_factor > 0.0) {
        const double scaled = std::ceil(static_cast<double>(args.k) * args.break_factor);
        const size_t derived = static_cast<size_t>(std::max(1.0, scaled));
        return std::max(static_cast<size_t>(args.k), derived);
    }

    return static_cast<size_t>(std::max(100, args.k));
}

double resolve_effective_break_factor(const Args& args, size_t resolved_ef) {
    if (args.k <= 0) {
        return 0.0;
    }
    if (args.ef_explicit) {
        return static_cast<double>(resolved_ef) / static_cast<double>(args.k);
    }
    if (args.break_factor_explicit && args.break_factor > 0.0) {
        return args.break_factor;
    }
    return static_cast<double>(resolved_ef) / static_cast<double>(args.k);
}

template <typename DistT>
using Pair = std::pair<DistT, hnswlib::tableint>;

template <typename DistT>
using CandidateQueue = std::priority_queue<
    Pair<DistT>,
    std::vector<Pair<DistT>>,
    typename hnswlib::HierarchicalNSW<DistT>::CompareByFirst>;

struct EqPredicateSpec {
    std::string field;
    filter_expr::Literal literal;
};

struct AsyncEqRuntime {
    bool enabled = false;
    bool empty_result = false;
    std::string field;
    uint8_t target_bucket = 0;
    size_t n_elements = 0;
    compass_iaa_filter::detail::IaaBlockStorage fid_storage;
    compass_iaa_filter::detail::IaaBlockStorage tb_storage;
};

struct CompressionStats {
    uint64_t fid_raw_bytes = 0;
    uint64_t fid_compressed_bytes = 0;
    uint64_t tb_raw_bytes = 0;
    uint64_t tb_compressed_bytes = 0;
};

uint64_t safe_size_t_to_u64(size_t value) {
    return static_cast<uint64_t>(value);
}

uint64_t total_compressed_block_bytes(const std::vector<std::vector<uint8_t>>& blocks) {
    uint64_t total = 0;
    for (const auto& block : blocks) {
        total += safe_size_t_to_u64(block.size());
    }
    return total;
}

CompressionStats compute_compression_stats(const AsyncEqRuntime& runtime) {
    CompressionStats out;
    out.fid_raw_bytes = safe_size_t_to_u64(runtime.fid_storage.raw_size);
    out.fid_compressed_bytes = total_compressed_block_bytes(runtime.fid_storage.compressed_blocks);
    out.tb_raw_bytes = safe_size_t_to_u64(runtime.tb_storage.raw_size);
    out.tb_compressed_bytes = total_compressed_block_bytes(runtime.tb_storage.compressed_blocks);
    return out;
}

double compression_ratio_raw_over_compressed(uint64_t raw_bytes, uint64_t compressed_bytes) {
    if (raw_bytes == 0 || compressed_bytes == 0) {
        return 0.0;
    }
    return static_cast<double>(raw_bytes) / static_cast<double>(compressed_bytes);
}

double compression_pct_of_raw(uint64_t raw_bytes, uint64_t compressed_bytes) {
    if (raw_bytes == 0) {
        return 0.0;
    }
    return (static_cast<double>(compressed_bytes) * 100.0) / static_cast<double>(raw_bytes);
}

struct FidScanBlock {
    std::vector<uint8_t> matches;
    uint32_t input_elements = 0;
    uint32_t output_bytes = 0;
    bool byte_per_element = false;
};

struct AsyncEqQueryCache {
    std::vector<std::vector<uint8_t>> tb_blocks;
    std::vector<uint8_t> tb_ready;
    std::vector<FidScanBlock> fid_blocks;
    std::vector<uint8_t> fid_ready;
};

bool is_sift_exact_dataset(std::string_view dataset_type) {
    return dataset_type == "sift1m" ||
        dataset_type == "sift1b";
}

std::optional<EqPredicateSpec> parse_single_equality_predicate(const std::string& expression) {
    std::vector<filter_expr::detail::Token> tokens = filter_expr::detail::tokenize(expression);
    filter_expr::detail::Parser parser(std::move(tokens));
    std::unique_ptr<filter_expr::Node> root = parser.parse();
    if (root == nullptr ||
        root->kind != filter_expr::Node::Kind::Compare ||
        root->compare_op != filter_expr::CompareOp::Eq) {
        return std::nullopt;
    }
    EqPredicateSpec out;
    out.field = root->field;
    out.literal = root->literal;
    return out;
}

int derive_usable_bins(int nfilters, int used_bins) {
    int usable_bins = used_bins;
    if (usable_bins <= 0) {
        usable_bins = std::max(1, nfilters - 1);
    }
    usable_bins = std::min(usable_bins, std::max(1, nfilters - 1));
    return usable_bins;
}

int bucket_from_numeric(
    int usable_bins,
    double min_value,
    double max_value,
    double rhs) {
    if (usable_bins <= 1 || max_value <= min_value) {
        return 0;
    }
    const double scale = static_cast<double>(usable_bins - 1) / (max_value - min_value);
    int bucket = static_cast<int>(std::floor((rhs - min_value) * scale));
    if (bucket < 0) {
        bucket = 0;
    }
    if (bucket >= usable_bins) {
        bucket = usable_bins - 1;
    }
    return bucket;
}

AsyncEqRuntime build_async_eq_runtime(
    const Args& args,
    const compass_iaa_filter::CompassIaaFilterEngine& engine,
    const std::optional<EqPredicateSpec>& spec) {
    AsyncEqRuntime runtime;
    if (!spec.has_value()) {
        return runtime;
    }

    const compass_iaa_filter::ManifestData& manifest = engine.manifest();
    const int nfilters = manifest.nfilters;
    if (nfilters <= 0 || nfilters > static_cast<int>(compass_iaa_filter::kMaxBuckets)) {
        return runtime;
    }

    auto attr_it = std::find_if(
        manifest.attributes.begin(),
        manifest.attributes.end(),
        [&](const compass_iaa_filter::ManifestAttribute& attr) {
            return attr.key == spec->field;
        });
    if (attr_it == manifest.attributes.end()) {
        return runtime;
    }

    const compass_iaa_filter::ManifestAttribute& attr = *attr_it;
    const int usable_bins = derive_usable_bins(nfilters, attr.used_bins);
    if (usable_bins <= 0) {
        return runtime;
    }

    int target_bucket = -1;
    if (attr.numeric) {
        if (attr.encoding != "numeric_minmax_quantized") {
            return runtime;
        }
        double rhs = 0.0;
        if (spec->literal.is_number) {
            rhs = spec->literal.number;
        } else if (!filter_expr::detail::try_parse_double(spec->literal.text, &rhs)) {
            return runtime;
        }
        target_bucket = bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, rhs);
    } else {
        auto bucket_it = attr.category_map.find(spec->literal.text);
        if (bucket_it == attr.category_map.end()) {
            runtime.enabled = true;
            runtime.empty_result = true;
            runtime.field = spec->field;
            runtime.n_elements = manifest.n_elements;
            return runtime;
        }
        target_bucket = bucket_it->second;
    }

    if (target_bucket < 0 ||
        target_bucket >= usable_bins ||
        target_bucket >= static_cast<int>(compass_iaa_filter::kMaxBuckets)) {
        runtime.enabled = true;
        runtime.empty_result = true;
        runtime.field = spec->field;
        runtime.n_elements = manifest.n_elements;
        return runtime;
    }

    size_t fid_count = 0;
    std::vector<uint8_t> fid_raw = compass_iaa_filter::detail::read_payload_with_size_header(
        attr.fid_file,
        &fid_count,
        sizeof(uint8_t));
    if (fid_count != manifest.n_elements) {
        throw std::runtime_error(
            "FID element count mismatch while building async runtime for field '" + attr.key + "'");
    }

    size_t tb_count = 0;
    std::vector<uint8_t> tb_raw = compass_iaa_filter::detail::read_payload_with_size_header(
        attr.tb_file,
        &tb_count,
        compass_iaa_filter::kTbBytesPerNode);
    if (tb_count != manifest.n_elements) {
        throw std::runtime_error(
            "TB element count mismatch while building async runtime for field '" + attr.key + "'");
    }

    std::vector<uint8_t> tb_bucket_bits((manifest.n_elements + 7) / 8, 0);
    const size_t bucket_byte = static_cast<size_t>(target_bucket) / 8;
    const uint8_t bucket_mask = static_cast<uint8_t>(1u << (target_bucket % 8));
    for (size_t node_id = 0; node_id < manifest.n_elements; ++node_id) {
        const size_t offset = node_id * compass_iaa_filter::kTbBytesPerNode + bucket_byte;
        if ((tb_raw[offset] & bucket_mask) != 0) {
            tb_bucket_bits[node_id / 8] |= static_cast<uint8_t>(1u << (node_id % 8));
        }
    }

    std::unique_ptr<compass_iaa_filter::detail::QplJobHandle> compressor =
        compass_iaa_filter::detail::make_hardware_job("async eq runtime compressor");
    runtime.fid_storage = compass_iaa_filter::detail::compress_to_iaa_blocks(
        fid_raw,
        args.fid_block_size_bytes,
        compressor.get());
    runtime.tb_storage = compass_iaa_filter::detail::compress_to_iaa_blocks(
        tb_bucket_bits,
        args.tb_block_size_bytes,
        compressor.get());

    runtime.enabled = true;
    runtime.empty_result = false;
    runtime.field = spec->field;
    runtime.target_bucket = static_cast<uint8_t>(target_bucket);
    runtime.n_elements = manifest.n_elements;
    return runtime;
}

size_t max_raw_block_size(const compass_iaa_filter::detail::IaaBlockStorage& storage) {
    size_t out = 0;
    for (uint32_t raw_len : storage.raw_block_sizes) {
        out = std::max(out, static_cast<size_t>(raw_len));
    }
    return out;
}

size_t required_async_job_output_bytes(const AsyncEqRuntime& runtime) {
    const size_t max_fid = max_raw_block_size(runtime.fid_storage);
    const size_t max_tb = max_raw_block_size(runtime.tb_storage);
    return std::max<size_t>(1, std::max(max_fid, max_tb));
}

AsyncEqQueryCache make_preallocated_query_cache(const AsyncEqRuntime& runtime) {
    AsyncEqQueryCache cache;

    const size_t tb_blocks = runtime.tb_storage.block_count();
    cache.tb_blocks.resize(tb_blocks);
    cache.tb_ready.assign(tb_blocks, 0);
    for (size_t block_id = 0; block_id < tb_blocks; ++block_id) {
        cache.tb_blocks[block_id].assign(runtime.tb_storage.raw_block_sizes[block_id], 0);
    }

    const size_t fid_blocks = runtime.fid_storage.block_count();
    cache.fid_blocks.resize(fid_blocks);
    cache.fid_ready.assign(fid_blocks, 0);
    for (size_t block_id = 0; block_id < fid_blocks; ++block_id) {
        FidScanBlock& block = cache.fid_blocks[block_id];
        block.matches.assign(runtime.fid_storage.raw_block_sizes[block_id], 0);
        block.input_elements = runtime.fid_storage.raw_block_sizes[block_id];
        block.output_bytes = 0;
        block.byte_per_element = false;
    }

    return cache;
}

class AsyncJobRing {
public:
    AsyncJobRing(size_t queue_size = 128, size_t wait_batch = 64, size_t output_buffer_bytes = 1)
        : queue_size_(queue_size), wait_batch_(wait_batch), output_buffer_bytes_(output_buffer_bytes) {
        if (queue_size_ == 0 || wait_batch_ == 0 || wait_batch_ > queue_size_) {
            throw std::runtime_error("Invalid AsyncJobRing configuration");
        }
        if (output_buffer_bytes_ == 0) {
            throw std::runtime_error("AsyncJobRing output buffer must be > 0");
        }

        uint32_t job_size = 0;
        qpl_status status = qpl_get_job_size(qpl_path_hardware, &job_size);
        if (status != QPL_STS_OK) {
            throw std::runtime_error("AsyncJobRing: qpl_get_job_size failed");
        }

        slots_.resize(queue_size_);
        for (size_t i = 0; i < queue_size_; ++i) {
            slots_[i].job_storage = std::make_unique<uint8_t[]>(job_size);
            slots_[i].job = reinterpret_cast<qpl_job*>(slots_[i].job_storage.get());
            status = qpl_init_job(qpl_path_hardware, slots_[i].job);
            if (status != QPL_STS_OK) {
                throw std::runtime_error("AsyncJobRing: qpl_init_job failed");
            }
            slots_[i].output.resize(output_buffer_bytes_);
            free_slots_.push_back(i);
        }
    }

    ~AsyncJobRing() {
        for (Slot& slot : slots_) {
            if (slot.job != nullptr) {
                (void)qpl_fini_job(slot.job);
            }
        }
    }

    template <typename SetupFn, typename CompleteFn>
    void submit(size_t output_capacity, SetupFn&& setup_fn, CompleteFn&& complete_fn) {
        ensure_free_slot();

        const size_t slot_id = free_slots_.front();
        free_slots_.pop_front();

        Slot& slot = slots_[slot_id];
        if (output_capacity > output_buffer_bytes_) {
            throw std::runtime_error(
                "AsyncJobRing submit output capacity exceeds preallocated slot buffer");
        }

        setup_fn(slot.job, slot.output);
        const qpl_status submit_status = qpl_submit_job(slot.job);
        if (submit_status != QPL_STS_OK) {
            throw std::runtime_error(
                "AsyncJobRing: qpl_submit_job failed with status " +
                std::to_string(static_cast<int>(submit_status)));
        }

        slot.complete = std::forward<CompleteFn>(complete_fn);
        pending_slots_.push_back(slot_id);

        if (pending_slots_.size() >= queue_size_) {
            wait_oldest(wait_batch_);
        }
    }

    void flush() {
        wait_oldest(pending_slots_.size());
    }

    void wait_one() {
        wait_oldest(1);
    }

    bool has_pending() const {
        return !pending_slots_.empty();
    }

private:
    struct Slot {
        std::unique_ptr<uint8_t[]> job_storage;
        qpl_job* job = nullptr;
        std::vector<uint8_t> output;
        std::function<void(qpl_job*, std::vector<uint8_t>&)> complete;
    };

    void ensure_free_slot() {
        if (!free_slots_.empty()) {
            return;
        }
        wait_oldest(wait_batch_);
        if (free_slots_.empty()) {
            throw std::runtime_error("AsyncJobRing internal error: no free slots after wait");
        }
    }

    void wait_oldest(size_t count) {
        const size_t wait_count = std::min(count, pending_slots_.size());
        for (size_t i = 0; i < wait_count; ++i) {
            const size_t slot_id = pending_slots_.front();
            pending_slots_.pop_front();
            Slot& slot = slots_[slot_id];

            const qpl_status wait_status = qpl_wait_job(slot.job);
            if (wait_status != QPL_STS_OK) {
                throw std::runtime_error(
                    "AsyncJobRing: qpl_wait_job failed with status " +
                    std::to_string(static_cast<int>(wait_status)));
            }

            if (slot.complete) {
                slot.complete(slot.job, slot.output);
            }
            slot.complete = nullptr;
            free_slots_.push_back(slot_id);
        }
    }

    size_t queue_size_ = 128;
    size_t wait_batch_ = 64;
    size_t output_buffer_bytes_ = 1;
    std::vector<Slot> slots_;
    std::deque<size_t> free_slots_;
    std::deque<size_t> pending_slots_;
};

void submit_tb_prefetch_jobs(
    AsyncJobRing* ring,
    const AsyncEqRuntime& runtime,
    AsyncEqQueryCache* cache,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (ring == nullptr || cache == nullptr) {
        throw std::runtime_error("TB prefetch received null pointer");
    }
    for (size_t block_id = 0; block_id < runtime.tb_storage.block_count(); ++block_id) {
        if (block_id >= cache->tb_ready.size() || cache->tb_ready[block_id] != 0) {
            continue;
        }
        const uint32_t raw_len = runtime.tb_storage.raw_block_sizes[block_id];
        if (raw_len == 0) {
            cache->tb_blocks[block_id].clear();
            cache->tb_ready[block_id] = 1;
            continue;
        }

        ring->submit(
            static_cast<size_t>(raw_len),
            [&](qpl_job* job, std::vector<uint8_t>& out) {
                job->op = qpl_op_extract;
                job->next_in_ptr = const_cast<uint8_t*>(runtime.tb_storage.compressed_blocks[block_id].data());
                job->available_in = static_cast<uint32_t>(runtime.tb_storage.compressed_blocks[block_id].size());
                job->next_out_ptr = out.data();
                job->available_out = raw_len;
                job->src1_bit_width = 8;
                job->out_bit_width = qpl_ow_nom;
                job->param_low = 0;
                job->param_high = raw_len - 1;
                job->num_input_elements = raw_len;
                job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
            },
            [cache, metrics, block_id, raw_len](qpl_job* job, std::vector<uint8_t>& out) {
                const size_t produced = static_cast<size_t>(job->total_out);
                std::vector<uint8_t>& dst = cache->tb_blocks[block_id];
                if (produced > dst.size()) {
                    throw std::runtime_error("TB output exceeds preallocated query cache block");
                }
                std::copy_n(out.begin(), produced, dst.begin());
                cache->tb_ready[block_id] = 1;
                if (metrics != nullptr) {
                    ++metrics->tb_blocks_decompressed;
                    metrics->tb_bytes_decompressed += static_cast<uint64_t>(raw_len);
                }
            });
    }
}

void submit_fid_scan_job(
    AsyncJobRing* ring,
    const AsyncEqRuntime& runtime,
    size_t block_id,
    AsyncEqQueryCache* cache,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (ring == nullptr || cache == nullptr) {
        throw std::runtime_error("FID async scan received null pointer");
    }
    if (block_id >= runtime.fid_storage.block_count()) {
        return;
    }
    if (block_id >= cache->fid_ready.size() || cache->fid_ready[block_id] != 0) {
        return;
    }

    const uint32_t raw_len = runtime.fid_storage.raw_block_sizes[block_id];
    if (raw_len == 0) {
        FidScanBlock& block = cache->fid_blocks[block_id];
        block.input_elements = 0;
        block.output_bytes = 0;
        block.byte_per_element = false;
        cache->fid_ready[block_id] = 1;
        return;
    }

    ring->submit(
        static_cast<size_t>(raw_len),
        [&](qpl_job* job, std::vector<uint8_t>& out) {
            job->op = qpl_op_scan_eq;
            job->next_in_ptr = const_cast<uint8_t*>(runtime.fid_storage.compressed_blocks[block_id].data());
            job->available_in = static_cast<uint32_t>(runtime.fid_storage.compressed_blocks[block_id].size());
            job->next_out_ptr = out.data();
            job->available_out = raw_len;
            job->src1_bit_width = 8;
            job->out_bit_width = qpl_ow_nom;
            job->param_low = runtime.target_bucket;
            job->num_input_elements = raw_len;
            job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
        },
        [cache, metrics, block_id, raw_len](qpl_job* job, std::vector<uint8_t>& out) {
            const size_t produced = static_cast<size_t>(job->total_out);
            FidScanBlock& block = cache->fid_blocks[block_id];
            if (produced > block.matches.size()) {
                throw std::runtime_error("FID output exceeds preallocated query cache block");
            }
            std::copy_n(out.begin(), produced, block.matches.begin());
            block.input_elements = raw_len;
            block.output_bytes = static_cast<uint32_t>(produced);
            block.byte_per_element = (job->total_out >= raw_len);
            cache->fid_ready[block_id] = 1;
            if (metrics != nullptr) {
                ++metrics->fid_blocks_decompressed;
                metrics->fid_bytes_decompressed += static_cast<uint64_t>(raw_len);
            }
        });
}

bool tb_match_node(
    const AsyncEqRuntime& runtime,
    AsyncJobRing* ring,
    AsyncEqQueryCache* cache,
    size_t node_id,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr || runtime.tb_storage.block_size == 0 || node_id >= runtime.n_elements) {
        return false;
    }
    const size_t byte_offset = node_id / 8;
    const size_t block_id = byte_offset / runtime.tb_storage.block_size;
    const size_t in_block = byte_offset % runtime.tb_storage.block_size;

    if (block_id >= cache->tb_blocks.size()) {
        return false;
    }
    while (cache->tb_ready[block_id] == 0) {
        if (ring == nullptr || !ring->has_pending()) {
            throw std::runtime_error(
                "TB block is not ready and no pending async jobs exist");
        }
        ring->wait_one();
    }
    const std::vector<uint8_t>& block = cache->tb_blocks[block_id];
    if (in_block >= block.size()) {
        return false;
    }
    if (metrics != nullptr) {
        ++metrics->tb_cache_hits;
    }
    const uint8_t byte = block[in_block];
    return ((byte >> (node_id % 8)) & 1u) != 0;
}

bool fid_match_node(
    const AsyncEqRuntime& runtime,
    AsyncJobRing* ring,
    AsyncEqQueryCache* cache,
    size_t node_id,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr || runtime.fid_storage.block_size == 0 || node_id >= runtime.n_elements) {
        return false;
    }

    const size_t block_id = node_id / runtime.fid_storage.block_size;
    const size_t in_block = node_id % runtime.fid_storage.block_size;
    if (block_id >= cache->fid_blocks.size()) {
        return false;
    }
    while (cache->fid_ready[block_id] == 0) {
        if (ring == nullptr || !ring->has_pending()) {
            throw std::runtime_error(
                "FID block is not ready and no pending async jobs exist");
        }
        ring->wait_one();
    }
    if (metrics != nullptr) {
        ++metrics->fid_cache_hits;
    }

    const FidScanBlock& block = cache->fid_blocks[block_id];
    if (block.byte_per_element) {
        return in_block < block.output_bytes && block.matches[in_block] != 0;
    }

    const size_t byte_idx = in_block / 8;
    if (byte_idx >= block.output_bytes) {
        return false;
    }
    return ((block.matches[byte_idx] >> (in_block % 8)) & 1u) != 0;
}

template <typename DistT, typename QueryT>
std::vector<std::pair<DistT, hnswlib::labeltype>> search_with_compass_filter(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* query_data,
    size_t k,
    size_t ef,
    const compass_iaa_filter::CompassIaaFilterEngine& engine,
    compass_iaa_filter::QueryBlockCache* cache,
    SearchCallStats* call_stats) {
    std::vector<std::pair<DistT, hnswlib::labeltype>> result;
    if (index.cur_element_count == 0) {
        return result;
    }

    hnswlib::tableint curr_obj = index.enterpoint_node_;
    if (static_cast<int>(curr_obj) == -1) {
        return result;
    }

    DistT curdist = index.fstdistfunc_(query_data, index.getDataByInternalId(curr_obj), index.dist_func_param_);

    for (int level = index.maxlevel_; level > 0; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data = reinterpret_cast<unsigned int*>(index.get_linklist(curr_obj, level));
            const int size = index.getListCount(data);
            auto* datal = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int i = 0; i < size; ++i) {
                const hnswlib::tableint cand = datal[i];
                if (cand >= index.cur_element_count) {
                    throw std::runtime_error("Corrupted graph link while traversing upper layers");
                }

                const DistT d = index.fstdistfunc_(query_data, index.getDataByInternalId(cand), index.dist_func_param_);
                if (d < curdist) {
                    curdist = d;
                    curr_obj = cand;
                    changed = true;
                }
            }
        }
    }

    hnswlib::VisitedList* vl = index.visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type* visited = vl->mass;
    const hnswlib::vl_type tag = vl->curV;

    CandidateQueue<DistT> top_candidates;
    CandidateQueue<DistT> candidate_set;

    {
        const DistT dist = index.fstdistfunc_(query_data, index.getDataByInternalId(curr_obj), index.dist_func_param_);
        candidate_set.emplace(-dist, curr_obj);
    }

    visited[curr_obj] = tag;
    DistT lower_bound = std::numeric_limits<DistT>::max();

    auto timed_eval = [&](bool traversal_mode, size_t node_id) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        bool allowed = false;
        if (traversal_mode) {
            allowed = engine.allow_traversal(node_id, cache, &call_stats->decomp);
        } else {
            allowed = engine.allow_result(node_id, cache, &call_stats->decomp);
        }
        const auto t1 = std::chrono::steady_clock::now();
        ++call_stats->filter_eval_calls;
        call_stats->filter_eval_time_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        return allowed;
    };

    while (!candidate_set.empty()) {
        const Pair<DistT> current = candidate_set.top();
        const DistT candidate_dist = -current.first;

        if (candidate_dist > lower_bound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        const hnswlib::tableint current_node_id = current.second;
        int* data = reinterpret_cast<int*>(index.get_linklist0(current_node_id));
        const size_t size = index.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));

        for (size_t j = 1; j <= size; ++j) {
            const hnswlib::tableint candidate_id = static_cast<hnswlib::tableint>(*(data + j));
            if (candidate_id >= index.cur_element_count) {
                throw std::runtime_error("Corrupted graph link while traversing level-0");
            }
            if (visited[candidate_id] == tag) {
                continue;
            }
            visited[candidate_id] = tag;

            if (!timed_eval(true, static_cast<size_t>(candidate_id))) {
                continue;
            }

            const DistT dist = index.fstdistfunc_(
                query_data,
                index.getDataByInternalId(candidate_id),
                index.dist_func_param_);

            if (top_candidates.size() < ef || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);

                if (!index.isMarkedDeleted(candidate_id) &&
                    timed_eval(false, static_cast<size_t>(candidate_id))) {
                    top_candidates.emplace(dist, candidate_id);
                }

                while (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }

    size_t sz = top_candidates.size();
    result.resize(sz);
    while (!top_candidates.empty()) {
        const Pair<DistT> item = top_candidates.top();
        top_candidates.pop();
        result[--sz] = std::make_pair(item.first, index.getExternalLabel(item.second));
    }

    index.visited_list_pool_->releaseVisitedList(vl);
    return result;
}

template <typename DistT, typename QueryT>
std::vector<std::pair<DistT, hnswlib::labeltype>> search_with_async_eq_filter(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* query_data,
    size_t k,
    size_t ef,
    const AsyncEqRuntime& runtime,
    AsyncJobRing* ring,
    AsyncEqQueryCache* cache,
    SearchCallStats* call_stats) {
    std::vector<std::pair<DistT, hnswlib::labeltype>> result;
    if (index.cur_element_count == 0 || runtime.empty_result) {
        return result;
    }

    if (runtime.n_elements != index.cur_element_count) {
        throw std::runtime_error("Async runtime element count does not match index");
    }

    SearchCallStats local_stats;
    if (call_stats == nullptr) {
        call_stats = &local_stats;
    }
    if (ring == nullptr) {
        throw std::runtime_error("AsyncJobRing is null");
    }
    if (cache == nullptr) {
        throw std::runtime_error("AsyncEqQueryCache is null");
    }
    ring->flush();

    hnswlib::tableint curr_obj = index.enterpoint_node_;
    if (static_cast<int>(curr_obj) == -1) {
        return result;
    }

    DistT curdist = index.fstdistfunc_(query_data, index.getDataByInternalId(curr_obj), index.dist_func_param_);

    if (cache->tb_blocks.size() != runtime.tb_storage.block_count() ||
        cache->tb_ready.size() != runtime.tb_storage.block_count() ||
        cache->fid_blocks.size() != runtime.fid_storage.block_count() ||
        cache->fid_ready.size() != runtime.fid_storage.block_count()) {
        throw std::runtime_error(
            "AsyncEqQueryCache must be fully preallocated before search");
    }
    submit_tb_prefetch_jobs(ring, runtime, cache, &call_stats->decomp);

    for (int level = index.maxlevel_; level > 0; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data = reinterpret_cast<unsigned int*>(index.get_linklist(curr_obj, level));
            const int size = index.getListCount(data);
            auto* datal = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int i = 0; i < size; ++i) {
                const hnswlib::tableint cand = datal[i];
                if (cand >= index.cur_element_count) {
                    throw std::runtime_error("Corrupted graph link while traversing upper layers");
                }

                const DistT d = index.fstdistfunc_(query_data, index.getDataByInternalId(cand), index.dist_func_param_);
                if (d < curdist) {
                    curdist = d;
                    curr_obj = cand;
                    changed = true;
                }
            }
        }
    }

    hnswlib::VisitedList* vl = index.visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type* visited = vl->mass;
    const hnswlib::vl_type tag = vl->curV;

    CandidateQueue<DistT> top_candidates;
    CandidateQueue<DistT> candidate_set;
    {
        const DistT dist = index.fstdistfunc_(query_data, index.getDataByInternalId(curr_obj), index.dist_func_param_);
        candidate_set.emplace(-dist, curr_obj);
    }

    visited[curr_obj] = tag;
    DistT lower_bound = std::numeric_limits<DistT>::max();

    struct FrontierCandidate {
        hnswlib::tableint candidate_id = 0;
        DistT dist{};
        size_t fid_block_id = 0;
        bool deleted = false;
        bool need_fid = false;
    };
    std::vector<hnswlib::tableint> neighbor_ids;
    neighbor_ids.reserve(index.maxM0_);
    std::vector<FrontierCandidate> frontier_candidates;
    frontier_candidates.reserve(index.maxM0_);
    std::vector<size_t> fid_blocks_to_submit;
    fid_blocks_to_submit.reserve(index.maxM0_);

    while (!candidate_set.empty()) {
        const Pair<DistT> current = candidate_set.top();
        const DistT candidate_dist = -current.first;
        if (candidate_dist > lower_bound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        const hnswlib::tableint current_node_id = current.second;
        int* data = reinterpret_cast<int*>(index.get_linklist0(current_node_id));
        const size_t size = index.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));

        neighbor_ids.clear();

        for (size_t j = 1; j <= size; ++j) {
            const hnswlib::tableint candidate_id = static_cast<hnswlib::tableint>(*(data + j));
            if (candidate_id >= index.cur_element_count) {
                throw std::runtime_error("Corrupted graph link while traversing level-0");
            }
            if (visited[candidate_id] == tag) {
                continue;
            }
            visited[candidate_id] = tag;
            neighbor_ids.push_back(candidate_id);
        }

        frontier_candidates.clear();
        fid_blocks_to_submit.clear();

        // Stage 1: TB check neighbors and group survivors.
        for (size_t idx = 0; idx < neighbor_ids.size(); ++idx) {
            const hnswlib::tableint candidate_id = neighbor_ids[idx];

            const bool tb_match = tb_match_node(
                runtime,
                ring,
                cache,
                static_cast<size_t>(candidate_id),
                &call_stats->decomp);

            if (!tb_match) {
                ++call_stats->filter_eval_calls;
                continue;
            }

            FrontierCandidate entry;
            entry.candidate_id = candidate_id;
            entry.deleted = index.isMarkedDeleted(candidate_id);
            if (runtime.fid_storage.block_size != 0) {
                entry.fid_block_id = static_cast<size_t>(candidate_id) / runtime.fid_storage.block_size;
                entry.need_fid = true;
                if (entry.fid_block_id < cache->fid_ready.size() && cache->fid_ready[entry.fid_block_id] == 0) {
                    fid_blocks_to_submit.push_back(entry.fid_block_id);
                }
            }
            frontier_candidates.push_back(entry);
        }

        // Stage 2: submit grouped FID jobs.
        if (!fid_blocks_to_submit.empty()) {
            std::sort(fid_blocks_to_submit.begin(), fid_blocks_to_submit.end());
            fid_blocks_to_submit.erase(
                std::unique(fid_blocks_to_submit.begin(), fid_blocks_to_submit.end()),
                fid_blocks_to_submit.end());
            for (size_t fid_block_id : fid_blocks_to_submit) {
                submit_fid_scan_job(ring, runtime, fid_block_id, cache, &call_stats->decomp);
            }
        }

        // Stage 3: run distance calculation for grouped neighbors.
        for (FrontierCandidate& entry : frontier_candidates) {
            entry.dist = index.fstdistfunc_(
                query_data,
                index.getDataByInternalId(entry.candidate_id),
                index.dist_func_param_);
        }

        // Stage 4: update candidates/results.
        for (const FrontierCandidate& entry : frontier_candidates) {
            const bool consider = (top_candidates.size() < ef || lower_bound > entry.dist);
            if (!consider) {
                ++call_stats->filter_eval_calls;
                continue;
            }

            candidate_set.emplace(-entry.dist, entry.candidate_id);

            bool fid_match = true;
            if (entry.need_fid) {
                fid_match = fid_match_node(
                    runtime,
                    ring,
                    cache,
                    static_cast<size_t>(entry.candidate_id),
                    &call_stats->decomp);
            }

            ++call_stats->filter_eval_calls;

            const bool result_allowed = !entry.deleted && fid_match;
            if (result_allowed) {
                top_candidates.emplace(entry.dist, entry.candidate_id);
            }

            while (top_candidates.size() > ef) {
                top_candidates.pop();
            }
            if (!top_candidates.empty()) {
                lower_bound = top_candidates.top().first;
            }
        }
    }

    ring->flush();
    while (top_candidates.size() > k) {
        top_candidates.pop();
    }

    size_t sz = top_candidates.size();
    result.resize(sz);
    while (!top_candidates.empty()) {
        const Pair<DistT> item = top_candidates.top();
        top_candidates.pop();
        result[--sz] = std::make_pair(item.first, index.getExternalLabel(item.second));
    }

    index.visited_list_pool_->releaseVisitedList(vl);
    return result;
}

template <typename DistT, typename QueryT>
std::priority_queue<std::pair<DistT, hnswlib::labeltype>> build_enns_heap(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* qptr,
    size_t k,
    const std::vector<FilteredCandidate>& filtered_candidates) {
    std::priority_queue<std::pair<DistT, hnswlib::labeltype>> heap;
    for (const FilteredCandidate& candidate : filtered_candidates) {
        const char* data_ptr = index.getDataByInternalId(candidate.internal_id);
        const DistT dist = index.fstdistfunc_(qptr, data_ptr, index.dist_func_param_);

        if (heap.size() < k) {
            heap.emplace(dist, candidate.label);
        } else if (dist < heap.top().first) {
            heap.pop();
            heap.emplace(dist, candidate.label);
        }
    }
    return heap;
}

template <typename DistT, typename SpaceT, typename QueryT>
RunStats run_search_typed(
    const Args& args,
    int dim,
    const DenseVectors<QueryT>& queries,
    const compass_iaa_filter::CompassIaaFilterEngine& engine,
    const MetadataTable& metadata,
    const AsyncEqRuntime* async_runtime) {
    const size_t resolved_ef = resolve_ef(args);
    const double effective_break_factor = resolve_effective_break_factor(args, resolved_ef);

    SpaceT space(static_cast<size_t>(dim));
    hnswlib::HierarchicalNSW<DistT> index(&space, args.graph_path);
    index.setEf(resolved_ef);

    if (engine.num_elements() != index.getCurrentElementCount()) {
        throw std::runtime_error(
            "FID/TB manifest n_elements does not match graph element count: manifest=" +
            std::to_string(engine.num_elements()) +
            ", graph=" + std::to_string(index.getCurrentElementCount()));
    }

    RunStats stats;
    const bool use_async_eq = (async_runtime != nullptr && async_runtime->enabled);
    if (!use_async_eq) {
        throw std::runtime_error(
            "IAA hardware async exact-match runtime is required; software fallback is disabled");
    }
    stats.execution_mode = "async_eq_hardware";
    stats.resolved_ef = resolved_ef;
    stats.effective_break_factor = effective_break_factor;
    const size_t total_elements = index.getCurrentElementCount();
    const size_t async_job_output_bytes = required_async_job_output_bytes(*async_runtime);
    std::unique_ptr<AsyncJobRing> shared_ring;
    shared_ring = std::make_unique<AsyncJobRing>(128, 64, async_job_output_bytes);

    std::vector<size_t> result_nodes = engine.collect_result_candidates();
    std::vector<FilteredCandidate> filtered_candidates;
    filtered_candidates.reserve(result_nodes.size());
    for (size_t node_id : result_nodes) {
        const hnswlib::tableint iid = static_cast<hnswlib::tableint>(node_id);
        filtered_candidates.push_back(FilteredCandidate{iid, index.getExternalLabel(iid)});
    }

    stats.total_elements = total_elements;
    stats.selectivity_count = filtered_candidates.size();
    if (total_elements > 0) {
        stats.selectivity_ratio =
            static_cast<double>(stats.selectivity_count) / static_cast<double>(total_elements);
    }

    size_t query_count = static_cast<size_t>(queries.num);
    if (args.num_queries > 0) {
        query_count = std::min(query_count, static_cast<size_t>(args.num_queries));
    }
    if (query_count == 0) {
        throw std::runtime_error("No query vectors available after applying --num-queries/--max-queries");
    }

    std::ofstream topk_out;
    if (!args.topk_out.empty()) {
        fs::path out_path(args.topk_out);
        if (!out_path.parent_path().empty()) {
            fs::create_directories(out_path.parent_path());
        }
        topk_out.open(args.topk_out);
        if (!topk_out.is_open()) {
            throw std::runtime_error("Failed to open --topk-out: " + args.topk_out);
        }
        topk_out << "query_id\tresults(id:dist,...)\n";
    }

    std::ofstream per_query_out;
    if (!args.per_query_out.empty()) {
        fs::path out_path(args.per_query_out);
        if (!out_path.parent_path().empty()) {
            fs::create_directories(out_path.parent_path());
        }
        per_query_out.open(args.per_query_out);
        if (!per_query_out.is_open()) {
            throw std::runtime_error("Failed to open --per-query-out: " + args.per_query_out);
        }
        per_query_out
            << "query_id,recall_at_k,enns_size,anns_size,filter_time_ms,search_time_ms,"
            << "lz4_decompress_time_ms,iaa_decompress_time_ms,fid_blocks_decompressed,tb_blocks_decompressed,"
            << "fid_cache_hits,tb_cache_hits,fid_tb_mismatch_count,fid_tb_mismatch_log_capped\n";
    }

    const bool capture_per_query = per_query_out.is_open();
    if (capture_per_query) {
        stats.per_query_metrics.reserve(query_count);
    }

    size_t returned_results = 0;

    for (size_t qid = 0; qid < query_count; ++qid) {
        const QueryT* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);

        SearchCallStats call_stats;
        AsyncEqQueryCache query_async_cache = make_preallocated_query_cache(*async_runtime);

        const auto search_start = std::chrono::steady_clock::now();
        std::vector<std::pair<DistT, hnswlib::labeltype>> result;
        result = search_with_async_eq_filter<DistT, QueryT>(
            index,
            qptr,
            static_cast<size_t>(args.k),
            resolved_ef,
            *async_runtime,
            shared_ring.get(),
            &query_async_cache,
            &call_stats);
        const auto search_end = std::chrono::steady_clock::now();

        const uint64_t query_search_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(search_end - search_start).count());

        std::priority_queue<std::pair<DistT, hnswlib::labeltype>> enns_heap =
            build_enns_heap(index, qptr, static_cast<size_t>(args.k), filtered_candidates);

        std::unordered_set<hnswlib::labeltype> enns_labels;
        enns_labels.reserve(enns_heap.size() * 2 + 1);
        while (!enns_heap.empty()) {
            enns_labels.insert(enns_heap.top().second);
            enns_heap.pop();
        }

        std::unordered_set<hnswlib::labeltype> anns_labels;
        anns_labels.reserve(result.size() * 2 + 1);
        for (const auto& item : result) {
            anns_labels.insert(item.second);
        }

        size_t overlap = 0;
        for (hnswlib::labeltype label : enns_labels) {
            if (anns_labels.find(label) != anns_labels.end()) {
                ++overlap;
            }
        }

        const size_t enns_size = enns_labels.size();
        if (enns_size < static_cast<size_t>(args.k)) {
            ++stats.queries_with_enns_lt_k;
        }

        double recall = 0.0;
        if (enns_size == 0) {
            recall = result.empty() ? 1.0 : 0.0;
        } else {
            const size_t denom = std::min(static_cast<size_t>(args.k), enns_size);
            recall = static_cast<double>(overlap) / static_cast<double>(denom);
        }

        stats.recall_sum += recall;
        stats.filter_eval_calls += call_stats.filter_eval_calls;
        stats.filter_eval_time_ns += call_stats.filter_eval_time_ns;
        stats.filter_time_total_ns += call_stats.filter_eval_time_ns;
        stats.search_time_total_ns += query_search_ns;

        stats.iaa_decompress_time_ns += call_stats.decomp.iaa_decompress_time_ns;
        stats.fid_blocks_decompressed += call_stats.decomp.fid_blocks_decompressed;
        stats.tb_blocks_decompressed += call_stats.decomp.tb_blocks_decompressed;
        stats.fid_cache_hits += call_stats.decomp.fid_cache_hits;
        stats.tb_cache_hits += call_stats.decomp.tb_cache_hits;
        stats.fid_bytes_decompressed += call_stats.decomp.fid_bytes_decompressed;
        stats.tb_bytes_decompressed += call_stats.decomp.tb_bytes_decompressed;
        stats.fid_tb_mismatch_count += call_stats.fid_tb_mismatch_count;
        stats.fid_tb_mismatch_log_capped += call_stats.fid_tb_mismatch_log_capped;
        stats.search_loop_time_ns += query_search_ns;

        returned_results += result.size();

        if (topk_out.is_open()) {
            topk_out << qid << '\t';
            for (size_t i = 0; i < result.size(); ++i) {
                if (i > 0) {
                    topk_out << ',';
                }
                topk_out << result[i].second << ':' << static_cast<double>(result[i].first);
            }
            topk_out << '\n';
        }

        if (capture_per_query) {
            QueryMetrics m;
            m.qid = qid;
            m.recall_at_k = recall;
            m.enns_size = enns_size;
            m.anns_size = result.size();
            m.filter_time_ns = call_stats.filter_eval_time_ns;
            m.search_time_ns = query_search_ns;

            m.iaa_decompress_time_ns = call_stats.decomp.iaa_decompress_time_ns;
            m.fid_blocks_decompressed = call_stats.decomp.fid_blocks_decompressed;
            m.tb_blocks_decompressed = call_stats.decomp.tb_blocks_decompressed;
            m.fid_cache_hits = call_stats.decomp.fid_cache_hits;
            m.tb_cache_hits = call_stats.decomp.tb_cache_hits;
            m.fid_tb_mismatch_count = call_stats.fid_tb_mismatch_count;
            m.fid_tb_mismatch_log_capped = call_stats.fid_tb_mismatch_log_capped;
            stats.per_query_metrics.push_back(m);

            per_query_out
                << qid << ','
                << std::fixed << std::setprecision(6) << recall << ','
                << enns_size << ','
                << result.size() << ','
                << std::setprecision(6) << (static_cast<double>(call_stats.filter_eval_time_ns) / 1e6) << ','
                << std::setprecision(6) << (static_cast<double>(query_search_ns) / 1e6) << ','
                << std::setprecision(6) << (static_cast<double>(call_stats.decomp.iaa_decompress_time_ns) / 1e6) << ','
                << std::setprecision(6) << (static_cast<double>(call_stats.decomp.iaa_decompress_time_ns) / 1e6) << ','
                << call_stats.decomp.fid_blocks_decompressed << ','
                << call_stats.decomp.tb_blocks_decompressed << ','
                << call_stats.decomp.fid_cache_hits << ','
                << call_stats.decomp.tb_cache_hits << ','
                << call_stats.fid_tb_mismatch_count << ','
                << call_stats.fid_tb_mismatch_log_capped
                << '\n';
        }
    }

    stats.query_count = query_count;
    stats.returned_results = returned_results;
    stats.average_recall_at_k =
        (query_count > 0) ? (stats.recall_sum / static_cast<double>(query_count)) : 0.0;

    (void)metadata;
    return stats;
}

std::string build_summary(
    const Args& args,
    const VecFileInfo& query_info,
    const IndexFileInfo& index_info,
    const std::string& index_vector_type,
    size_t index_dim,
    const MetadataTable& metadata,
    const filter_expr::Expression& expr,
    const RunStats& stats,
    const CompressionStats& compression_stats) {
    const double loop_ms = static_cast<double>(stats.search_loop_time_ns) / 1e6;
    const double avg_query_ms =
        (stats.query_count > 0) ? (loop_ms / static_cast<double>(stats.query_count)) : 0.0;
    const double filter_ms = static_cast<double>(stats.filter_eval_time_ns) / 1e6;
    const double avg_filter_ns =
        (stats.filter_eval_calls > 0)
            ? (static_cast<double>(stats.filter_eval_time_ns) / static_cast<double>(stats.filter_eval_calls))
            : 0.0;
    const double avg_filter_per_query_ms =
        (stats.query_count > 0)
            ? (static_cast<double>(stats.filter_time_total_ns) / 1e6 / static_cast<double>(stats.query_count))
            : 0.0;
    const double avg_search_per_query_ms =
        (stats.query_count > 0)
            ? (static_cast<double>(stats.search_time_total_ns) / 1e6 / static_cast<double>(stats.query_count))
            : 0.0;

    const double decomp_ms = static_cast<double>(stats.iaa_decompress_time_ns) / 1e6;
    const double avg_decomp_per_query_ms =
        (stats.query_count > 0)
            ? (decomp_ms / static_cast<double>(stats.query_count))
            : 0.0;
    const double qps =
        (stats.search_loop_time_ns > 0)
            ? (static_cast<double>(stats.query_count) * 1e9 / static_cast<double>(stats.search_loop_time_ns))
            : 0.0;
    const uint64_t total_raw_bytes = compression_stats.fid_raw_bytes + compression_stats.tb_raw_bytes;
    const uint64_t total_compressed_bytes =
        compression_stats.fid_compressed_bytes + compression_stats.tb_compressed_bytes;
    const double fid_ratio = compression_ratio_raw_over_compressed(
        compression_stats.fid_raw_bytes,
        compression_stats.fid_compressed_bytes);
    const double tb_ratio = compression_ratio_raw_over_compressed(
        compression_stats.tb_raw_bytes,
        compression_stats.tb_compressed_bytes);
    const double total_ratio =
        compression_ratio_raw_over_compressed(total_raw_bytes, total_compressed_bytes);
    const double total_compressed_pct = compression_pct_of_raw(total_raw_bytes, total_compressed_bytes);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "hnswlib_filter_search summary\n";
    oss << "dataset_type: " << args.dataset_type << "\n";
    oss << "query_path: " << args.query_path << "\n";
    oss << "graph_path: " << args.graph_path << "\n";
    oss << "fidtb_manifest: " << args.fidtb_manifest << "\n";
    oss << "fid_block_size_bytes: " << args.fid_block_size_bytes << "\n";
    oss << "tb_block_size_bytes: " << args.tb_block_size_bytes << "\n";
    oss << "index_elements: " << index_info.cur_element_count << "\n";
    oss << "index_dimension: " << index_dim << "\n";
    oss << "index_vector_type: " << index_vector_type << "\n";
    oss << "query_num: " << query_info.num << ", query_dim: " << query_info.dim << "\n";
    oss << "queries_requested: "
        << (args.num_queries > 0 ? std::to_string(args.num_queries) : std::string("all")) << "\n";
    oss << "k: " << args.k << ", ef: " << stats.resolved_ef
        << ", break_factor: " << stats.effective_break_factor << "\n";
    oss << "fid_raw_bytes: " << compression_stats.fid_raw_bytes << "\n";
    oss << "fid_compressed_bytes: " << compression_stats.fid_compressed_bytes << "\n";
    oss << "tb_raw_bytes: " << compression_stats.tb_raw_bytes << "\n";
    oss << "tb_compressed_bytes: " << compression_stats.tb_compressed_bytes << "\n";
    oss << "fid_compression_ratio_raw_over_compressed: " << fid_ratio << "\n";
    oss << "tb_compression_ratio_raw_over_compressed: " << tb_ratio << "\n";
    oss << "filter_payload_raw_bytes: " << total_raw_bytes << "\n";
    oss << "filter_payload_compressed_bytes: " << total_compressed_bytes << "\n";
    oss << "filter_payload_compression_ratio_raw_over_compressed: " << total_ratio << "\n";
    oss << "filter_payload_compressed_pct_of_raw: " << total_compressed_pct << "\n";
    oss << "filter: " << expr.source() << "\n";
    oss << "execution_mode: " << stats.execution_mode << "\n";

    // Keep legacy metadata_* keys for script compatibility.
    oss << "metadata_total_labels: " << metadata.total_labels << "\n";
    oss << "metadata_populated_labels: " << metadata.populated_rows << "\n";
    oss << "metadata_missing_labels: " << metadata.missing_rows << "\n";
    oss << "metadata_invalid_rows: " << metadata.invalid_rows << "\n";
    oss << "metadata_dropped_rows: " << metadata.dropped_rows << "\n";

    oss << "selectivity_count: " << stats.selectivity_count << "\n";
    oss << "selectivity_ratio: " << stats.selectivity_ratio << "\n";
    oss << "queries_executed: " << stats.query_count << "\n";
    oss << "results_returned_total: " << stats.returned_results << "\n";
    oss << "average_recall_at_k: " << stats.average_recall_at_k << "\n";
    oss << "queries_with_enns_lt_k: " << stats.queries_with_enns_lt_k << "\n";

    oss << "search_loop_time_ms: " << loop_ms << "\n";
    oss << "avg_query_time_ms: " << avg_query_ms << "\n";
    oss << "qps: " << qps << "\n";

    oss << "filter_eval_calls: " << stats.filter_eval_calls << "\n";
    oss << "filter_eval_time_ms: " << filter_ms << "\n";
    oss << "avg_filter_eval_ns: " << avg_filter_ns << "\n";
    oss << "avg_filter_time_per_query_ms: " << avg_filter_per_query_ms << "\n";
    oss << "avg_search_time_per_query_ms: " << avg_search_per_query_ms << "\n";

    // Keep the legacy key for compatibility with scripts that parse previous output.
    oss << "lz4_decompress_time_ms: " << decomp_ms << "\n";
    oss << "iaa_decompress_time_ms: " << decomp_ms << "\n";
    oss << "avg_decompress_time_per_query_ms: " << avg_decomp_per_query_ms << "\n";
    oss << "fid_blocks_decompressed: " << stats.fid_blocks_decompressed << "\n";
    oss << "tb_blocks_decompressed: " << stats.tb_blocks_decompressed << "\n";
    oss << "fid_cache_hits: " << stats.fid_cache_hits << "\n";
    oss << "tb_cache_hits: " << stats.tb_cache_hits << "\n";
    oss << "fid_bytes_decompressed: " << stats.fid_bytes_decompressed << "\n";
    oss << "tb_bytes_decompressed: " << stats.tb_bytes_decompressed << "\n";
    oss << "fid_tb_mismatch_count: " << stats.fid_tb_mismatch_count << "\n";
    oss << "fid_tb_mismatch_log_capped: " << stats.fid_tb_mismatch_log_capped << "\n";

    if (stats.queries_with_enns_lt_k > 0) {
        oss << "warning: filtering condition is too restrictive for k on "
            << stats.queries_with_enns_lt_k << " queries\n";
    }
    if (stats.fid_tb_mismatch_count > 0) {
        oss << "warning: FID/TB consistency mismatches detected\n";
    }
    return oss.str();
}

void write_summary_if_needed(const std::string& summary_text, const std::string& summary_path) {
    if (summary_path.empty()) {
        return;
    }
    fs::path out(summary_path);
    if (!out.parent_path().empty()) {
        fs::create_directories(out.parent_path());
    }
    std::ofstream ofs(summary_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to write --summary-out: " + summary_path);
    }
    ofs << summary_text;
}

MetadataTable build_manifest_backed_metadata(const compass_iaa_filter::ManifestData& manifest) {
    MetadataTable table;
    table.total_labels = manifest.n_elements;
    table.populated_rows = manifest.n_elements;
    table.missing_rows = 0;
    table.invalid_rows = 0;
    table.dropped_rows = 0;
    return table;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (!is_sift_exact_dataset(args.dataset_type)) {
            throw std::runtime_error(
                "compass_search_w_iaa_async is optimized for exact-match SIFT datasets only; "
                "--dataset-type must be sift1m or sift1b");
        }

        const VecFileInfo query_info = filter_search_io::inspect_vector_file(args.query_path);
        const IndexFileInfo index_info = filter_search_io::inspect_hnsw_index_file(args.graph_path);

        filter_expr::Expression expr(args.filter_expression);
        const std::optional<EqPredicateSpec> eq_spec =
            parse_single_equality_predicate(args.filter_expression);
        if (!eq_spec.has_value()) {
            throw std::runtime_error(
                "This executable only supports exact-match filters of the form <field> == <value>");
        }
        std::unordered_set<std::string> fields = expr.referenced_fields();
        if (fields.empty()) {
            throw std::runtime_error("Filter expression did not reference any fields");
        }

        const compass_iaa_filter::CompassIaaFilterEngine engine =
            compass_iaa_filter::CompassIaaFilterEngine::Build(
                args.fidtb_manifest,
                args.filter_expression,
                fields,
                args.fid_block_size_bytes,
                args.tb_block_size_bytes);
        const AsyncEqRuntime async_runtime = build_async_eq_runtime(args, engine, eq_spec);
        if (!async_runtime.enabled) {
            throw std::runtime_error(
                "Failed to initialize IAA hardware exact-match runtime; software fallback is disabled");
        }
        const CompressionStats compression_stats = compute_compression_stats(async_runtime);

        const bool query_is_fvecs = filter_search_io::ends_with(args.query_path, ".fvecs");
        const size_t query_elem_bytes = query_is_fvecs ? sizeof(float) : sizeof(uint8_t);
        const std::string index_vector_type = query_is_fvecs ? "fvecs" : "bvecs";

        if (index_info.data_size_bytes % query_elem_bytes != 0) {
            std::ostringstream oss;
            oss << "Graph/query type mismatch: index data size bytes " << index_info.data_size_bytes
                << " is not divisible by query element size " << query_elem_bytes;
            throw std::runtime_error(oss.str());
        }

        const size_t index_dim = index_info.data_size_bytes / query_elem_bytes;
        if (index_dim == 0) {
            throw std::runtime_error("Derived graph index dimension is zero");
        }
        if (index_dim != static_cast<size_t>(query_info.dim)) {
            std::ostringstream oss;
            oss << "Graph/query dimension mismatch: index dim " << index_dim
                << " vs query dim " << query_info.dim;
            throw std::runtime_error(oss.str());
        }

        MetadataTable metadata = build_manifest_backed_metadata(engine.manifest());

        RunStats stats;
        if (query_is_fvecs) {
            DenseVectors<float> queries = filter_search_io::read_fvecs(args.query_path);
            stats = run_search_typed<float, hnswlib::L2Space, float>(
                args,
                static_cast<int>(index_dim),
                queries,
                engine,
                metadata,
                &async_runtime);
        } else {
            DenseVectors<uint8_t> queries = filter_search_io::read_bvecs(args.query_path);
            stats = run_search_typed<int, hnswlib::L2SpaceI, uint8_t>(
                args,
                static_cast<int>(index_dim),
                queries,
                engine,
                metadata,
                &async_runtime);
        }

        const std::string summary = build_summary(
            args,
            query_info,
            index_info,
            index_vector_type,
            index_dim,
            metadata,
            expr,
            stats,
            compression_stats);
        std::cout << summary;
        write_summary_if_needed(summary, args.summary_out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// ./compass_search_w_iaa_async.run \
//   --dataset-type sift1m \
//   --graph /storage/jykang5/compass_graphs/sift_m128_efc200.bin \
//   --query /storage/jykang5/compass_base_query/sift1m_query.fvecs \
//   --fidtb-manifest /storage/jykang5/fid_tb/n_filter_100/sift1m/manifest.json \
//   --filter "synthetic_id_bucket == 0" \
//   --k 10 \
//   --ef 200 \
//   --num-queries 100 \
//   --summary-out /tmp/iaa_async_summary.txt \
//   --per-query-out /tmp/iaa_async_per_query.csv
