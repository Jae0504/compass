#include "../../../hnswlib/filter_search_hnswlib/hnswlib.h"
#include "../../../hnswlib/filter_search_iaa/compass_iaa_filter.h"

#include "../filter_expr.h"
#include "../io_utils.h"

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

    size_t returned_results = 0;
    size_t selectivity_count = 0;
    size_t total_elements = 0;
    double selectivity_ratio = 0.0;

    double recall_sum = 0.0;
    double average_recall_at_k = 0.0;
    size_t queries_with_enns_lt_k = 0;
    std::string execution_mode = "async_range";

    std::vector<QueryMetrics> per_query_metrics;
};

struct SearchCallStats {
    uint64_t filter_eval_calls = 0;
    uint64_t filter_eval_time_ns = 0;
    compass_iaa_filter::QueryDecompressionMetrics decomp;
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

    if (args.ef <= 0) {
        args.ef = std::max(100, args.k);
    }
    if (args.id_column.empty()) {
        args.id_column = "id";
    }

    return args;
}

template <typename DistT>
using Pair = std::pair<DistT, hnswlib::tableint>;

template <typename DistT>
using CandidateQueue = std::priority_queue<
    Pair<DistT>,
    std::vector<Pair<DistT>>,
    typename hnswlib::HierarchicalNSW<DistT>::CompareByFirst>;

struct RangePredicateSpec {
    std::string field;
    double low = -std::numeric_limits<double>::infinity();
    double high = std::numeric_limits<double>::infinity();
    bool low_inclusive = true;
    bool high_inclusive = true;
};

struct AsyncRangeRuntime {
    bool enabled = true;
    bool empty_result = false;
    std::string field;
    uint8_t bucket_low = 0;
    uint8_t bucket_high = 0;
    size_t n_elements = 0;
    int nfilters = 0;
    int missing_bucket = -1;
    int usable_bins = 0;
    double min_value = 0.0;
    double max_value = 0.0;
    double low = 0.0;
    double high = 0.0;
    bool low_bounded = false;
    bool high_bounded = false;
    bool low_inclusive = true;
    bool high_inclusive = true;
    compass_iaa_filter::detail::IaaBlockStorage fid_storage;
    compass_iaa_filter::detail::IaaBlockStorage tb_storage;
};

struct FidScanBlock {
    std::vector<uint8_t> matches;
    uint32_t input_elements = 0;
    bool byte_per_element = false;
};

struct AsyncRangeQueryCache {
    std::unordered_map<size_t, std::vector<uint8_t>> tb_blocks;
    std::unordered_map<size_t, FidScanBlock> fid_blocks;
};

std::optional<RangePredicateSpec> parse_single_numeric_range_predicate(
    const std::string& expression,
    std::string* reason) {
    std::vector<filter_expr::detail::Token> tokens = filter_expr::detail::tokenize(expression);
    filter_expr::detail::Parser parser(std::move(tokens));
    std::unique_ptr<filter_expr::Node> root = parser.parse();
    if (root == nullptr) {
        if (reason != nullptr) {
            *reason = "parsed filter expression is empty";
        }
        return std::nullopt;
    }

    auto parse_number = [&](const filter_expr::Literal& lit, double* out) -> bool {
        if (lit.is_number) {
            *out = lit.number;
            return true;
        }
        return filter_expr::detail::try_parse_double(lit.text, out);
    };

    auto parse_compare = [&](const filter_expr::Node* n, RangePredicateSpec* out) -> bool {
        if (n == nullptr || n->kind != filter_expr::Node::Kind::Compare) {
            return false;
        }
        double rhs = 0.0;
        if (!parse_number(n->literal, &rhs)) {
            return false;
        }
        out->field = n->field;
        switch (n->compare_op) {
            case filter_expr::CompareOp::Gt:
                out->low = rhs;
                out->low_inclusive = false;
                out->high = std::numeric_limits<double>::infinity();
                out->high_inclusive = true;
                return true;
            case filter_expr::CompareOp::Ge:
                out->low = rhs;
                out->low_inclusive = true;
                out->high = std::numeric_limits<double>::infinity();
                out->high_inclusive = true;
                return true;
            case filter_expr::CompareOp::Lt:
                out->low = -std::numeric_limits<double>::infinity();
                out->low_inclusive = true;
                out->high = rhs;
                out->high_inclusive = false;
                return true;
            case filter_expr::CompareOp::Le:
                out->low = -std::numeric_limits<double>::infinity();
                out->low_inclusive = true;
                out->high = rhs;
                out->high_inclusive = true;
                return true;
            default:
                return false;
        }
    };

    if (root->kind == filter_expr::Node::Kind::Between) {
        RangePredicateSpec out;
        double lo = 0.0;
        double hi = 0.0;
        if (!parse_number(root->lower, &lo) || !parse_number(root->upper, &hi)) {
            if (reason != nullptr) {
                *reason = "BETWEEN bounds must be numeric";
            }
            return std::nullopt;
        }
        out.field = root->field;
        out.low = lo;
        out.high = hi;
        out.low_inclusive = true;
        out.high_inclusive = true;
        return out;
    }

    if (root->kind == filter_expr::Node::Kind::Compare) {
        RangePredicateSpec out;
        if (!parse_compare(root.get(), &out)) {
            if (reason != nullptr) {
                *reason = "supported compare operators are >, >=, <, <=";
            }
            return std::nullopt;
        }
        return out;
    }

    if (root->kind == filter_expr::Node::Kind::Logical && root->logical_op == filter_expr::LogicalOp::And) {
        RangePredicateSpec lhs;
        RangePredicateSpec rhs;
        if (!parse_compare(root->left.get(), &lhs) || !parse_compare(root->right.get(), &rhs)) {
            if (reason != nullptr) {
                *reason = "AND form must be (field >= lo AND field <= hi)-style compares";
            }
            return std::nullopt;
        }
        if (lhs.field != rhs.field) {
            if (reason != nullptr) {
                *reason = "AND range compares must reference the same field";
            }
            return std::nullopt;
        }

        RangePredicateSpec out;
        out.field = lhs.field;
        if (lhs.low > rhs.low) {
            out.low = lhs.low;
            out.low_inclusive = lhs.low_inclusive;
        } else if (rhs.low > lhs.low) {
            out.low = rhs.low;
            out.low_inclusive = rhs.low_inclusive;
        } else {
            out.low = lhs.low;
            out.low_inclusive = lhs.low_inclusive && rhs.low_inclusive;
        }

        if (lhs.high < rhs.high) {
            out.high = lhs.high;
            out.high_inclusive = lhs.high_inclusive;
        } else if (rhs.high < lhs.high) {
            out.high = rhs.high;
            out.high_inclusive = rhs.high_inclusive;
        } else {
            out.high = lhs.high;
            out.high_inclusive = lhs.high_inclusive && rhs.high_inclusive;
        }
        return out;
    }

    if (reason != nullptr) {
        *reason = "supported range shapes: BETWEEN, single compare, or AND of two compares on one field";
    }
    return std::nullopt;
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

AsyncRangeRuntime build_async_range_runtime(
    const Args& args,
    const compass_iaa_filter::CompassIaaFilterEngine& engine) {
    AsyncRangeRuntime runtime;
    std::string parse_reason;
    const std::optional<RangePredicateSpec> spec =
        parse_single_numeric_range_predicate(args.filter_expression, &parse_reason);
    if (!spec.has_value()) {
        throw std::runtime_error(
            "Unsupported filter for async range mode: " + args.filter_expression +
            ". Accepted: single-field numeric BETWEEN / > / >= / < / <= / AND-combined range. Details: " +
            parse_reason);
    }

    const compass_iaa_filter::ManifestData& manifest = engine.manifest();
    const int nfilters = manifest.nfilters;
    if (nfilters <= 0 || nfilters > static_cast<int>(compass_iaa_filter::kMaxBuckets)) {
        throw std::runtime_error("Manifest nfilters is out of range for async range runtime");
    }

    auto attr_it = std::find_if(
        manifest.attributes.begin(),
        manifest.attributes.end(),
        [&](const compass_iaa_filter::ManifestAttribute& attr) {
            return attr.key == spec->field;
        });
    if (attr_it == manifest.attributes.end()) {
        throw std::runtime_error("Range filter field not found in manifest: " + spec->field);
    }

    const compass_iaa_filter::ManifestAttribute& attr = *attr_it;
    if (!attr.numeric || attr.encoding != "numeric_minmax_quantized") {
        throw std::runtime_error(
            "Async range mode supports only numeric_minmax_quantized fields; field=" + attr.key);
    }
    const int usable_bins = derive_usable_bins(nfilters, attr.used_bins);
    if (usable_bins <= 0) {
        throw std::runtime_error("Derived usable_bins is invalid for async range runtime");
    }

    int bucket_lo = 0;
    int bucket_hi = usable_bins - 1;
    if (std::isfinite(spec->low)) {
        const int b = bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, spec->low);
        bucket_lo = spec->low_inclusive ? b : std::min(usable_bins, b + 1);
    }
    if (std::isfinite(spec->high)) {
        const int b = bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, spec->high);
        bucket_hi = spec->high_inclusive ? b : std::max(-1, b - 1);
    }

    runtime.field = spec->field;
    runtime.n_elements = manifest.n_elements;
    runtime.nfilters = nfilters;
    runtime.missing_bucket = nfilters - 1;
    runtime.usable_bins = usable_bins;
    runtime.min_value = attr.min_value;
    runtime.max_value = attr.max_value;
    runtime.low = spec->low;
    runtime.high = spec->high;
    runtime.low_bounded = std::isfinite(spec->low);
    runtime.high_bounded = std::isfinite(spec->high);
    runtime.low_inclusive = spec->low_inclusive;
    runtime.high_inclusive = spec->high_inclusive;
    if (bucket_lo > bucket_hi) {
        runtime.empty_result = true;
        runtime.bucket_low = 0;
        runtime.bucket_high = 0;
        return runtime;
    }

    if (bucket_lo < 0 || bucket_hi >= usable_bins) {
        throw std::runtime_error("Computed bucket range is out of bounds for async range runtime");
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
    for (size_t node_id = 0; node_id < manifest.n_elements; ++node_id) {
        bool matches = false;
        for (int bucket = bucket_lo; bucket <= bucket_hi; ++bucket) {
            const size_t bucket_byte = static_cast<size_t>(bucket) / 8;
            const uint8_t bucket_mask = static_cast<uint8_t>(1u << (bucket % 8));
            const size_t offset = node_id * compass_iaa_filter::kTbBytesPerNode + bucket_byte;
            if ((tb_raw[offset] & bucket_mask) != 0) {
                matches = true;
                break;
            }
        }
        if (matches) {
            tb_bucket_bits[node_id / 8] |= static_cast<uint8_t>(1u << (node_id % 8));
        }
    }

    std::unique_ptr<compass_iaa_filter::detail::QplJobHandle> compressor =
        compass_iaa_filter::detail::make_hardware_job("async range runtime compressor");
    runtime.fid_storage = compass_iaa_filter::detail::compress_to_iaa_blocks(
        fid_raw,
        args.fid_block_size_bytes,
        compressor.get());
    runtime.tb_storage = compass_iaa_filter::detail::compress_to_iaa_blocks(
        tb_bucket_bits,
        args.tb_block_size_bytes,
        compressor.get());

    runtime.empty_result = false;
    runtime.bucket_low = static_cast<uint8_t>(bucket_lo);
    runtime.bucket_high = static_cast<uint8_t>(bucket_hi);
    return runtime;
}

class AsyncJobRing {
public:
    AsyncJobRing(size_t queue_size = 128, size_t wait_batch = 64)
        : queue_size_(queue_size), wait_batch_(wait_batch) {
        if (queue_size_ == 0 || wait_batch_ == 0 || wait_batch_ > queue_size_) {
            throw std::runtime_error("Invalid AsyncJobRing configuration");
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
        slot.output.assign(output_capacity, 0);

        setup_fn(slot.job, slot.output);
        const qpl_status submit_status = qpl_submit_job(slot.job);
        if (submit_status != QPL_STS_OK) {
            throw std::runtime_error(
                "AsyncJobRing: qpl_submit_job failed with status " +
                std::to_string(static_cast<int>(submit_status)));
        }

        slot.submit_time = std::chrono::steady_clock::now();
        slot.complete = std::forward<CompleteFn>(complete_fn);
        pending_slots_.push_back(slot_id);

        if (pending_slots_.size() >= queue_size_) {
            wait_oldest(wait_batch_);
        }
    }

    void flush() {
        wait_oldest(pending_slots_.size());
    }

private:
    struct Slot {
        std::unique_ptr<uint8_t[]> job_storage;
        qpl_job* job = nullptr;
        std::vector<uint8_t> output;
        std::chrono::steady_clock::time_point submit_time;
        std::function<void(qpl_job*, std::vector<uint8_t>&, uint64_t)> complete;
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
            const auto done_time = std::chrono::steady_clock::now();
            if (wait_status != QPL_STS_OK) {
                throw std::runtime_error(
                    "AsyncJobRing: qpl_wait_job failed with status " +
                    std::to_string(static_cast<int>(wait_status)));
            }

            const uint64_t elapsed_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(done_time - slot.submit_time).count());
            if (slot.complete) {
                slot.complete(slot.job, slot.output, elapsed_ns);
            }
            slot.complete = nullptr;
            slot.output.clear();
            free_slots_.push_back(slot_id);
        }
    }

    size_t queue_size_ = 128;
    size_t wait_batch_ = 64;
    std::vector<Slot> slots_;
    std::deque<size_t> free_slots_;
    std::deque<size_t> pending_slots_;
};

void submit_tb_prefetch_jobs(
    AsyncJobRing* ring,
    const AsyncRangeRuntime& runtime,
    AsyncRangeQueryCache* cache,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (ring == nullptr || cache == nullptr) {
        throw std::runtime_error("TB prefetch received null pointer");
    }
    for (size_t block_id = 0; block_id < runtime.tb_storage.block_count(); ++block_id) {
        if (cache->tb_blocks.find(block_id) != cache->tb_blocks.end()) {
            continue;
        }
        const uint32_t raw_len = runtime.tb_storage.raw_block_sizes[block_id];
        if (raw_len == 0) {
            cache->tb_blocks.emplace(block_id, std::vector<uint8_t>());
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
            [cache, metrics, block_id, raw_len](qpl_job* job, std::vector<uint8_t>& out, uint64_t elapsed_ns) {
                out.resize(static_cast<size_t>(job->total_out));
                cache->tb_blocks[block_id] = out;
                if (metrics != nullptr) {
                    metrics->iaa_decompress_time_ns += elapsed_ns;
                    ++metrics->tb_blocks_decompressed;
                    metrics->tb_bytes_decompressed += static_cast<uint64_t>(raw_len);
                }
            });
    }
}

void submit_fid_scan_jobs(
    AsyncJobRing* ring,
    const AsyncRangeRuntime& runtime,
    const std::unordered_set<size_t>& block_ids,
    AsyncRangeQueryCache* cache,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (ring == nullptr || cache == nullptr) {
        throw std::runtime_error("FID async scan received null pointer");
    }

    for (size_t block_id : block_ids) {
        if (block_id >= runtime.fid_storage.block_count()) {
            continue;
        }
        if (cache->fid_blocks.find(block_id) != cache->fid_blocks.end()) {
            continue;
        }

        const uint32_t raw_len = runtime.fid_storage.raw_block_sizes[block_id];
        if (raw_len == 0) {
            cache->fid_blocks.emplace(block_id, FidScanBlock{});
            continue;
        }

        ring->submit(
            static_cast<size_t>(raw_len),
            [&](qpl_job* job, std::vector<uint8_t>& out) {
                job->op = qpl_op_scan_range;
                job->next_in_ptr = const_cast<uint8_t*>(runtime.fid_storage.compressed_blocks[block_id].data());
                job->available_in = static_cast<uint32_t>(runtime.fid_storage.compressed_blocks[block_id].size());
                job->next_out_ptr = out.data();
                job->available_out = raw_len;
                job->src1_bit_width = 8;
                job->out_bit_width = qpl_ow_nom;
                job->param_low = runtime.bucket_low;
                job->param_high = runtime.bucket_high;
                job->num_input_elements = raw_len;
                job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
            },
            [cache, metrics, block_id, raw_len](qpl_job* job, std::vector<uint8_t>& out, uint64_t elapsed_ns) {
                out.resize(static_cast<size_t>(job->total_out));
                FidScanBlock block;
                block.matches = out;
                block.input_elements = raw_len;
                block.byte_per_element = (job->total_out >= raw_len);
                cache->fid_blocks[block_id] = std::move(block);
                if (metrics != nullptr) {
                    metrics->iaa_decompress_time_ns += elapsed_ns;
                    ++metrics->fid_blocks_decompressed;
                    metrics->fid_bytes_decompressed += static_cast<uint64_t>(raw_len);
                }
            });
    }
}

bool tb_match_node(
    const AsyncRangeRuntime& runtime,
    AsyncRangeQueryCache* cache,
    size_t node_id,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr || runtime.tb_storage.block_size == 0 || node_id >= runtime.n_elements) {
        return false;
    }
    const size_t byte_offset = node_id / 8;
    const size_t block_id = byte_offset / runtime.tb_storage.block_size;
    const size_t in_block = byte_offset % runtime.tb_storage.block_size;

    auto it = cache->tb_blocks.find(block_id);
    if (it == cache->tb_blocks.end() || in_block >= it->second.size()) {
        return false;
    }
    if (metrics != nullptr) {
        ++metrics->tb_cache_hits;
    }
    const uint8_t byte = it->second[in_block];
    return ((byte >> (node_id % 8)) & 1u) != 0;
}

bool fid_match_node(
    const AsyncRangeRuntime& runtime,
    AsyncRangeQueryCache* cache,
    size_t node_id,
    compass_iaa_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr || runtime.fid_storage.block_size == 0 || node_id >= runtime.n_elements) {
        return false;
    }

    const size_t block_id = node_id / runtime.fid_storage.block_size;
    const size_t in_block = node_id % runtime.fid_storage.block_size;
    auto it = cache->fid_blocks.find(block_id);
    if (it == cache->fid_blocks.end()) {
        return false;
    }
    if (metrics != nullptr) {
        ++metrics->fid_cache_hits;
    }

    const FidScanBlock& block = it->second;
    if (block.byte_per_element) {
        return in_block < block.matches.size() && block.matches[in_block] != 0;
    }

    const size_t byte_idx = in_block / 8;
    if (byte_idx >= block.matches.size()) {
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
std::vector<std::pair<DistT, hnswlib::labeltype>> search_with_async_range_filter(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* query_data,
    size_t k,
    size_t ef,
    const AsyncRangeRuntime& runtime,
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

    hnswlib::tableint curr_obj = index.enterpoint_node_;
    if (static_cast<int>(curr_obj) == -1) {
        return result;
    }

    DistT curdist = index.fstdistfunc_(query_data, index.getDataByInternalId(curr_obj), index.dist_func_param_);

    AsyncRangeQueryCache cache;
    cache.tb_blocks.reserve(runtime.tb_storage.block_count());

    AsyncJobRing ring(128, 64);
    submit_tb_prefetch_jobs(&ring, runtime, &cache, &call_stats->decomp);

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
    ring.flush();

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

        std::vector<hnswlib::tableint> neighbor_ids;
        neighbor_ids.reserve(size);
        std::unordered_set<size_t> blocks_to_submit;
        blocks_to_submit.reserve(size);

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

            const size_t block_id = static_cast<size_t>(candidate_id) / runtime.fid_storage.block_size;
            if (cache.fid_blocks.find(block_id) == cache.fid_blocks.end()) {
                blocks_to_submit.insert(block_id);
            }
        }

        submit_fid_scan_jobs(&ring, runtime, blocks_to_submit, &cache, &call_stats->decomp);

        std::vector<DistT> dists(neighbor_ids.size(), DistT());
        for (size_t idx = 0; idx < neighbor_ids.size(); ++idx) {
            dists[idx] = index.fstdistfunc_(
                query_data,
                index.getDataByInternalId(neighbor_ids[idx]),
                index.dist_func_param_);
        }

        ring.flush();

        for (size_t idx = 0; idx < neighbor_ids.size(); ++idx) {
            const hnswlib::tableint candidate_id = neighbor_ids[idx];
            const DistT dist = dists[idx];

            const auto t0 = std::chrono::steady_clock::now();
            const bool fid_match = fid_match_node(runtime, &cache, static_cast<size_t>(candidate_id), &call_stats->decomp);
            const bool tb_match = tb_match_node(runtime, &cache, static_cast<size_t>(candidate_id), &call_stats->decomp);
            const bool traversal_allowed = fid_match || tb_match;
            const auto t1 = std::chrono::steady_clock::now();
            ++call_stats->filter_eval_calls;
            call_stats->filter_eval_time_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

            if (!traversal_allowed) {
                continue;
            }

            if (top_candidates.size() < ef || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);

                bool result_allowed = false;
                if (!index.isMarkedDeleted(candidate_id)) {
                    const auto r0 = std::chrono::steady_clock::now();
                    result_allowed = fid_match;
                    const auto r1 = std::chrono::steady_clock::now();
                    ++call_stats->filter_eval_calls;
                    call_stats->filter_eval_time_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(r1 - r0).count());
                }
                if (!index.isMarkedDeleted(candidate_id) && result_allowed) {
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
    const AsyncRangeRuntime* async_runtime) {
    SpaceT space(static_cast<size_t>(dim));
    hnswlib::HierarchicalNSW<DistT> index(&space, args.graph_path);
    index.setEf(static_cast<size_t>(args.ef));

    if (engine.num_elements() != index.getCurrentElementCount()) {
        throw std::runtime_error(
            "FID/TB manifest n_elements does not match graph element count: manifest=" +
            std::to_string(engine.num_elements()) +
            ", graph=" + std::to_string(index.getCurrentElementCount()));
    }

    if (async_runtime == nullptr || !async_runtime->enabled) {
        throw std::runtime_error("Async range runtime is not initialized");
    }

    RunStats stats;
    stats.execution_mode = "async_range";
    const size_t total_elements = index.getCurrentElementCount();

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
            << "fid_cache_hits,tb_cache_hits\n";
    }

    const bool capture_per_query = per_query_out.is_open();
    if (capture_per_query) {
        stats.per_query_metrics.reserve(query_count);
    }

    size_t returned_results = 0;
    const auto loop_start = std::chrono::steady_clock::now();

    for (size_t qid = 0; qid < query_count; ++qid) {
        const QueryT* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);

        SearchCallStats call_stats;

        const auto search_start = std::chrono::steady_clock::now();
        std::vector<std::pair<DistT, hnswlib::labeltype>> result =
            search_with_async_range_filter<DistT, QueryT>(
                index,
                qptr,
                static_cast<size_t>(args.k),
                static_cast<size_t>(args.ef),
                *async_runtime,
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
                << call_stats.decomp.tb_cache_hits
                << '\n';
        }
    }

    const auto loop_end = std::chrono::steady_clock::now();

    stats.query_count = query_count;
    stats.search_loop_time_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start).count());
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
    const AsyncRangeRuntime& runtime) {
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

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "compass_search_w_iaa summary\n";
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
    oss << "k: " << args.k << ", ef: " << args.ef << "\n";
    oss << "filter: " << expr.source() << "\n";
    oss << "execution_mode: " << stats.execution_mode << "\n";
    oss << "range_bucket_mapping_status: " << (runtime.empty_result ? "empty" : "ok") << "\n";
    oss << "range_bucket_mapping_field: " << runtime.field << "\n";
    oss << "range_bucket_mapping_nfilters: " << runtime.nfilters << "\n";
    oss << "range_bucket_mapping_missing_bucket: " << runtime.missing_bucket << "\n";
    oss << "range_bucket_mapping_usable_bins: " << runtime.usable_bins << "\n";
    oss << "range_bucket_mapping_min_value: " << runtime.min_value << "\n";
    oss << "range_bucket_mapping_max_value: " << runtime.max_value << "\n";
    if (runtime.low_bounded) {
        oss << "range_bucket_mapping_low: " << runtime.low << "\n";
        oss << "range_bucket_mapping_low_inclusive: "
            << (runtime.low_inclusive ? "true" : "false") << "\n";
    }
    if (runtime.high_bounded) {
        oss << "range_bucket_mapping_high: " << runtime.high << "\n";
        oss << "range_bucket_mapping_high_inclusive: "
            << (runtime.high_inclusive ? "true" : "false") << "\n";
    }
    oss << "range_bucket_mapping_empty_result: " << (runtime.empty_result ? "true" : "false") << "\n";
    oss << "range_bucket_mapping_bucket_low: " << static_cast<int>(runtime.bucket_low) << "\n";
    oss << "range_bucket_mapping_bucket_high: " << static_cast<int>(runtime.bucket_high) << "\n";
    oss << "range_bucket_mapping_selected_bucket_count: "
        << (runtime.empty_result ? 0 : (static_cast<int>(runtime.bucket_high) - static_cast<int>(runtime.bucket_low) + 1))
        << "\n";

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

    if (stats.queries_with_enns_lt_k > 0) {
        oss << "warning: filtering condition is too restrictive for k on "
            << stats.queries_with_enns_lt_k << " queries\n";
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

        const VecFileInfo query_info = filter_search_io::inspect_vector_file(args.query_path);
        const IndexFileInfo index_info = filter_search_io::inspect_hnsw_index_file(args.graph_path);

        filter_expr::Expression expr(args.filter_expression);
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
        const AsyncRangeRuntime async_runtime = build_async_range_runtime(args, engine);
        std::cout
            << "[RANGE_BUCKET_MAP] field=" << async_runtime.field
            << " status=" << (async_runtime.empty_result ? "empty" : "ok")
            << " low="
            << (async_runtime.low_bounded ? std::to_string(async_runtime.low) : std::string("-inf"))
            << " high="
            << (async_runtime.high_bounded ? std::to_string(async_runtime.high) : std::string("+inf"))
            << " min_value=" << async_runtime.min_value
            << " max_value=" << async_runtime.max_value
            << " usable_bins=" << async_runtime.usable_bins
            << " missing_bucket=" << async_runtime.missing_bucket
            << " bucket_low=" << static_cast<int>(async_runtime.bucket_low)
            << " bucket_high=" << static_cast<int>(async_runtime.bucket_high)
            << " selected_bucket_count="
            << (async_runtime.empty_result
                ? 0
                : (static_cast<int>(async_runtime.bucket_high) - static_cast<int>(async_runtime.bucket_low) + 1))
            << "\n";

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
            async_runtime);
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
