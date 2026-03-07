#include "../../../hnswlib/filter_search_hnswlib/hnswlib.h"
#include "../../../hnswlib/filter_search_hnswlib_with_lz4/compass_lz4_filter.h"

#include "../filter_expr.h"
#include "../io_utils.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
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
constexpr size_t kDefaultLz4FidBlockSizeBytes = 8192 * 8;
constexpr size_t kDefaultLz4TbBlockSizeBytes = 8192 * 128;

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
    size_t fid_block_size_bytes = kDefaultLz4FidBlockSizeBytes;
    size_t tb_block_size_bytes = kDefaultLz4TbBlockSizeBytes;

    // Backward-compatible optional flags. Not used in the LZ4 FID/TB path.
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

    uint64_t lz4_decompress_time_ns = 0;
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

    uint64_t lz4_decompress_time_ns = 0;
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
    size_t resolved_ef = 0;
    double effective_break_factor = 0.0;

    std::vector<QueryMetrics> per_query_metrics;
};

struct SearchCallStats {
    uint64_t filter_eval_calls = 0;
    uint64_t filter_eval_time_ns = 0;
    compass_lz4_filter::QueryDecompressionMetrics decomp;
    uint64_t fid_tb_mismatch_count = 0;
    uint64_t fid_tb_mismatch_log_capped = 0;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --dataset-type <sift|laion|hnm>"
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

    if (args.dataset_type != "sift" && args.dataset_type != "laion" && args.dataset_type != "hnm") {
        throw std::runtime_error("--dataset-type must be one of: sift, laion, hnm");
    }

    ensure_readable_file(args.graph_path, "--graph");
    ensure_readable_file(args.query_path, "--query");
    ensure_readable_file(args.fidtb_manifest, "--fidtb-manifest");

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

struct SequentialEqRuntime {
    struct Leaf {
        bool active = false;
        bool always_false = false;
        std::string field;
        int target_bucket = -1;
        size_t n_elements = 0;
        compass_lz4_filter::detail::Lz4BlockStorage fid_storage;
        compass_lz4_filter::detail::Lz4BlockStorage tb_storage;
    };

    bool enabled = false;
    bool empty_result = false;
    filter_expr::LogicalOp logical_op = filter_expr::LogicalOp::And;
    std::string left_field;
    std::string right_field;
    std::array<Leaf, 2> leaves;
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

template <typename ElemT>
uint64_t total_compressed_block_bytes(const std::vector<std::vector<ElemT>>& blocks) {
    uint64_t total = 0;
    for (const auto& block : blocks) {
        total += safe_size_t_to_u64(block.size());
    }
    return total;
}

CompressionStats compute_compression_stats(const SequentialEqRuntime& runtime) {
    CompressionStats out;
    for (const SequentialEqRuntime::Leaf& leaf : runtime.leaves) {
        if (!leaf.active || leaf.always_false) {
            continue;
        }
        out.fid_raw_bytes += safe_size_t_to_u64(leaf.fid_storage.raw_size);
        out.fid_compressed_bytes += total_compressed_block_bytes(leaf.fid_storage.compressed_blocks);
        out.tb_raw_bytes += safe_size_t_to_u64(leaf.tb_storage.raw_size);
        out.tb_compressed_bytes += total_compressed_block_bytes(leaf.tb_storage.compressed_blocks);
    }
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

struct SequentialEqQueryCache {
    struct Leaf {
        std::vector<std::vector<uint8_t>> tb_blocks;
        std::vector<uint8_t> tb_ready;
        std::vector<std::vector<uint8_t>> fid_blocks;
        std::vector<uint8_t> fid_ready;
    };

    std::array<Leaf, 2> leaves;
};

struct SequentialEqSearchWorkspace {
    std::vector<hnswlib::tableint> neighbor_ids;
};

struct LeafExactSpec {
    std::string field;
    filter_expr::Literal literal;
};

struct TwoLeafExactSpec {
    filter_expr::LogicalOp logical_op = filter_expr::LogicalOp::And;
    LeafExactSpec left;
    LeafExactSpec right;
};

std::unique_ptr<filter_expr::Node> parse_filter_ast(const std::string& expression) {
    std::vector<filter_expr::detail::Token> tokens = filter_expr::detail::tokenize(expression);
    filter_expr::detail::Parser parser(std::move(tokens));
    return parser.parse();
}

std::string logical_op_name(filter_expr::LogicalOp op) {
    return (op == filter_expr::LogicalOp::And) ? "AND" : "OR";
}

LeafExactSpec parse_eq_leaf_or_throw(const filter_expr::Node* node) {
    if (node == nullptr ||
        node->kind != filter_expr::Node::Kind::Compare ||
        node->compare_op != filter_expr::CompareOp::Eq) {
        throw std::runtime_error(
            "multiple_exact_match requires equality leaves only: <field1> == <value1> "
            "AND/OR <field2> == <value2>");
    }
    LeafExactSpec out;
    out.field = node->field;
    out.literal = node->literal;
    return out;
}

TwoLeafExactSpec parse_two_leaf_exact_expression(const std::string& expression) {
    std::unique_ptr<filter_expr::Node> root = parse_filter_ast(expression);
    if (root == nullptr) {
        throw std::runtime_error("Filter expression is empty");
    }
    if (root->kind != filter_expr::Node::Kind::Logical ||
        (root->logical_op != filter_expr::LogicalOp::And &&
         root->logical_op != filter_expr::LogicalOp::Or)) {
        throw std::runtime_error(
            "multiple_exact_match requires top-level AND/OR between two equality leaves");
    }

    LeafExactSpec left = parse_eq_leaf_or_throw(root->left.get());
    LeafExactSpec right = parse_eq_leaf_or_throw(root->right.get());
    if (left.field == right.field) {
        throw std::runtime_error(
            "multiple_exact_match requires two different attribute fields");
    }

    TwoLeafExactSpec out;
    out.logical_op = root->logical_op;
    out.left = std::move(left);
    out.right = std::move(right);
    return out;
}

std::string literal_to_filter_text(const filter_expr::Literal& lit) {
    if (lit.is_number) {
        return lit.text.empty() ? std::to_string(lit.number) : lit.text;
    }
    std::string escaped;
    escaped.reserve(lit.text.size() + 8);
    for (char c : lit.text) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return "\"" + escaped + "\"";
}

std::string build_or_traversal_expression(const TwoLeafExactSpec& spec) {
    return spec.left.field + " == " + literal_to_filter_text(spec.left.literal) +
        " OR " + spec.right.field + " == " + literal_to_filter_text(spec.right.literal);
}

bool is_single_field_expression(const filter_expr::Node* node, std::string* field_out) {
    if (node == nullptr) {
        return false;
    }
    if (node->kind == filter_expr::Node::Kind::Logical) {
        return is_single_field_expression(node->left.get(), field_out) &&
               is_single_field_expression(node->right.get(), field_out);
    }
    if (field_out == nullptr) {
        return false;
    }
    if (field_out->empty()) {
        *field_out = node->field;
        return true;
    }
    return *field_out == node->field;
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

bool parse_numeric_literal(const filter_expr::Literal& lit, double* out) {
    if (lit.is_number) {
        if (out != nullptr) {
            *out = lit.number;
        }
        return true;
    }
    return filter_expr::detail::try_parse_double(lit.text, out);
}

double bucket_representative(int usable_bins, double min_value, double max_value, int bucket) {
    if (usable_bins <= 1 || max_value <= min_value) {
        return min_value;
    }
    const double ratio = static_cast<double>(bucket) / static_cast<double>(usable_bins - 1);
    return min_value + ratio * (max_value - min_value);
}

std::bitset<compass_lz4_filter::kMaxBuckets> all_data_buckets(int usable_bins) {
    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    const int limit = std::min(usable_bins, static_cast<int>(compass_lz4_filter::kMaxBuckets));
    for (int b = 0; b < limit; ++b) {
        out.set(static_cast<size_t>(b));
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_compare_numeric_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    filter_expr::CompareOp op,
    const filter_expr::Literal& lit) {
    double rhs = 0.0;
    if (!parse_numeric_literal(lit, &rhs)) {
        throw std::runtime_error(
            "Numeric predicate has non-numeric literal for field '" + attr.key + "': " + lit.text);
    }

    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    if (op == filter_expr::CompareOp::Eq) {
        out.set(static_cast<size_t>(bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, rhs)));
        return out;
    }
    if (op == filter_expr::CompareOp::Ne) {
        out = all_data_buckets(usable_bins);
        out.reset(static_cast<size_t>(bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, rhs)));
        return out;
    }

    const int limit = std::min(usable_bins, static_cast<int>(compass_lz4_filter::kMaxBuckets));
    for (int b = 0; b < limit; ++b) {
        const double rep = bucket_representative(usable_bins, attr.min_value, attr.max_value, b);
        if (filter_expr::detail::compare_numeric(rep, rhs, op)) {
            out.set(static_cast<size_t>(b));
        }
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_between_numeric_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    const filter_expr::Literal& lo,
    const filter_expr::Literal& hi) {
    double lo_v = 0.0;
    double hi_v = 0.0;
    if (!parse_numeric_literal(lo, &lo_v) || !parse_numeric_literal(hi, &hi_v)) {
        throw std::runtime_error("Numeric BETWEEN has non-numeric bounds for field '" + attr.key + "'");
    }
    if (lo_v > hi_v) {
        return std::bitset<compass_lz4_filter::kMaxBuckets>();
    }

    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    const int limit = std::min(usable_bins, static_cast<int>(compass_lz4_filter::kMaxBuckets));
    for (int b = 0; b < limit; ++b) {
        const double rep = bucket_representative(usable_bins, attr.min_value, attr.max_value, b);
        if (rep >= lo_v && rep <= hi_v) {
            out.set(static_cast<size_t>(b));
        }
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_in_numeric_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    const std::vector<filter_expr::Literal>& list) {
    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    for (const filter_expr::Literal& lit : list) {
        double v = 0.0;
        if (!parse_numeric_literal(lit, &v)) {
            throw std::runtime_error(
                "Numeric IN has non-numeric literal for field '" + attr.key + "': " + lit.text);
        }
        out.set(static_cast<size_t>(bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, v)));
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_compare_categorical_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    filter_expr::CompareOp op,
    const filter_expr::Literal& lit) {
    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    auto bucket_it = attr.category_map.find(lit.text);
    if (op == filter_expr::CompareOp::Eq) {
        if (bucket_it != attr.category_map.end() &&
            bucket_it->second >= 0 &&
            bucket_it->second < usable_bins) {
            out.set(static_cast<size_t>(bucket_it->second));
        }
        return out;
    }
    if (op == filter_expr::CompareOp::Ne) {
        out = all_data_buckets(usable_bins);
        if (bucket_it != attr.category_map.end() &&
            bucket_it->second >= 0 &&
            bucket_it->second < usable_bins) {
            out.reset(static_cast<size_t>(bucket_it->second));
        }
        return out;
    }

    for (const auto& kv : attr.category_map) {
        const std::string& key = kv.first;
        const int bucket = kv.second;
        if (bucket < 0 || bucket >= usable_bins || bucket >= static_cast<int>(compass_lz4_filter::kMaxBuckets)) {
            continue;
        }
        if (filter_expr::detail::compare_string(key, lit.text, op)) {
            out.set(static_cast<size_t>(bucket));
        }
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_between_categorical_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    const filter_expr::Literal& lo,
    const filter_expr::Literal& hi) {
    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    if (lo.text > hi.text) {
        return out;
    }
    for (const auto& kv : attr.category_map) {
        const std::string& key = kv.first;
        const int bucket = kv.second;
        if (bucket < 0 || bucket >= usable_bins || bucket >= static_cast<int>(compass_lz4_filter::kMaxBuckets)) {
            continue;
        }
        if (key >= lo.text && key <= hi.text) {
            out.set(static_cast<size_t>(bucket));
        }
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_in_categorical_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    const std::vector<filter_expr::Literal>& list) {
    std::bitset<compass_lz4_filter::kMaxBuckets> out;
    for (const filter_expr::Literal& lit : list) {
        auto it = attr.category_map.find(lit.text);
        if (it == attr.category_map.end()) {
            continue;
        }
        if (it->second < 0 || it->second >= usable_bins || it->second >= static_cast<int>(compass_lz4_filter::kMaxBuckets)) {
            continue;
        }
        out.set(static_cast<size_t>(it->second));
    }
    return out;
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_leaf_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    const filter_expr::Node* node) {
    if (node == nullptr) {
        throw std::runtime_error("Internal error: null filter node");
    }
    if (node->kind == filter_expr::Node::Kind::Compare) {
        if (attr.numeric) {
            return compile_compare_numeric_buckets(attr, usable_bins, node->compare_op, node->literal);
        }
        return compile_compare_categorical_buckets(attr, usable_bins, node->compare_op, node->literal);
    }
    if (node->kind == filter_expr::Node::Kind::Between) {
        if (attr.numeric) {
            return compile_between_numeric_buckets(attr, usable_bins, node->lower, node->upper);
        }
        return compile_between_categorical_buckets(attr, usable_bins, node->lower, node->upper);
    }
    if (node->kind == filter_expr::Node::Kind::In) {
        if (attr.numeric) {
            return compile_in_numeric_buckets(attr, usable_bins, node->list);
        }
        return compile_in_categorical_buckets(attr, usable_bins, node->list);
    }
    throw std::runtime_error("Unexpected logical node passed to compile_leaf_buckets");
}

std::bitset<compass_lz4_filter::kMaxBuckets> compile_node_buckets(
    const compass_lz4_filter::ManifestAttribute& attr,
    int usable_bins,
    const filter_expr::Node* node) {
    if (node == nullptr) {
        throw std::runtime_error("Parsed filter expression node is null");
    }
    if (node->kind == filter_expr::Node::Kind::Logical) {
        const std::bitset<compass_lz4_filter::kMaxBuckets> lhs =
            compile_node_buckets(attr, usable_bins, node->left.get());
        const std::bitset<compass_lz4_filter::kMaxBuckets> rhs =
            compile_node_buckets(attr, usable_bins, node->right.get());
        if (node->logical_op == filter_expr::LogicalOp::And) {
            return lhs & rhs;
        }
        return lhs | rhs;
    }
    return compile_leaf_buckets(attr, usable_bins, node);
}

SequentialEqRuntime build_sequential_eq_runtime(
    const Args& args,
    const compass_lz4_filter::CompassLz4FilterEngine& engine) {
    SequentialEqRuntime runtime;
    const TwoLeafExactSpec spec = parse_two_leaf_exact_expression(args.filter_expression);
    const compass_lz4_filter::ManifestData& manifest = engine.manifest();
    const int nfilters = manifest.nfilters;
    if (nfilters <= 0 || nfilters > static_cast<int>(compass_lz4_filter::kMaxBuckets)) {
        return runtime;
    }

    auto find_attribute = [&](const std::string& field)
        -> const compass_lz4_filter::ManifestAttribute& {
        auto it = std::find_if(
            manifest.attributes.begin(),
            manifest.attributes.end(),
            [&](const compass_lz4_filter::ManifestAttribute& attr) {
                return attr.key == field;
            });
        if (it == manifest.attributes.end()) {
            throw std::runtime_error(
                "Field '" + field + "' was not found in FID/TB manifest");
        }
        return *it;
    };

    auto resolve_eq_bucket = [&](const compass_lz4_filter::ManifestAttribute& attr,
                                 const filter_expr::Literal& lit) -> std::optional<int> {
        const int usable_bins = derive_usable_bins(nfilters, attr.used_bins);
        if (usable_bins <= 0) {
            return std::nullopt;
        }
        if (attr.numeric) {
            double value = 0.0;
            if (!parse_numeric_literal(lit, &value)) {
                throw std::runtime_error(
                    "Numeric field '" + attr.key + "' requires numeric literal");
            }
            return bucket_from_numeric(usable_bins, attr.min_value, attr.max_value, value);
        }

        auto bucket_it = attr.category_map.find(lit.text);
        if (bucket_it == attr.category_map.end()) {
            return std::nullopt;
        }
        if (bucket_it->second < 0 || bucket_it->second >= usable_bins) {
            return std::nullopt;
        }
        return bucket_it->second;
    };

    auto build_leaf_runtime = [&](const LeafExactSpec& leaf_spec) -> SequentialEqRuntime::Leaf {
        SequentialEqRuntime::Leaf leaf;
        leaf.field = leaf_spec.field;
        leaf.n_elements = manifest.n_elements;

        const compass_lz4_filter::ManifestAttribute& attr = find_attribute(leaf_spec.field);
        const std::optional<int> maybe_bucket = resolve_eq_bucket(attr, leaf_spec.literal);
        if (!maybe_bucket.has_value()) {
            leaf.active = true;
            leaf.always_false = true;
            return leaf;
        }

        const int target_bucket = *maybe_bucket;
        size_t fid_count = 0;
        std::vector<uint8_t> fid_raw = compass_lz4_filter::detail::read_payload_with_size_header(
            attr.fid_file,
            &fid_count,
            sizeof(uint8_t));
        if (fid_count != manifest.n_elements) {
            throw std::runtime_error(
                "FID element count mismatch while building sequential runtime for field '" + attr.key + "'");
        }

        size_t tb_count = 0;
        std::vector<uint8_t> tb_raw = compass_lz4_filter::detail::read_payload_with_size_header(
            attr.tb_file,
            &tb_count,
            compass_lz4_filter::kTbBytesPerNode);
        if (tb_count != manifest.n_elements) {
            throw std::runtime_error(
                "TB element count mismatch while building sequential runtime for field '" + attr.key + "'");
        }

        std::vector<uint8_t> tb_bucket_bits((manifest.n_elements + 7) / 8, 0);
        const size_t bucket_byte = static_cast<size_t>(target_bucket) / 8;
        const uint8_t bucket_mask = static_cast<uint8_t>(1u << (target_bucket % 8));
        for (size_t node_id = 0; node_id < manifest.n_elements; ++node_id) {
            const size_t offset = node_id * compass_lz4_filter::kTbBytesPerNode + bucket_byte;
            if ((tb_raw[offset] & bucket_mask) != 0) {
                tb_bucket_bits[node_id / 8] |= static_cast<uint8_t>(1u << (node_id % 8));
            }
        }

        leaf.target_bucket = target_bucket;
        leaf.fid_storage = compass_lz4_filter::detail::compress_to_lz4_blocks(
            fid_raw,
            args.fid_block_size_bytes);
        leaf.tb_storage = compass_lz4_filter::detail::compress_to_lz4_blocks(
            tb_bucket_bits,
            args.tb_block_size_bytes);
        leaf.active = true;
        leaf.always_false = false;
        return leaf;
    };

    runtime.logical_op = spec.logical_op;
    runtime.left_field = spec.left.field;
    runtime.right_field = spec.right.field;
    runtime.leaves[0] = build_leaf_runtime(spec.left);
    runtime.leaves[1] = build_leaf_runtime(spec.right);
    runtime.enabled = true;
    if (runtime.logical_op == filter_expr::LogicalOp::And) {
        runtime.empty_result = runtime.leaves[0].always_false || runtime.leaves[1].always_false;
    } else {
        runtime.empty_result = runtime.leaves[0].always_false && runtime.leaves[1].always_false;
    }
    return runtime;
}

SequentialEqQueryCache make_preallocated_query_cache(const SequentialEqRuntime& runtime) {
    SequentialEqQueryCache cache;
    for (size_t leaf_idx = 0; leaf_idx < runtime.leaves.size(); ++leaf_idx) {
        const SequentialEqRuntime::Leaf& leaf = runtime.leaves[leaf_idx];
        if (!leaf.active || leaf.always_false) {
            continue;
        }

        SequentialEqQueryCache::Leaf& leaf_cache = cache.leaves[leaf_idx];
        const size_t tb_blocks = leaf.tb_storage.block_count();
        leaf_cache.tb_blocks.resize(tb_blocks);
        leaf_cache.tb_ready.assign(tb_blocks, 0);
        for (size_t block_id = 0; block_id < tb_blocks; ++block_id) {
            leaf_cache.tb_blocks[block_id].assign(leaf.tb_storage.raw_block_sizes[block_id], 0);
        }

        const size_t fid_blocks = leaf.fid_storage.block_count();
        leaf_cache.fid_blocks.resize(fid_blocks);
        leaf_cache.fid_ready.assign(fid_blocks, 0);
        for (size_t block_id = 0; block_id < fid_blocks; ++block_id) {
            leaf_cache.fid_blocks[block_id].assign(leaf.fid_storage.raw_block_sizes[block_id], 0);
        }
    }
    return cache;
}

void decompress_lz4_block_into(
    const compass_lz4_filter::detail::Lz4BlockStorage& storage,
    size_t block_id,
    bool is_fid,
    std::vector<uint8_t>* output,
    compass_lz4_filter::QueryDecompressionMetrics* metrics) {
    if (output == nullptr) {
        throw std::runtime_error("LZ4 decompress destination buffer is null");
    }
    if (block_id >= storage.block_count()) {
        throw std::runtime_error("Block id out of range during decompression");
    }

    const size_t raw_len = static_cast<size_t>(storage.raw_block_sizes[block_id]);
    if (output->size() < raw_len) {
        throw std::runtime_error("Preallocated LZ4 output buffer is too small");
    }
    if (raw_len == 0) {
        return;
    }

    const int ret = LZ4_decompress_safe(
        storage.compressed_blocks[block_id].data(),
        reinterpret_cast<char*>(output->data()),
        static_cast<int>(storage.compressed_blocks[block_id].size()),
        static_cast<int>(raw_len));

    if (ret != static_cast<int>(raw_len)) {
        throw std::runtime_error("LZ4 decompression failed at block " + std::to_string(block_id));
    }

    if (metrics != nullptr) {
        if (is_fid) {
            ++metrics->fid_blocks_decompressed;
            metrics->fid_bytes_decompressed += static_cast<uint64_t>(raw_len);
        } else {
            ++metrics->tb_blocks_decompressed;
            metrics->tb_bytes_decompressed += static_cast<uint64_t>(raw_len);
        }
    }
}

void prefetch_tb_blocks(
    const SequentialEqRuntime::Leaf& leaf,
    SequentialEqQueryCache::Leaf* cache,
    compass_lz4_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr) {
        throw std::runtime_error("TB prefetch received null cache");
    }
    if (!leaf.active || leaf.always_false) {
        return;
    }
    for (size_t block_id = 0; block_id < leaf.tb_storage.block_count(); ++block_id) {
        if (block_id >= cache->tb_ready.size() || block_id >= cache->tb_blocks.size()) {
            throw std::runtime_error("TB query cache is not preallocated for all blocks");
        }
        if (cache->tb_ready[block_id] != 0) {
            continue;
        }
        decompress_lz4_block_into(
            leaf.tb_storage,
            block_id,
            false,
            &cache->tb_blocks[block_id],
            metrics);
        cache->tb_ready[block_id] = 1;
    }
}

void prefetch_fid_block(
    const SequentialEqRuntime::Leaf& leaf,
    size_t block_id,
    SequentialEqQueryCache::Leaf* cache,
    compass_lz4_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr) {
        throw std::runtime_error("FID prefetch received null cache");
    }
    if (!leaf.active || leaf.always_false) {
        return;
    }
    if (block_id >= leaf.fid_storage.block_count()) {
        return;
    }
    if (block_id >= cache->fid_ready.size() || block_id >= cache->fid_blocks.size()) {
        throw std::runtime_error("FID query cache is not preallocated for all blocks");
    }
    if (cache->fid_ready[block_id] != 0) {
        return;
    }
    decompress_lz4_block_into(
        leaf.fid_storage,
        block_id,
        true,
        &cache->fid_blocks[block_id],
        metrics);
    cache->fid_ready[block_id] = 1;
}

bool tb_match_node(
    const SequentialEqRuntime::Leaf& leaf,
    SequentialEqQueryCache::Leaf* cache,
    size_t node_id,
    compass_lz4_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr || !leaf.active || leaf.always_false || leaf.tb_storage.block_size == 0 ||
        node_id >= leaf.n_elements) {
        return false;
    }

    const size_t byte_offset = node_id / 8;
    const size_t block_id = byte_offset / leaf.tb_storage.block_size;
    const size_t in_block = byte_offset % leaf.tb_storage.block_size;
    if (block_id >= cache->tb_blocks.size() || block_id >= cache->tb_ready.size() || cache->tb_ready[block_id] == 0) {
        return false;
    }
    if (in_block >= cache->tb_blocks[block_id].size()) {
        return false;
    }

    if (metrics != nullptr) {
        ++metrics->tb_cache_hits;
    }
    const uint8_t byte = cache->tb_blocks[block_id][in_block];
    return ((byte >> (node_id % 8)) & 1u) != 0;
}

bool fid_match_node(
    const SequentialEqRuntime::Leaf& leaf,
    SequentialEqQueryCache::Leaf* cache,
    size_t node_id,
    compass_lz4_filter::QueryDecompressionMetrics* metrics) {
    if (cache == nullptr || !leaf.active || leaf.always_false || leaf.fid_storage.block_size == 0 ||
        node_id >= leaf.n_elements) {
        return false;
    }

    const size_t block_id = node_id / leaf.fid_storage.block_size;
    const size_t in_block = node_id % leaf.fid_storage.block_size;
    if (block_id >= cache->fid_blocks.size() || block_id >= cache->fid_ready.size() || cache->fid_ready[block_id] == 0) {
        return false;
    }
    if (in_block >= cache->fid_blocks[block_id].size()) {
        return false;
    }

    if (metrics != nullptr) {
        ++metrics->fid_cache_hits;
    }
    const uint8_t bucket = cache->fid_blocks[block_id][in_block];
    return bucket == static_cast<uint8_t>(leaf.target_bucket);
}

template <typename DistT, typename QueryT>
std::vector<std::pair<DistT, hnswlib::labeltype>> search_with_compass_filter(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* query_data,
    size_t k,
    size_t ef,
    const compass_lz4_filter::CompassLz4FilterEngine& traversal_engine,
    const compass_lz4_filter::CompassLz4FilterEngine& result_engine,
    compass_lz4_filter::QueryBlockCache* cache,
    SearchCallStats* call_stats) {
    std::vector<std::pair<DistT, hnswlib::labeltype>> result;
    result.reserve(k);
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
        bool allowed = false;
        if (traversal_mode) {
            allowed = traversal_engine.allow_traversal(node_id, cache, &call_stats->decomp);
        } else {
            allowed = result_engine.allow_result(node_id, cache, &call_stats->decomp);
        }
        ++call_stats->filter_eval_calls;
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

    while (!top_candidates.empty()) {
        const Pair<DistT> item = top_candidates.top();
        top_candidates.pop();
        result.emplace_back(item.first, index.getExternalLabel(item.second));
    }
    std::reverse(result.begin(), result.end());

    index.visited_list_pool_->releaseVisitedList(vl);
    return result;
}

template <typename DistT, typename QueryT>
std::vector<std::pair<DistT, hnswlib::labeltype>> search_with_sequential_eq_filter(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* query_data,
    size_t k,
    size_t ef,
    const SequentialEqRuntime& runtime,
    SequentialEqQueryCache* cache,
    SequentialEqSearchWorkspace* workspace,
    SearchCallStats* call_stats) {
    std::vector<std::pair<DistT, hnswlib::labeltype>> result;
    result.reserve(k);
    if (index.cur_element_count == 0 || runtime.empty_result) {
        return result;
    }

    for (const SequentialEqRuntime::Leaf& leaf : runtime.leaves) {
        if (!leaf.active || leaf.always_false) {
            continue;
        }
        if (leaf.n_elements != index.cur_element_count) {
            throw std::runtime_error("Sequential runtime element count does not match index");
        }
    }

    SearchCallStats local_stats;
    if (call_stats == nullptr) {
        call_stats = &local_stats;
    }
    if (cache == nullptr || workspace == nullptr) {
        throw std::runtime_error("SequentialEq search received null cache/workspace");
    }
    for (size_t leaf_idx = 0; leaf_idx < runtime.leaves.size(); ++leaf_idx) {
        const SequentialEqRuntime::Leaf& leaf = runtime.leaves[leaf_idx];
        if (!leaf.active || leaf.always_false) {
            continue;
        }
        const SequentialEqQueryCache::Leaf& leaf_cache = cache->leaves[leaf_idx];
        if (leaf_cache.tb_blocks.size() != leaf.tb_storage.block_count() ||
            leaf_cache.tb_ready.size() != leaf.tb_storage.block_count() ||
            leaf_cache.fid_blocks.size() != leaf.fid_storage.block_count() ||
            leaf_cache.fid_ready.size() != leaf.fid_storage.block_count()) {
            throw std::runtime_error(
                "SequentialEq query cache must be fully preallocated before search");
        }
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

    // Decompress TB blocks after upper-layer traversal, before level-0 expansion.
    for (size_t leaf_idx = 0; leaf_idx < runtime.leaves.size(); ++leaf_idx) {
        prefetch_tb_blocks(runtime.leaves[leaf_idx], &cache->leaves[leaf_idx], &call_stats->decomp);
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

        std::vector<hnswlib::tableint>& neighbor_ids = workspace->neighbor_ids;
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

        auto traversal_leaf_match = [&](size_t leaf_idx, size_t node_id) -> bool {
            const SequentialEqRuntime::Leaf& leaf = runtime.leaves[leaf_idx];
            SequentialEqQueryCache::Leaf& leaf_cache = cache->leaves[leaf_idx];
            if (!leaf.active || leaf.always_false) {
                return false;
            }

            const bool tb_match = tb_match_node(leaf, &leaf_cache, node_id, &call_stats->decomp);
            if (tb_match) {
                return true;
            }

            if (leaf.fid_storage.block_size != 0) {
                const size_t fid_block_id = node_id / leaf.fid_storage.block_size;
                prefetch_fid_block(leaf, fid_block_id, &leaf_cache, &call_stats->decomp);
            }
            return fid_match_node(leaf, &leaf_cache, node_id, &call_stats->decomp);
        };

        auto result_leaf_match = [&](size_t leaf_idx, size_t node_id) -> bool {
            const SequentialEqRuntime::Leaf& leaf = runtime.leaves[leaf_idx];
            SequentialEqQueryCache::Leaf& leaf_cache = cache->leaves[leaf_idx];
            if (!leaf.active || leaf.always_false) {
                return false;
            }
            if (leaf.fid_storage.block_size != 0) {
                const size_t fid_block_id = node_id / leaf.fid_storage.block_size;
                prefetch_fid_block(leaf, fid_block_id, &leaf_cache, &call_stats->decomp);
            }
            return fid_match_node(leaf, &leaf_cache, node_id, &call_stats->decomp);
        };

        for (size_t idx = 0; idx < neighbor_ids.size(); ++idx) {
            const hnswlib::tableint candidate_id = neighbor_ids[idx];
            const size_t node_id = static_cast<size_t>(candidate_id);
            const bool traversal_left = traversal_leaf_match(0, node_id);
            const bool traversal_right = traversal_leaf_match(1, node_id);
            const bool traversal_allowed = traversal_left || traversal_right;
            ++call_stats->filter_eval_calls;
            if (!traversal_allowed) {
                continue;
            }

            const DistT dist = index.fstdistfunc_(
                query_data,
                index.getDataByInternalId(candidate_id),
                index.dist_func_param_);

            const bool result_left = result_leaf_match(0, node_id);
            const bool result_right = result_leaf_match(1, node_id);
            const bool result_allowed_by_predicate =
                (runtime.logical_op == filter_expr::LogicalOp::And)
                    ? (result_left && result_right)
                    : (result_left || result_right);
            ++call_stats->filter_eval_calls;

            if (top_candidates.size() < ef || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);

                const bool result_allowed =
                    !index.isMarkedDeleted(candidate_id) && result_allowed_by_predicate;
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

    while (!top_candidates.empty()) {
        const Pair<DistT> item = top_candidates.top();
        top_candidates.pop();
        result.emplace_back(item.first, index.getExternalLabel(item.second));
    }
    std::reverse(result.begin(), result.end());

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
    const compass_lz4_filter::CompassLz4FilterEngine& traversal_engine,
    const compass_lz4_filter::CompassLz4FilterEngine& result_engine,
    const MetadataTable& metadata,
    const SequentialEqRuntime* sequential_runtime) {
    const size_t resolved_ef = resolve_ef(args);
    const double effective_break_factor = resolve_effective_break_factor(args, resolved_ef);

    SpaceT space(static_cast<size_t>(dim));
    hnswlib::HierarchicalNSW<DistT> index(&space, args.graph_path);
    index.setEf(resolved_ef);

    if (result_engine.num_elements() != index.getCurrentElementCount()) {
        throw std::runtime_error(
            "FID/TB manifest n_elements does not match graph element count: manifest=" +
            std::to_string(result_engine.num_elements()) +
            ", graph=" + std::to_string(index.getCurrentElementCount()));
    }
    if (traversal_engine.num_elements() != index.getCurrentElementCount()) {
        throw std::runtime_error(
            "Traversal manifest n_elements does not match graph element count: manifest=" +
            std::to_string(traversal_engine.num_elements()) +
            ", graph=" + std::to_string(index.getCurrentElementCount()));
    }

    RunStats stats;
    const bool use_sequential_eq = (sequential_runtime != nullptr && sequential_runtime->enabled);
    stats.resolved_ef = resolved_ef;
    stats.effective_break_factor = effective_break_factor;
    const size_t total_elements = index.getCurrentElementCount();

    std::vector<size_t> result_nodes = result_engine.collect_result_candidates();
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
            << "lz4_decompress_time_ms,fid_blocks_decompressed,tb_blocks_decompressed,"
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
        SequentialEqQueryCache query_cache;
        SequentialEqSearchWorkspace query_workspace;
        if (use_sequential_eq) {
            query_cache = make_preallocated_query_cache(*sequential_runtime);
            query_workspace.neighbor_ids.reserve(index.maxM0_);
        }

        const auto search_start = std::chrono::steady_clock::now();
        std::vector<std::pair<DistT, hnswlib::labeltype>> result;
        if (use_sequential_eq) {
            result = search_with_sequential_eq_filter<DistT, QueryT>(
                index,
                qptr,
                static_cast<size_t>(args.k),
                resolved_ef,
                *sequential_runtime,
                &query_cache,
                &query_workspace,
                &call_stats);
        } else {
            compass_lz4_filter::QueryBlockCache engine_query_cache(result_engine.attribute_count());
            result = search_with_compass_filter<DistT, QueryT>(
                index,
                qptr,
                static_cast<size_t>(args.k),
                resolved_ef,
                traversal_engine,
                result_engine,
                &engine_query_cache,
                &call_stats);
        }
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

        stats.lz4_decompress_time_ns += call_stats.decomp.lz4_decompress_time_ns;
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

            m.lz4_decompress_time_ns = call_stats.decomp.lz4_decompress_time_ns;
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
                << std::setprecision(6) << (static_cast<double>(call_stats.decomp.lz4_decompress_time_ns) / 1e6) << ','
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
    const CompressionStats& compression_stats,
    const TwoLeafExactSpec& spec) {
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

    const double decomp_ms = static_cast<double>(stats.lz4_decompress_time_ns) / 1e6;
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
    oss << "compass_search_w_lz4 summary\n";
    oss << "dataset_type: " << args.dataset_type << "\n";
    oss << "query_path: " << args.query_path << "\n";
    oss << "graph_path: " << args.graph_path << "\n";
    oss << "fidtb_manifest: " << args.fidtb_manifest << "\n";
    oss << "index_elements: " << index_info.cur_element_count << "\n";
    oss << "index_dimension: " << index_dim << "\n";
    oss << "index_vector_type: " << index_vector_type << "\n";
    oss << "query_num: " << query_info.num << ", query_dim: " << query_info.dim << "\n";
    oss << "queries_requested: "
        << (args.num_queries > 0 ? std::to_string(args.num_queries) : std::string("all")) << "\n";
    oss << "k: " << args.k << ", ef: " << stats.resolved_ef
        << ", break_factor: " << stats.effective_break_factor << "\n";
    oss << "fid_block_size_bytes: " << args.fid_block_size_bytes << "\n";
    oss << "tb_block_size_bytes: " << args.tb_block_size_bytes << "\n";
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
    oss << "multi_exact_fields: " << spec.left.field << "," << spec.right.field << "\n";
    oss << "multi_exact_operator: " << logical_op_name(spec.logical_op) << "\n";
    oss << "traversal_rule: leaf_or\n";
    oss << "result_rule: predicate_op\n";

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

    oss << "lz4_decompress_time_ms: " << decomp_ms << "\n";
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

MetadataTable build_manifest_backed_metadata(const compass_lz4_filter::ManifestData& manifest) {
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

        const TwoLeafExactSpec multi_exact = parse_two_leaf_exact_expression(args.filter_expression);
        const std::string traversal_expression = build_or_traversal_expression(multi_exact);

        std::unordered_set<std::string> fields;
        fields.insert(multi_exact.left.field);
        fields.insert(multi_exact.right.field);

        const compass_lz4_filter::CompassLz4FilterEngine traversal_engine =
            compass_lz4_filter::CompassLz4FilterEngine::Build(
                args.fidtb_manifest,
                traversal_expression,
                fields,
                args.fid_block_size_bytes,
                args.tb_block_size_bytes);
        const compass_lz4_filter::CompassLz4FilterEngine result_engine =
            compass_lz4_filter::CompassLz4FilterEngine::Build(
                args.fidtb_manifest,
                args.filter_expression,
                fields,
                args.fid_block_size_bytes,
                args.tb_block_size_bytes);

        filter_expr::Expression expr(args.filter_expression);
        const SequentialEqRuntime sequential_runtime = build_sequential_eq_runtime(args, result_engine);
        const CompressionStats compression_stats = compute_compression_stats(sequential_runtime);

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

        MetadataTable metadata = build_manifest_backed_metadata(result_engine.manifest());

        RunStats stats;
        if (query_is_fvecs) {
            DenseVectors<float> queries = filter_search_io::read_fvecs(args.query_path);
            stats = run_search_typed<float, hnswlib::L2Space, float>(
                args,
                static_cast<int>(index_dim),
                queries,
                traversal_engine,
                result_engine,
                metadata,
                sequential_runtime.enabled ? &sequential_runtime : nullptr);
        } else {
            DenseVectors<uint8_t> queries = filter_search_io::read_bvecs(args.query_path);
            stats = run_search_typed<int, hnswlib::L2SpaceI, uint8_t>(
                args,
                static_cast<int>(index_dim),
                queries,
                traversal_engine,
                result_engine,
                metadata,
                sequential_runtime.enabled ? &sequential_runtime : nullptr);
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
            compression_stats,
            multi_exact);
        std::cout << summary;
        write_summary_if_needed(summary, args.summary_out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
