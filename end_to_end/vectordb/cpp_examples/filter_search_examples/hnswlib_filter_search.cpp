#include "../../hnswlib/filter_search_hnswlib/hnswlib.h"

#include "filter_expr.h"
#include "io_utils.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using filter_search_io::DenseVectors;
using filter_search_io::MetadataTable;
using filter_search_io::VecFileInfo;

namespace {

struct Args {
    std::string dataset_type;
    std::string graph_path;
    std::string base_path;
    std::string query_path;
    std::string filter_expression;

    int k = 0;
    int ef = -1;
    int max_queries = -1;

    std::string payload_jsonl;
    std::string metadata_csv;
    std::string id_column = "id";

    std::string topk_out;
    std::string summary_out;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --dataset-type <sift|laion|hnm>"
        << " --graph <path>"
        << " --base <path(.fvecs|.bvecs)>"
        << " --query <path(.fvecs|.bvecs)>"
        << " --k <int>"
        << " --filter \"<expression>\""
        << " [--ef <int>]"
        << " [--payload-jsonl <path>]"
        << " [--metadata-csv <path>]"
        << " [--id-column id]"
        << " [--max-queries <int>]"
        << " [--topk-out <path>]"
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
        } else if (cur == "--base") {
            args.base_path = require_value(cur);
        } else if (cur == "--query") {
            args.query_path = require_value(cur);
        } else if (cur == "--k") {
            args.k = std::stoi(require_value(cur));
        } else if (cur == "--ef") {
            args.ef = std::stoi(require_value(cur));
        } else if (cur == "--filter") {
            args.filter_expression = require_value(cur);
        } else if (cur == "--payload-jsonl") {
            args.payload_jsonl = require_value(cur);
        } else if (cur == "--metadata-csv") {
            args.metadata_csv = require_value(cur);
        } else if (cur == "--id-column") {
            args.id_column = require_value(cur);
        } else if (cur == "--max-queries") {
            args.max_queries = std::stoi(require_value(cur));
        } else if (cur == "--topk-out") {
            args.topk_out = require_value(cur);
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
    ensure_readable_file(args.base_path, "--base");
    ensure_readable_file(args.query_path, "--query");
    if (args.k <= 0) {
        throw std::runtime_error("--k must be > 0");
    }
    if (args.filter_expression.empty()) {
        throw std::runtime_error("--filter is required");
    }
    if (args.max_queries == 0) {
        throw std::runtime_error("--max-queries must be > 0 when provided");
    }

    const bool base_fvec = filter_search_io::ends_with(args.base_path, ".fvecs");
    const bool base_bvec = filter_search_io::ends_with(args.base_path, ".bvecs");
    const bool query_fvec = filter_search_io::ends_with(args.query_path, ".fvecs");
    const bool query_bvec = filter_search_io::ends_with(args.query_path, ".bvecs");

    if ((!base_fvec && !base_bvec) || (!query_fvec && !query_bvec)) {
        throw std::runtime_error("--base/--query must end with .fvecs or .bvecs");
    }
    if (base_fvec != query_fvec || base_bvec != query_bvec) {
        throw std::runtime_error("--base and --query must have the same vector file type");
    }

    if (args.dataset_type == "sift") {
        ensure_readable_file(args.metadata_csv, "--metadata-csv");
    } else {
        ensure_readable_file(args.payload_jsonl, "--payload-jsonl");
    }

    if (args.ef <= 0) {
        args.ef = std::max(100, args.k);
    }
    if (args.id_column.empty()) {
        args.id_column = "id";
    }
    return args;
}

class ExpressionFilterFunctor : public hnswlib::BaseFilterFunctor {
public:
    ExpressionFilterFunctor(const MetadataTable& metadata, const filter_expr::Expression& expr)
        : metadata_(metadata), expr_(expr) {}

    bool operator()(hnswlib::labeltype label) override {
        const auto start = std::chrono::steady_clock::now();
        ++eval_calls_;

        bool matched = false;
        auto row_it = metadata_.rows.find(static_cast<size_t>(label));
        if (row_it != metadata_.rows.end()) {
            const auto& row = row_it->second;
            auto accessor = [&](const std::string& field) -> std::optional<std::string_view> {
                auto it = row.find(field);
                if (it == row.end()) {
                    return std::nullopt;
                }
                return std::string_view(it->second);
            };
            matched = expr_.evaluate(accessor);
        }

        const auto end = std::chrono::steady_clock::now();
        filter_eval_time_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return matched;
    }

    uint64_t eval_calls() const {
        return eval_calls_;
    }

    uint64_t eval_time_ns() const {
        return filter_eval_time_ns_;
    }

private:
    const MetadataTable& metadata_;
    const filter_expr::Expression& expr_;
    uint64_t eval_calls_ = 0;
    uint64_t filter_eval_time_ns_ = 0;
};

struct RunStats {
    size_t query_count = 0;
    uint64_t search_loop_time_ns = 0;
    uint64_t filter_eval_calls = 0;
    uint64_t filter_eval_time_ns = 0;
    size_t returned_results = 0;
};

template <typename DistT, typename SpaceT, typename QueryT>
RunStats run_search_typed(
    const Args& args,
    int dim,
    const DenseVectors<QueryT>& queries,
    const MetadataTable& metadata,
    const filter_expr::Expression& expr) {
    SpaceT space(static_cast<size_t>(dim));
    hnswlib::HierarchicalNSW<DistT> index(&space, args.graph_path);
    index.setEf(static_cast<size_t>(args.ef));

    ExpressionFilterFunctor filter_functor(metadata, expr);

    size_t query_count = static_cast<size_t>(queries.num);
    if (args.max_queries > 0) {
        query_count = std::min(query_count, static_cast<size_t>(args.max_queries));
    }
    if (query_count == 0) {
        throw std::runtime_error("No query vectors available after --max-queries");
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

    size_t returned_results = 0;
    const auto loop_start = std::chrono::steady_clock::now();
    for (size_t qid = 0; qid < query_count; ++qid) {
        const QueryT* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);
        std::vector<std::pair<DistT, hnswlib::labeltype>> result =
            index.searchKnnCloserFirst(static_cast<const void*>(qptr), static_cast<size_t>(args.k), &filter_functor);

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
    }
    const auto loop_end = std::chrono::steady_clock::now();

    RunStats stats;
    stats.query_count = query_count;
    stats.search_loop_time_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start).count());
    stats.filter_eval_calls = filter_functor.eval_calls();
    stats.filter_eval_time_ns = filter_functor.eval_time_ns();
    stats.returned_results = returned_results;
    return stats;
}

std::string build_summary(
    const Args& args,
    const VecFileInfo& base_info,
    const VecFileInfo& query_info,
    const MetadataTable& metadata,
    const filter_expr::Expression& expr,
    const RunStats& stats) {
    const double loop_ms = static_cast<double>(stats.search_loop_time_ns) / 1e6;
    const double avg_query_ms = (stats.query_count > 0) ? (loop_ms / static_cast<double>(stats.query_count)) : 0.0;
    const double filter_ms = static_cast<double>(stats.filter_eval_time_ns) / 1e6;
    const double avg_filter_ns =
        (stats.filter_eval_calls > 0) ? (static_cast<double>(stats.filter_eval_time_ns) / stats.filter_eval_calls) : 0.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "hnswlib_filter_search summary\n";
    oss << "dataset_type: " << args.dataset_type << "\n";
    oss << "base_path: " << args.base_path << "\n";
    oss << "query_path: " << args.query_path << "\n";
    oss << "graph_path: " << args.graph_path << "\n";
    oss << "base_num: " << base_info.num << ", base_dim: " << base_info.dim << "\n";
    oss << "query_num: " << query_info.num << ", query_dim: " << query_info.dim << "\n";
    oss << "k: " << args.k << ", ef: " << args.ef << "\n";
    oss << "filter: " << expr.source() << "\n";
    oss << "metadata_total_labels: " << metadata.total_labels << "\n";
    oss << "metadata_populated_labels: " << metadata.populated_rows << "\n";
    oss << "metadata_missing_labels: " << metadata.missing_rows << "\n";
    oss << "metadata_invalid_rows: " << metadata.invalid_rows << "\n";
    oss << "metadata_dropped_rows: " << metadata.dropped_rows << "\n";
    oss << "queries_executed: " << stats.query_count << "\n";
    oss << "results_returned_total: " << stats.returned_results << "\n";
    oss << "search_loop_time_ms: " << loop_ms << "\n";
    oss << "avg_query_time_ms: " << avg_query_ms << "\n";
    oss << "filter_eval_calls: " << stats.filter_eval_calls << "\n";
    oss << "filter_eval_time_ms: " << filter_ms << "\n";
    oss << "avg_filter_eval_ns: " << avg_filter_ns << "\n";
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

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        const VecFileInfo base_info = filter_search_io::inspect_vector_file(args.base_path);
        const VecFileInfo query_info = filter_search_io::inspect_vector_file(args.query_path);
        if (base_info.dim != query_info.dim) {
            throw std::runtime_error("Base/query dimension mismatch");
        }

        filter_expr::Expression expr(args.filter_expression);
        std::unordered_set<std::string> fields = expr.referenced_fields();
        if (fields.empty()) {
            throw std::runtime_error("Filter expression did not reference any fields");
        }

        MetadataTable metadata;
        if (args.dataset_type == "sift") {
            metadata = filter_search_io::load_csv_metadata(
                args.metadata_csv, fields, args.id_column, static_cast<size_t>(base_info.num));
        } else {
            metadata = filter_search_io::load_jsonl_metadata(
                args.payload_jsonl, fields, static_cast<size_t>(base_info.num));
        }

        if (metadata.missing_rows > 0) {
            std::cerr << "Warning: " << metadata.missing_rows
                      << " labels do not have metadata for referenced fields\n";
        }
        if (metadata.invalid_rows > 0) {
            std::cerr << "Warning: " << metadata.invalid_rows << " metadata rows were invalid\n";
        }
        if (metadata.dropped_rows > 0) {
            std::cerr << "Warning: " << metadata.dropped_rows << " metadata rows were dropped\n";
        }

        RunStats stats;
        if (filter_search_io::ends_with(args.base_path, ".fvecs")) {
            DenseVectors<float> queries = filter_search_io::read_fvecs(args.query_path);
            stats = run_search_typed<float, hnswlib::L2Space, float>(
                args, base_info.dim, queries, metadata, expr);
        } else {
            DenseVectors<uint8_t> queries = filter_search_io::read_bvecs(args.query_path);
            stats = run_search_typed<int, hnswlib::L2SpaceI, uint8_t>(
                args, base_info.dim, queries, metadata, expr);
        }

        const std::string summary = build_summary(args, base_info, query_info, metadata, expr, stats);
        std::cout << summary;
        write_summary_if_needed(summary, args.summary_out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

