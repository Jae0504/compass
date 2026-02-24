#include "../../hnswlib/filter_search_hnswlib/hnswlib.h"

#include "filter_expr.h"
#include "io_utils.h"

#include <chrono>
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

struct Args {
    std::string dataset_type;
    std::string graph_path;
    std::string query_path;
    std::string filter_expression;

    int k = 0;
    int ef = -1;
    int num_queries = -1;

    std::string payload_jsonl;
    std::string metadata_csv;
    std::string id_column = "id";

    std::string topk_out;
    std::string per_query_out;
    std::string summary_out;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --dataset-type <sift|laion|hnm>"
        << " --graph <path>"
        << " --query <path(.fvecs|.bvecs)>"
        << " --k <int>"
        << " --filter \"<expression>\""
        << " [--ef <int>]"
        << " [--payload-jsonl <path>]"
        << " [--metadata-csv <path>]"
        << " [--id-column id]"
        << " [--num-queries <int>]"
        << " [--max-queries <int>]"  // Backward-compatible alias.
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
        } else if (cur == "--payload-jsonl") {
            args.payload_jsonl = require_value(cur);
        } else if (cur == "--metadata-csv") {
            args.metadata_csv = require_value(cur);
        } else if (cur == "--id-column") {
            args.id_column = require_value(cur);
        } else if (cur == "--num-queries" || cur == "--max-queries") {
            args.num_queries = std::stoi(require_value(cur));
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
    if (args.k <= 0) {
        throw std::runtime_error("--k must be > 0");
    }
    if (args.filter_expression.empty()) {
        throw std::runtime_error("--filter is required");
    }
    if (args.num_queries == 0) {
        throw std::runtime_error("--num-queries/--max-queries must be > 0 when provided");
    }

    const bool query_fvec = filter_search_io::ends_with(args.query_path, ".fvecs");
    const bool query_bvec = filter_search_io::ends_with(args.query_path, ".bvecs");

    if (!query_fvec && !query_bvec) {
        throw std::runtime_error("--query must end with .fvecs or .bvecs");
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

bool metadata_matches_filter(
    hnswlib::labeltype label,
    const MetadataTable& metadata,
    const filter_expr::Expression& expr) {
    auto row_it = metadata.rows.find(static_cast<size_t>(label));
    if (row_it == metadata.rows.end()) {
        return false;
    }
    const auto& row = row_it->second;
    auto accessor = [&](const std::string& field) -> std::optional<std::string_view> {
        auto it = row.find(field);
        if (it == row.end()) {
            return std::nullopt;
        }
        return std::string_view(it->second);
    };
    return expr.evaluate(accessor);
}

float l2_sqr_f32(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

int l2_sqr_u8(const uint8_t* a, const uint8_t* b, int dim) {
    int acc = 0;
    for (int i = 0; i < dim; ++i) {
        const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
        acc += d * d;
    }
    return acc;
}

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
};

struct RunStats {
    size_t query_count = 0;
    uint64_t search_loop_time_ns = 0;
    uint64_t filter_eval_calls = 0;
    uint64_t filter_eval_time_ns = 0;
    uint64_t filter_time_total_ns = 0;
    uint64_t search_time_total_ns = 0;
    size_t returned_results = 0;
    size_t selectivity_count = 0;
    size_t total_elements = 0;
    double selectivity_ratio = 0.0;
    double recall_sum = 0.0;
    double average_recall_at_k = 0.0;
    size_t queries_with_enns_lt_k = 0;
    std::vector<QueryMetrics> per_query_metrics;
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
    RunStats stats;

    const size_t total_elements = index.getCurrentElementCount();
    std::vector<FilteredCandidate> filtered_candidates;
    filtered_candidates.reserve(total_elements);
    for (size_t internal = 0; internal < total_elements; ++internal) {
        const hnswlib::tableint iid = static_cast<hnswlib::tableint>(internal);
        const hnswlib::labeltype label = index.getExternalLabel(iid);
        if (metadata_matches_filter(label, metadata, expr)) {
            filtered_candidates.push_back(FilteredCandidate{iid, label});
        }
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
        per_query_out << "query_id,recall_at_k,enns_size,anns_size,filter_time_ms,search_time_ms\n";
    }

    const bool capture_per_query = per_query_out.is_open();
    if (capture_per_query) {
        stats.per_query_metrics.reserve(query_count);
    }

    size_t returned_results = 0;
    const auto loop_start = std::chrono::steady_clock::now();
    uint64_t prev_filter_time_ns = filter_functor.eval_time_ns();
    for (size_t qid = 0; qid < query_count; ++qid) {
        const QueryT* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);
        const auto search_start = std::chrono::steady_clock::now();
        std::vector<std::pair<DistT, hnswlib::labeltype>> result =
            index.searchKnnCloserFirst(static_cast<const void*>(qptr), static_cast<size_t>(args.k), &filter_functor);
        const auto search_end = std::chrono::steady_clock::now();
        const uint64_t query_search_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(search_end - search_start).count());
        const uint64_t cur_filter_time_ns = filter_functor.eval_time_ns();
        const uint64_t query_filter_ns = cur_filter_time_ns - prev_filter_time_ns;
        prev_filter_time_ns = cur_filter_time_ns;

        std::priority_queue<std::pair<DistT, hnswlib::labeltype>> enns_heap;
        for (const FilteredCandidate& candidate : filtered_candidates) {
            const char* data_ptr = index.getDataByInternalId(candidate.internal_id);
            DistT dist;
            if constexpr (std::is_same_v<QueryT, float>) {
                dist = static_cast<DistT>(l2_sqr_f32(
                    qptr,
                    reinterpret_cast<const float*>(data_ptr),
                    dim));
            } else {
                dist = static_cast<DistT>(l2_sqr_u8(
                    qptr,
                    reinterpret_cast<const uint8_t*>(data_ptr),
                    dim));
            }

            if (enns_heap.size() < static_cast<size_t>(args.k)) {
                enns_heap.emplace(dist, candidate.label);
            } else if (dist < enns_heap.top().first) {
                enns_heap.pop();
                enns_heap.emplace(dist, candidate.label);
            }
        }

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
        stats.filter_time_total_ns += query_filter_ns;
        stats.search_time_total_ns += query_search_ns;

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
            m.filter_time_ns = query_filter_ns;
            m.search_time_ns = query_search_ns;
            stats.per_query_metrics.push_back(m);

            per_query_out << qid << ','
                          << std::fixed << std::setprecision(6) << recall << ','
                          << enns_size << ','
                          << result.size() << ','
                          << std::setprecision(6) << (static_cast<double>(query_filter_ns) / 1e6) << ','
                          << std::setprecision(6) << (static_cast<double>(query_search_ns) / 1e6)
                          << '\n';
        }
    }
    const auto loop_end = std::chrono::steady_clock::now();

    stats.query_count = query_count;
    stats.search_loop_time_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start).count());
    stats.filter_eval_calls = filter_functor.eval_calls();
    stats.filter_eval_time_ns = filter_functor.eval_time_ns();
    stats.returned_results = returned_results;
    stats.average_recall_at_k =
        (query_count > 0) ? (stats.recall_sum / static_cast<double>(query_count)) : 0.0;
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
    const RunStats& stats) {
    const double loop_ms = static_cast<double>(stats.search_loop_time_ns) / 1e6;
    const double avg_query_ms = (stats.query_count > 0) ? (loop_ms / static_cast<double>(stats.query_count)) : 0.0;
    const double filter_ms = static_cast<double>(stats.filter_eval_time_ns) / 1e6;
    const double avg_filter_ns =
        (stats.filter_eval_calls > 0) ? (static_cast<double>(stats.filter_eval_time_ns) / stats.filter_eval_calls) : 0.0;
    const double avg_filter_per_query_ms =
        (stats.query_count > 0)
            ? (static_cast<double>(stats.filter_time_total_ns) / 1e6 / static_cast<double>(stats.query_count))
            : 0.0;
    const double avg_search_per_query_ms =
        (stats.query_count > 0)
            ? (static_cast<double>(stats.search_time_total_ns) / 1e6 / static_cast<double>(stats.query_count))
            : 0.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "hnswlib_filter_search summary\n";
    oss << "dataset_type: " << args.dataset_type << "\n";
    oss << "query_path: " << args.query_path << "\n";
    oss << "graph_path: " << args.graph_path << "\n";
    oss << "index_elements: " << index_info.cur_element_count << "\n";
    oss << "index_dimension: " << index_dim << "\n";
    oss << "index_vector_type: " << index_vector_type << "\n";
    oss << "query_num: " << query_info.num << ", query_dim: " << query_info.dim << "\n";
    oss << "queries_requested: "
        << (args.num_queries > 0 ? std::to_string(args.num_queries) : std::string("all")) << "\n";
    oss << "k: " << args.k << ", ef: " << args.ef << "\n";
    oss << "filter: " << expr.source() << "\n";
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
    oss << "filter_eval_calls: " << stats.filter_eval_calls << "\n";
    oss << "filter_eval_time_ms: " << filter_ms << "\n";
    oss << "avg_filter_eval_ns: " << avg_filter_ns << "\n";
    oss << "avg_filter_time_per_query_ms: " << avg_filter_per_query_ms << "\n";
    oss << "avg_search_time_per_query_ms: " << avg_search_per_query_ms << "\n";
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

        MetadataTable metadata;
        const size_t num_labels = index_info.cur_element_count;
        if (args.dataset_type == "sift") {
            metadata = filter_search_io::load_csv_metadata(
                args.metadata_csv, fields, args.id_column, num_labels);
        } else {
            metadata = filter_search_io::load_jsonl_metadata(
                args.payload_jsonl, fields, num_labels);
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
        if (query_is_fvecs) {
            DenseVectors<float> queries = filter_search_io::read_fvecs(args.query_path);
            stats = run_search_typed<float, hnswlib::L2Space, float>(
                args, static_cast<int>(index_dim), queries, metadata, expr);
        } else {
            DenseVectors<uint8_t> queries = filter_search_io::read_bvecs(args.query_path);
            stats = run_search_typed<int, hnswlib::L2SpaceI, uint8_t>(
                args, static_cast<int>(index_dim), queries, metadata, expr);
        }

        const std::string summary =
            build_summary(args, query_info, index_info, index_vector_type, index_dim, metadata, expr, stats);
        std::cout << summary;
        write_summary_if_needed(summary, args.summary_out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
// ./hnswlib_filter_search.run --dataset-type sift --graph /home/jykang5/compass/end_to_end/vectordb/dataset/hnsw_graph/sift1m/sift_m128_efc200.bin --query /fast-lab-share/benchmarks/VectorDB/ANN/sift1m/query.fvecs --k 10 --filter "synthetic_id_bucket == 0" --metadata-csv /home/jykang5/compass/end_to_end/vectordb/dataset/metadata/sift1m/sift1m_meta.csv --num-queries 1000
