#include "../filter_expr.h"
#include "../io_utils.h"

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using filter_search_io::DenseVectors;
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
    int nfilters = -1;

    std::string topk_out;
    std::string per_query_out;
    std::string summary_out;
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

struct FilterMapBuildResult {
    std::vector<char> filter_map;
    std::vector<faiss::idx_t> filtered_ids;
    uint64_t eval_calls = 0;
    uint64_t eval_time_ns = 0;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " --dataset-type <sift1m|sift|sift1b|laion|hnm>"
        << " --graph <acorn.index>"
        << " --query <path(.fvecs|.bvecs)>"
        << " --k <int>"
        << " --filter \"<expression>\""
        << " [--ef <int>]"
        << " [--num-queries <int>]"
        << " [--payload-jsonl <path>]"
        << " [--payload <path>]"
        << " [--metadata-csv <path>]"
        << " [--id-column id]"
        << " [--nfilters <int>]"
        << " [--topk-out <path>]"
        << " [--per-query-out <path>]"
        << " [--summary-out <path>]\n";
}

std::string lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
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
            args.dataset_type = lower_copy(require_value(cur));
        } else if (cur == "--graph" || cur == "--index") {
            args.graph_path = require_value(cur);
        } else if (cur == "--query") {
            args.query_path = require_value(cur);
        } else if (cur == "--k") {
            args.k = std::stoi(require_value(cur));
        } else if (cur == "--ef") {
            args.ef = std::stoi(require_value(cur));
        } else if (cur == "--filter") {
            args.filter_expression = require_value(cur);
        } else if (cur == "--payload-jsonl" || cur == "--payload") {
            args.payload_jsonl = require_value(cur);
        } else if (cur == "--metadata-csv") {
            args.metadata_csv = require_value(cur);
        } else if (cur == "--id-column") {
            args.id_column = require_value(cur);
        } else if (cur == "--nfilters") {
            args.nfilters = std::stoi(require_value(cur));
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

    if (args.dataset_type.empty()) {
        throw std::runtime_error("--dataset-type is required");
    }
    if (args.dataset_type != "sift" && args.dataset_type != "sift1m" &&
        args.dataset_type != "sift1b" && args.dataset_type != "laion" &&
        args.dataset_type != "hnm") {
        throw std::runtime_error(
            "--dataset-type must be one of: sift1m, sift, sift1b, laion, hnm");
    }

    ensure_readable_file(args.graph_path, "--graph/--index");
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

    if (!args.metadata_csv.empty()) {
        ensure_readable_file(args.metadata_csv, "--metadata-csv");
    }
    if (!args.payload_jsonl.empty()) {
        ensure_readable_file(args.payload_jsonl, "--payload-jsonl/--payload");
    }

    const bool is_sift =
        args.dataset_type == "sift" || args.dataset_type == "sift1m" || args.dataset_type == "sift1b";
    if (!is_sift && args.payload_jsonl.empty()) {
        throw std::runtime_error("--payload-jsonl/--payload is required for laion/hnm");
    }

    if (args.ef <= 0) {
        args.ef = std::max(100, args.k);
    }
    if (args.id_column.empty()) {
        args.id_column = "id";
    }

    return args;
}

MetadataTable build_sift_synthetic_metadata(
    const std::unordered_set<std::string>& referenced_fields,
    size_t num_labels,
    int nfilters) {
    if (!(referenced_fields.size() == 1 && referenced_fields.count("synthetic_id_bucket") == 1)) {
        throw std::runtime_error(
            "Synthetic SIFT fallback only supports expressions over field 'synthetic_id_bucket'. "
            "Provide --metadata-csv for other fields.");
    }
    if (nfilters <= 0) {
        throw std::runtime_error("Synthetic SIFT metadata fallback requires nfilters > 0");
    }

    MetadataTable table;
    table.total_labels = num_labels;
    table.rows.reserve(num_labels);

    for (size_t i = 0; i < num_labels; ++i) {
        int gid = static_cast<int>((i * static_cast<size_t>(nfilters)) / num_labels);
        if (gid >= nfilters) {
            gid = nfilters - 1;
        }
        table.rows[i]["synthetic_id_bucket"] = std::to_string(gid);
    }

    table.populated_rows = table.rows.size();
    table.missing_rows = 0;
    return table;
}

template <typename QueryT>
float l2_sqr_query_to_float_base(const QueryT* query, const float* base, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float qv = static_cast<float>(query[i]);
        const float d = qv - base[i];
        acc += d * d;
    }
    return acc;
}

FilterMapBuildResult build_filter_map(
    const MetadataTable& metadata,
    const filter_expr::Expression& expr,
    size_t num_labels) {
    FilterMapBuildResult out;
    out.filter_map.assign(num_labels, 0);

    for (size_t label = 0; label < num_labels; ++label) {
        const auto t0 = std::chrono::steady_clock::now();

        bool matched = false;
        auto row_it = metadata.rows.find(label);
        if (row_it != metadata.rows.end()) {
            const auto& row = row_it->second;
            auto accessor = [&](const std::string& field) -> std::optional<std::string_view> {
                auto it = row.find(field);
                if (it == row.end()) {
                    return std::nullopt;
                }
                return std::string_view(it->second);
            };
            matched = expr.evaluate(accessor);
        }

        const auto t1 = std::chrono::steady_clock::now();
        ++out.eval_calls;
        out.eval_time_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        if (matched) {
            out.filter_map[label] = 1;
            out.filtered_ids.push_back(static_cast<faiss::idx_t>(label));
        }
    }

    return out;
}

template <typename QueryT>
RunStats run_search_typed(
    const Args& args,
    const faiss::IndexACORN& index,
    const faiss::IndexFlat& flat,
    const DenseVectors<QueryT>& queries,
    std::vector<char>* filter_map,
    const std::vector<faiss::idx_t>& filtered_ids,
    uint64_t filter_eval_calls,
    uint64_t filter_eval_time_ns) {
    RunStats stats;
    stats.total_elements = static_cast<size_t>(index.ntotal);
    stats.selectivity_count = filtered_ids.size();
    stats.filter_eval_calls = filter_eval_calls;
    stats.filter_eval_time_ns = filter_eval_time_ns;
    stats.filter_time_total_ns = filter_eval_time_ns;

    if (stats.total_elements > 0) {
        stats.selectivity_ratio =
            static_cast<double>(stats.selectivity_count) / static_cast<double>(stats.total_elements);
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

    const float* xb = flat.get_xb();
    const int dim = index.d;
    std::vector<float> query_buf(static_cast<size_t>(dim));

    const auto loop_start = std::chrono::steady_clock::now();
    for (size_t qid = 0; qid < query_count; ++qid) {
        const QueryT* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);

        const float* query_f32 = nullptr;
        if constexpr (std::is_same_v<QueryT, float>) {
            query_f32 = qptr;
        } else {
            for (int j = 0; j < dim; ++j) {
                query_buf[static_cast<size_t>(j)] = static_cast<float>(qptr[j]);
            }
            query_f32 = query_buf.data();
        }

        std::vector<float> ann_dist(static_cast<size_t>(args.k), std::numeric_limits<float>::infinity());
        std::vector<faiss::idx_t> ann_ids(static_cast<size_t>(args.k), static_cast<faiss::idx_t>(-1));

        faiss::SearchParametersACORN params;
        params.efSearch = args.ef;

        const auto t_search0 = std::chrono::steady_clock::now();
        index.search(
            1,
            query_f32,
            static_cast<faiss::idx_t>(args.k),
            ann_dist.data(),
            ann_ids.data(),
            filter_map->data(),
            &params);
        const auto t_search1 = std::chrono::steady_clock::now();
        const uint64_t query_search_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_search1 - t_search0).count());

        std::vector<std::pair<float, faiss::idx_t>> ann;
        ann.reserve(static_cast<size_t>(args.k));
        for (int i = 0; i < args.k; ++i) {
            if (ann_ids[static_cast<size_t>(i)] >= 0) {
                ann.emplace_back(
                    ann_dist[static_cast<size_t>(i)],
                    ann_ids[static_cast<size_t>(i)]);
            }
        }

        std::priority_queue<std::pair<float, faiss::idx_t>> enns_heap;
        for (faiss::idx_t id : filtered_ids) {
            const float* base = xb + static_cast<size_t>(id) * static_cast<size_t>(dim);
            const float dist = l2_sqr_query_to_float_base(query_f32, base, dim);

            if (enns_heap.size() < static_cast<size_t>(args.k)) {
                enns_heap.emplace(dist, id);
            } else if (dist < enns_heap.top().first) {
                enns_heap.pop();
                enns_heap.emplace(dist, id);
            }
        }

        std::unordered_set<faiss::idx_t> enns_labels;
        enns_labels.reserve(enns_heap.size() * 2 + 1);
        while (!enns_heap.empty()) {
            enns_labels.insert(enns_heap.top().second);
            enns_heap.pop();
        }

        std::unordered_set<faiss::idx_t> ann_labels;
        ann_labels.reserve(ann.size() * 2 + 1);
        for (const auto& item : ann) {
            ann_labels.insert(item.second);
        }

        size_t overlap = 0;
        for (faiss::idx_t id : enns_labels) {
            if (ann_labels.find(id) != ann_labels.end()) {
                ++overlap;
            }
        }

        const size_t enns_size = enns_labels.size();
        if (enns_size < static_cast<size_t>(args.k)) {
            ++stats.queries_with_enns_lt_k;
        }

        double recall = 0.0;
        if (enns_size == 0) {
            recall = ann.empty() ? 1.0 : 0.0;
        } else {
            const size_t denom = std::min(static_cast<size_t>(args.k), enns_size);
            recall = static_cast<double>(overlap) / static_cast<double>(denom);
        }

        stats.recall_sum += recall;
        stats.search_time_total_ns += query_search_ns;
        stats.returned_results += ann.size();

        if (topk_out.is_open()) {
            topk_out << qid << '\t';
            for (size_t i = 0; i < ann.size(); ++i) {
                if (i > 0) {
                    topk_out << ',';
                }
                topk_out << ann[i].second << ':' << static_cast<double>(ann[i].first);
            }
            topk_out << '\n';
        }

        if (capture_per_query) {
            QueryMetrics m;
            m.qid = qid;
            m.recall_at_k = recall;
            m.enns_size = enns_size;
            m.anns_size = ann.size();
            m.filter_time_ns = 0;
            m.search_time_ns = query_search_ns;
            stats.per_query_metrics.push_back(m);

            per_query_out << qid << ','
                          << std::fixed << std::setprecision(6) << recall << ','
                          << enns_size << ','
                          << ann.size() << ','
                          << std::setprecision(6) << 0.0 << ','
                          << std::setprecision(6)
                          << (static_cast<double>(query_search_ns) / 1e6)
                          << '\n';
        }
    }
    const auto loop_end = std::chrono::steady_clock::now();

    stats.query_count = query_count;
    stats.search_loop_time_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start).count());
    stats.average_recall_at_k =
        (query_count > 0) ? (stats.recall_sum / static_cast<double>(query_count)) : 0.0;

    return stats;
}

std::string build_summary(
    const Args& args,
    const VecFileInfo& query_info,
    const std::string& query_type,
    int index_dim,
    const MetadataTable& metadata,
    const filter_expr::Expression& expr,
    const RunStats& stats) {
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
    const double qps =
        (stats.search_loop_time_ns > 0)
            ? (static_cast<double>(stats.query_count) * 1e9 /
               static_cast<double>(stats.search_loop_time_ns))
            : 0.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "acorn_search summary\n";
    oss << "dataset_type: " << args.dataset_type << "\n";
    oss << "query_path: " << args.query_path << "\n";
    oss << "graph_path: " << args.graph_path << "\n";
    oss << "index_elements: " << stats.total_elements << "\n";
    oss << "index_dimension: " << index_dim << "\n";
    oss << "index_vector_type: fvecs\n";
    oss << "query_vector_type: " << query_type << "\n";
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
    oss << "qps: " << qps << "\n";
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
        const Args args = parse_args(argc, argv);

        const VecFileInfo query_info = filter_search_io::inspect_vector_file(args.query_path);
        const bool query_is_fvecs = filter_search_io::ends_with(args.query_path, ".fvecs");
        const std::string query_type = query_is_fvecs ? "fvecs" : "bvecs";

        filter_expr::Expression expr(args.filter_expression);
        const std::unordered_set<std::string> fields = expr.referenced_fields();
        if (fields.empty()) {
            throw std::runtime_error("Filter expression did not reference any fields");
        }

        std::unique_ptr<faiss::Index> index_guard(faiss::read_index(args.graph_path.c_str()));
        if (!index_guard) {
            throw std::runtime_error("Failed to read ACORN index from file: " + args.graph_path);
        }

        auto* acorn_index = dynamic_cast<faiss::IndexACORN*>(index_guard.get());
        if (acorn_index == nullptr) {
            throw std::runtime_error("Loaded index is not faiss::IndexACORN");
        }

        auto* flat_storage = dynamic_cast<faiss::IndexFlat*>(acorn_index->storage);
        if (flat_storage == nullptr) {
            throw std::runtime_error("ACORN storage is not IndexFlat; ENNS baseline requires flat storage");
        }

        const size_t num_labels = static_cast<size_t>(acorn_index->ntotal);
        if (acorn_index->d != query_info.dim) {
            std::ostringstream oss;
            oss << "Graph/query dimension mismatch: ACORN dim " << acorn_index->d
                << " vs query dim " << query_info.dim;
            throw std::runtime_error(oss.str());
        }

        const bool is_sift = args.dataset_type == "sift" || args.dataset_type == "sift1m" ||
            args.dataset_type == "sift1b";

        MetadataTable metadata;
        if (is_sift) {
            if (!args.metadata_csv.empty()) {
                metadata = filter_search_io::load_csv_metadata(
                    args.metadata_csv, fields, args.id_column, num_labels);
            } else {
                int fallback_nfilters = args.nfilters;
                if (fallback_nfilters <= 0) {
                    fallback_nfilters = acorn_index->acorn.gamma;
                }
                metadata = build_sift_synthetic_metadata(fields, num_labels, fallback_nfilters);
            }
        } else {
            metadata = filter_search_io::load_jsonl_metadata(args.payload_jsonl, fields, num_labels);
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

        FilterMapBuildResult filter_build = build_filter_map(metadata, expr, num_labels);

        RunStats stats;
        if (query_is_fvecs) {
            DenseVectors<float> queries = filter_search_io::read_fvecs(args.query_path);
            stats = run_search_typed<float>(
                args,
                *acorn_index,
                *flat_storage,
                queries,
                &filter_build.filter_map,
                filter_build.filtered_ids,
                filter_build.eval_calls,
                filter_build.eval_time_ns);
        } else {
            DenseVectors<uint8_t> queries = filter_search_io::read_bvecs(args.query_path);
            stats = run_search_typed<uint8_t>(
                args,
                *acorn_index,
                *flat_storage,
                queries,
                &filter_build.filter_map,
                filter_build.filtered_ids,
                filter_build.eval_calls,
                filter_build.eval_time_ns);
        }

        const std::string summary = build_summary(
            args, query_info, query_type, acorn_index->d, metadata, expr, stats);
        std::cout << summary;
        write_summary_if_needed(summary, args.summary_out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
