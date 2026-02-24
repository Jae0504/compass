#include "../../hnswlib/build_fid_tb/hnswlib.h"
#include "../../hnswlib/build_fid_tb/globals.h"
#include "../../hnswlib/build_fid_tb/post_proc.h"
#include "../../hnswlib/build_fid_tb/io_utils.h"
#include "../../hnswlib/build_fid_tb/json.hpp"

#include <algorithm>
#include <bitset>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Global variables defined for headers that rely on globals.h
std::vector<std::uint8_t> filter_ids;
std::vector<std::bitset<256>> connector_bits;
std::vector<std::uint8_t> reordered_connector_bits;
std::vector<std::vector<std::uint8_t>> parsed_connector_bits;
std::vector<std::vector<std::uint8_t>> parsed_filter_ids;
std::vector<std::vector<std::uint8_t>> parsed_connector_bits_m1;
std::vector<std::vector<std::uint8_t>> parsed_connector_bits_m2;
std::vector<std::vector<std::uint8_t>> parsed_filter_ids_m1;
std::vector<std::vector<std::uint8_t>> parsed_filter_ids_m2;
bool and_flag = false;
bool ena_algorithm = true;
bool ena_cnt_distcal = false;
int nElements = 0;
int nFilters = 256;
int nThreads = 64;
int cnt_distcal_lv0 = 0;
int cnt_distcal_upper = 0;
std::vector<int> target_filter_ids;
std::vector<int> target_filter_ids_m1;
std::vector<int> target_filter_ids_m2;
std::ofstream RunResultFile;
std::string dataset_type;
int isolated_connection_factor = 1;
int steiner_factor = 1;
int ep_factor = 5;
float break_factor = 1.0f;
unsigned long long iaa_cpu_cycles = 0;
std::vector<std::vector<uint8_t>> compressed_filter_ids;
std::vector<std::vector<uint8_t>> compressed_filter_ids_m1;
std::vector<std::vector<uint8_t>> compressed_filter_ids_m2;
std::vector<int> ids_list;
std::vector<float> ids_dist_list;
std::size_t group_chunk_size = 0;
std::unordered_set<int> unique_ids_list;
std::vector<int> group_bit_visited_list;
std::vector<int> group_bit_decompression_plan;
std::vector<uint8_t> new_connector_bits;

namespace {

const std::vector<std::string> kLaionKeys = {
    "NSFW",
    "similarity",
    "original_width",
    "original_height"
};

const std::unordered_set<std::string> kHnmExcludedKeys = {
    "detail_desc",
    "prod_name"
};

struct Args {
    std::string dataset_type;
    std::string benchmark;
    std::string graph_path;
    std::string base_path;
    std::string payload_path;
    std::string filters_path;
    std::string out_dir = "../../dataset/fid_tb";

    int threads = 64;
    int nfilters = 256;
    int steiner_factor = 1;
    int ep_factor = 5;
    int isolated_connection_factor = 1;
};

struct DatasetInfo {
    int base_num = 0;
    int base_dim = 0;
};

struct AttributeData {
    std::string key;
    std::vector<std::string> values;
};

struct EncodedAttribute {
    std::string key;
    std::vector<uint8_t> fid;
    std::string encoding;
    bool numeric = false;
    size_t unique_non_missing = 0;
    size_t missing_count = 0;
    double min_value = 0.0;
    double max_value = 0.0;
    int used_bins = 0;
    std::map<std::string, int> category_map;
};

std::string trim_copy(const std::string& in) {
    size_t start = 0;
    while (start < in.size() && std::isspace(static_cast<unsigned char>(in[start]))) {
        ++start;
    }
    size_t end = in.size();
    while (end > start && std::isspace(static_cast<unsigned char>(in[end - 1]))) {
        --end;
    }
    return in.substr(start, end - start);
}

std::string strip_outer_quotes(const std::string& in) {
    if (in.size() >= 2) {
        if ((in.front() == '"' && in.back() == '"') ||
            (in.front() == '\'' && in.back() == '\'')) {
            return in.substr(1, in.size() - 2);
        }
    }
    return in;
}

std::string sanitize_key_for_filename(const std::string& key) {
    std::string out;
    out.reserve(key.size());
    for (char c : key) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        out = "attr";
    }
    return out;
}

void ensure_readable_file(const std::string& path, const std::string& flag_name) {
    if (path.empty()) {
        throw std::runtime_error("Missing required argument: " + flag_name);
    }
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
        throw std::runtime_error("File does not exist: " + path);
    }
}

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --dataset-type <sift|laion|hnm>"
        << " --benchmark <name>"
        << " --graph <path>"
        << " --base <path>"
        << " [--payload <path>]"
        << " [--filters <path>]"
        << " [--out-dir ../../dataset/fid_tb]"
        << " [--threads 64]"
        << " [--nfilters 256]"
        << " [--steiner-factor 1]"
        << " [--ep-factor 5]"
        << " [--isolated-connection-factor 1]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string cur = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            ++i;
            return argv[i];
        };

        if (cur == "--dataset-type") {
            args.dataset_type = require_value(cur);
        } else if (cur == "--benchmark") {
            args.benchmark = require_value(cur);
        } else if (cur == "--graph") {
            args.graph_path = require_value(cur);
        } else if (cur == "--base") {
            args.base_path = require_value(cur);
        } else if (cur == "--payload") {
            args.payload_path = require_value(cur);
        } else if (cur == "--filters") {
            args.filters_path = require_value(cur);
        } else if (cur == "--out-dir") {
            args.out_dir = require_value(cur);
        } else if (cur == "--threads") {
            args.threads = std::stoi(require_value(cur));
        } else if (cur == "--nfilters") {
            args.nfilters = std::stoi(require_value(cur));
        } else if (cur == "--steiner-factor") {
            args.steiner_factor = std::stoi(require_value(cur));
        } else if (cur == "--ep-factor") {
            args.ep_factor = std::stoi(require_value(cur));
        } else if (cur == "--isolated-connection-factor") {
            args.isolated_connection_factor = std::stoi(require_value(cur));
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

    if (args.benchmark.empty()) {
        throw std::runtime_error("--benchmark is required");
    }
    ensure_readable_file(args.graph_path, "--graph");
    ensure_readable_file(args.base_path, "--base");

    if ((args.dataset_type == "laion" || args.dataset_type == "hnm") && args.payload_path.empty()) {
        throw std::runtime_error("--payload is required for laion/hnm");
    }
    if (!args.payload_path.empty()) {
        ensure_readable_file(args.payload_path, "--payload");
    }
    if (!args.filters_path.empty()) {
        ensure_readable_file(args.filters_path, "--filters");
    }

    if (args.threads <= 0) {
        throw std::runtime_error("--threads must be > 0");
    }
    if (args.nfilters <= 0 || args.nfilters > 256) {
        throw std::runtime_error("--nfilters must be in [1, 256]");
    }
    if (args.steiner_factor < 0 || args.ep_factor < 0 || args.isolated_connection_factor < 0) {
        throw std::runtime_error("--steiner-factor, --ep-factor, --isolated-connection-factor must be >= 0");
    }

    return args;
}

DatasetInfo read_base_info(const std::string& base_path) {
    int num = 0;
    int dim = 0;
    float* data = nullptr;

    if (base_path.size() >= 6 && base_path.substr(base_path.size() - 6) == ".bvecs") {
        data = read_bvecs(base_path.c_str(), num, dim);
    } else {
        data = read_fvecs(base_path.c_str(), num, dim);
    }

    if (num <= 0 || dim <= 0) {
        delete[] data;
        throw std::runtime_error("Failed to read valid base vectors from: " + base_path);
    }
    delete[] data;
    return DatasetInfo{num, dim};
}

std::vector<json> read_jsonl_rows(const std::string& payload_path, size_t limit_rows) {
    std::ifstream in(payload_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload: " + payload_path);
    }

    std::vector<json> rows;
    rows.reserve(limit_rows);

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ++line_no;
        try {
            rows.push_back(json::parse(line));
            if (rows.size() >= limit_rows) {
                break;
            }
        } catch (...) {
            // Keep row position alignment by storing null row for malformed lines.
            rows.push_back(json::object());
            if (rows.size() >= limit_rows) {
                break;
            }
        }
    }

    if (rows.size() < limit_rows) {
        std::ostringstream oss;
        oss << "Payload rows (" << rows.size() << ") are fewer than base vectors (" << limit_rows << ")";
        throw std::runtime_error(oss.str());
    }

    return rows;
}

std::unordered_map<std::string, std::vector<std::string>> parse_filters_json(
    const std::string& filters_path) {
    std::unordered_map<std::string, std::vector<std::string>> value_map;
    if (filters_path.empty()) {
        return value_map;
    }

    std::ifstream in(filters_path);
    if (!in.is_open()) {
        return value_map;
    }

    json j;
    try {
        in >> j;
    } catch (...) {
        return value_map;
    }

    if (!j.is_array()) {
        return value_map;
    }

    for (const auto& item : j) {
        if (!item.is_object() || !item.contains("name") || !item.contains("values")) {
            continue;
        }
        if (!item["name"].is_string() || !item["values"].is_array()) {
            continue;
        }

        const std::string name = item["name"].get<std::string>();
        std::vector<std::string> values;
        for (const auto& v : item["values"]) {
            if (v.is_string()) {
                values.push_back(v.get<std::string>());
            } else {
                values.push_back(v.dump());
            }
        }
        value_map[name] = std::move(values);
    }

    return value_map;
}

std::string json_value_to_string(const json& row, const std::string& key, bool* missing) {
    if (!row.is_object() || !row.contains(key) || row[key].is_null()) {
        if (missing) {
            *missing = true;
        }
        return "";
    }

    const json& v = row[key];
    if (missing) {
        *missing = false;
    }

    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_unsigned()) {
        return std::to_string(v.get<unsigned long long>());
    }
    if (v.is_number_float()) {
        std::ostringstream oss;
        oss << std::setprecision(17) << v.get<double>();
        return oss.str();
    }
    return v.dump();
}

std::vector<AttributeData> build_sift_attributes(size_t row_count, int nfilters_local) {
    std::vector<AttributeData> attrs;
    attrs.push_back(AttributeData{"synthetic_id_bucket", std::vector<std::string>(row_count)});

    const int max_elements_per_group = static_cast<int>(std::ceil(static_cast<double>(row_count) / nfilters_local));
    for (size_t i = 0; i < row_count; ++i) {
        int gid = static_cast<int>(i / static_cast<size_t>(max_elements_per_group));
        if (gid >= nfilters_local) {
            gid = nfilters_local - 1;
        }
        attrs[0].values[i] = std::to_string(gid);
    }
    return attrs;
}

std::vector<AttributeData> build_laion_attributes(const std::vector<json>& rows) {
    std::vector<AttributeData> attrs;
    attrs.reserve(kLaionKeys.size());

    for (const auto& key : kLaionKeys) {
        AttributeData attr;
        attr.key = key;
        attr.values.reserve(rows.size());
        for (const auto& row : rows) {
            bool missing = false;
            attr.values.push_back(json_value_to_string(row, key, &missing));
        }
        attrs.push_back(std::move(attr));
    }
    return attrs;
}

std::vector<AttributeData> build_hnm_attributes(const std::vector<json>& rows) {
    std::set<std::string> keys;
    for (const auto& row : rows) {
        if (!row.is_object()) {
            continue;
        }
        for (auto it = row.begin(); it != row.end(); ++it) {
            if (kHnmExcludedKeys.count(it.key()) == 0) {
                keys.insert(it.key());
            }
        }
    }

    std::vector<AttributeData> attrs;
    attrs.reserve(keys.size());
    for (const auto& key : keys) {
        AttributeData attr;
        attr.key = key;
        attr.values.reserve(rows.size());
        for (const auto& row : rows) {
            bool missing = false;
            attr.values.push_back(json_value_to_string(row, key, &missing));
        }
        attrs.push_back(std::move(attr));
    }
    return attrs;
}

bool parse_double_strict(const std::string& s, double* out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        return false;
    }
    if (out) {
        *out = v;
    }
    return true;
}

EncodedAttribute encode_attribute(
    const AttributeData& attr,
    int nfilters_local,
    const std::unordered_map<std::string, std::vector<std::string>>& filter_value_map) {
    EncodedAttribute out;
    out.key = attr.key;
    out.fid.resize(attr.values.size(), static_cast<uint8_t>(255));

    size_t missing_count = 0;
    std::vector<double> numeric_values(attr.values.size(), 0.0);
    bool all_numeric = true;
    std::set<std::string> unique_values;

    for (size_t i = 0; i < attr.values.size(); ++i) {
        std::string v = trim_copy(strip_outer_quotes(attr.values[i]));
        if (v.empty()) {
            ++missing_count;
            continue;
        }

        unique_values.insert(v);
        double d = 0.0;
        if (!parse_double_strict(v, &d)) {
            all_numeric = false;
        } else {
            numeric_values[i] = d;
        }
    }

    out.missing_count = missing_count;
    out.unique_non_missing = unique_values.size();

    // Reserve one bucket for missing values.
    const int missing_bucket = nfilters_local - 1;
    const int usable_bins = std::max(1, missing_bucket);

    // Prefer categorical mapping when unique values fit in available bins.
    if (unique_values.size() <= static_cast<size_t>(usable_bins)) {
        out.encoding = "categorical_sorted";
        out.numeric = false;

        auto map_it = filter_value_map.find(attr.key);
        if (map_it != filter_value_map.end() && !map_it->second.empty()) {
            int next = 0;
            for (const auto& raw_value : map_it->second) {
                const std::string norm = trim_copy(strip_outer_quotes(raw_value));
                if (norm.empty()) {
                    continue;
                }
                if (out.category_map.find(norm) == out.category_map.end() && next < usable_bins) {
                    out.category_map[norm] = next++;
                }
            }
            for (const auto& v : unique_values) {
                if (out.category_map.find(v) == out.category_map.end() && next < usable_bins) {
                    out.category_map[v] = next++;
                }
            }
        } else {
            int idx = 0;
            for (const auto& v : unique_values) {
                out.category_map[v] = idx++;
            }
        }

        for (size_t i = 0; i < attr.values.size(); ++i) {
            std::string v = trim_copy(strip_outer_quotes(attr.values[i]));
            if (v.empty()) {
                out.fid[i] = static_cast<uint8_t>(missing_bucket);
                continue;
            }
            auto it = out.category_map.find(v);
            if (it == out.category_map.end()) {
                out.fid[i] = static_cast<uint8_t>(missing_bucket);
            } else {
                out.fid[i] = static_cast<uint8_t>(it->second);
            }
        }

        out.used_bins = static_cast<int>(out.category_map.size()) + (missing_count > 0 ? 1 : 0);
        return out;
    }

    // Numeric quantization fallback when cardinality is high.
    out.encoding = "numeric_minmax_quantized";
    out.numeric = true;

    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < attr.values.size(); ++i) {
        std::string v = trim_copy(strip_outer_quotes(attr.values[i]));
        if (v.empty()) {
            continue;
        }
        double d = 0.0;
        if (!parse_double_strict(v, &d)) {
            continue;
        }
        min_v = std::min(min_v, d);
        max_v = std::max(max_v, d);
    }

    if (!std::isfinite(min_v) || !std::isfinite(max_v) || max_v <= min_v) {
        // Degenerate case: map all present values to zero and missing to reserved bucket.
        min_v = 0.0;
        max_v = 0.0;
        for (size_t i = 0; i < attr.values.size(); ++i) {
            std::string v = trim_copy(strip_outer_quotes(attr.values[i]));
            out.fid[i] = v.empty() ? static_cast<uint8_t>(missing_bucket) : static_cast<uint8_t>(0);
        }
        out.used_bins = 1 + (missing_count > 0 ? 1 : 0);
        out.min_value = min_v;
        out.max_value = max_v;
        return out;
    }

    const double range = max_v - min_v;
    const int bins = usable_bins;
    for (size_t i = 0; i < attr.values.size(); ++i) {
        std::string v = trim_copy(strip_outer_quotes(attr.values[i]));
        if (v.empty()) {
            out.fid[i] = static_cast<uint8_t>(missing_bucket);
            continue;
        }
        double d = 0.0;
        if (!parse_double_strict(v, &d)) {
            out.fid[i] = static_cast<uint8_t>(missing_bucket);
            continue;
        }

        int bucket = 0;
        if (range > 0.0) {
            const double normalized = (d - min_v) / range;
            int idx = static_cast<int>(normalized * bins);
            if (idx < 0) {
                idx = 0;
            }
            if (idx >= bins) {
                idx = bins - 1;
            }
            bucket = idx;
        }
        out.fid[i] = static_cast<uint8_t>(bucket);
    }

    out.min_value = min_v;
    out.max_value = max_v;
    out.used_bins = bins + (missing_count > 0 ? 1 : 0);
    return out;
}

std::vector<std::bitset<256>> build_tb_for_attribute(
    hnswlib::HierarchicalNSW<float>* index,
    const std::vector<uint8_t>& fid_values,
    int nfilters_local,
    int base_dim,
    int steiner_factor_local,
    int ep_factor_local,
    float random_rate) {
    // post_proc utilities still read these globals directly.
    filter_ids = fid_values;

    std::vector<std::bitset<256>> local_connector_bits(nElements);
    std::vector<std::bitset<256>> connections = analyze_graph_connections(index, fid_values);

    std::vector<int> group_counts(nfilters_local, 0);
    for (uint8_t g : fid_values) {
        if (static_cast<size_t>(g) < group_counts.size()) {
            ++group_counts[g];
        }
    }

    for (int group_id = 0; group_id < nfilters_local; ++group_id) {
        if (group_counts[group_id] == 0) {
            continue;
        }

        auto isolated_points = leaf_search(index, static_cast<size_t>(group_id), fid_values);
        auto paths = search_leaf_path_4(index, isolated_points, connections, fid_values, static_cast<size_t>(group_id));
        local_connector_bits = connection_bit_generator(local_connector_bits, paths, static_cast<size_t>(group_id), random_rate);

        std::vector<int> node_indices;
        node_indices.reserve(group_counts[group_id]);
        for (int i = 0; i < nElements; ++i) {
            if (fid_values[i] == static_cast<uint8_t>(group_id)) {
                local_connector_bits[i].set(group_id);
                node_indices.push_back(i);
            }
        }

        // Keep global connector_bits aligned for helper routines that rely on globals.
        connector_bits = local_connector_bits;
        std::vector<std::vector<int>> cluster_node_info = find_num_cluster(index, node_indices, static_cast<size_t>(group_id));
        if (cluster_node_info.size() > 1) {
            std::vector<int> terminal_nodes = getTerminalNodes(cluster_node_info, steiner_factor_local);
            std::vector<int> connected_path = get_steiner_tree(index, base_dim, terminal_nodes);
            for (int node : connected_path) {
                local_connector_bits[node].set(group_id);
            }
        }

        if (!cluster_node_info.empty()) {
            std::vector<int> ep_ids;
            ep_ids.reserve(nElements / 8);
            for (int internal_id = 0; internal_id < nElements; ++internal_id) {
                auto* linklist = index->get_linklist_at_level(static_cast<tableint>(internal_id), 1);
                if (linklist != nullptr) {
                    ep_ids.push_back(internal_id);
                }
            }
            std::vector<std::vector<int>> connector_paths_from_eps =
                findConnectorPathsFromEPs(index, ep_ids, static_cast<size_t>(group_id), ep_factor_local);
            for (const auto& path : connector_paths_from_eps) {
                for (int node : path) {
                    local_connector_bits[node].set(group_id);
                }
            }
        }
    }

    return local_connector_bits;
}

void write_manifest(
    const fs::path& manifest_path,
    const Args& args,
    const DatasetInfo& ds,
    const std::vector<EncodedAttribute>& attrs,
    const std::vector<std::string>& fid_files,
    const std::vector<std::string>& tb_files) {
    json j;
    j["dataset_type"] = args.dataset_type;
    j["benchmark"] = args.benchmark;
    j["graph_path"] = args.graph_path;
    j["base_path"] = args.base_path;
    j["payload_path"] = args.payload_path;
    j["filters_path"] = args.filters_path;
    j["n_elements"] = ds.base_num;
    j["dimension"] = ds.base_dim;
    j["threads"] = args.threads;
    j["nfilters"] = args.nfilters;
    j["steiner_factor"] = args.steiner_factor;
    j["ep_factor"] = args.ep_factor;
    j["isolated_connection_factor"] = args.isolated_connection_factor;

    j["attributes"] = json::array();
    for (size_t i = 0; i < attrs.size(); ++i) {
        const auto& a = attrs[i];
        json item;
        item["key"] = a.key;
        item["encoding"] = a.encoding;
        item["numeric"] = a.numeric;
        item["unique_non_missing"] = a.unique_non_missing;
        item["missing_count"] = a.missing_count;
        item["used_bins"] = a.used_bins;
        item["fid_file"] = fid_files[i];
        item["tb_file"] = tb_files[i];
        if (a.numeric) {
            item["min_value"] = a.min_value;
            item["max_value"] = a.max_value;
        }
        if (!a.category_map.empty()) {
            json cat;
            for (const auto& kv : a.category_map) {
                cat[kv.first] = kv.second;
            }
            item["category_map"] = cat;
        }
        j["attributes"].push_back(std::move(item));
    }

    std::ofstream out(manifest_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write manifest: " + manifest_path.string());
    }
    out << std::setw(2) << j << "\n";
}

void write_log(
    const fs::path& log_path,
    const Args& args,
    const DatasetInfo& ds,
    const std::vector<EncodedAttribute>& attrs,
    const std::vector<std::string>& fid_files,
    const std::vector<std::string>& tb_files) {
    std::ofstream out(log_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write log: " + log_path.string());
    }

    out << "build_FID_TB log\n";
    out << "dataset_type: " << args.dataset_type << "\n";
    out << "benchmark: " << args.benchmark << "\n";
    out << "nElements: " << ds.base_num << "\n";
    out << "base_dim: " << ds.base_dim << "\n";
    out << "threads: " << args.threads << "\n";
    out << "nfilters: " << args.nfilters << "\n";
    out << "steiner_factor: " << args.steiner_factor << "\n";
    out << "ep_factor: " << args.ep_factor << "\n";
    out << "isolated_connection_factor: " << args.isolated_connection_factor << "\n\n";

    out << "attributes_used: " << attrs.size() << "\n";
    for (size_t i = 0; i < attrs.size(); ++i) {
        const auto& a = attrs[i];
        out << "- key: " << a.key << "\n";
        out << "  encoding: " << a.encoding << "\n";
        out << "  unique_non_missing: " << a.unique_non_missing << "\n";
        out << "  missing_count: " << a.missing_count << "\n";
        out << "  used_bins: " << a.used_bins << "\n";
        if (a.numeric) {
            out << "  min: " << a.min_value << "\n";
            out << "  max: " << a.max_value << "\n";
        }
        out << "  fid: " << fid_files[i] << "\n";
        out << "  tb: " << tb_files[i] << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        dataset_type = args.dataset_type;
        nThreads = args.threads;
        nFilters = args.nfilters;
        steiner_factor = args.steiner_factor;
        ep_factor = args.ep_factor;
        isolated_connection_factor = args.isolated_connection_factor;

        DatasetInfo ds = read_base_info(args.base_path);
        nElements = ds.base_num;

        hnswlib::L2Space space(ds.base_dim);
        auto* index = new hnswlib::HierarchicalNSW<float>(&space, args.graph_path);

        std::vector<AttributeData> attributes;
        std::unordered_map<std::string, std::vector<std::string>> filter_value_map =
            parse_filters_json(args.filters_path);

        if (args.dataset_type == "sift") {
            attributes = build_sift_attributes(static_cast<size_t>(ds.base_num), args.nfilters);
        } else if (args.dataset_type == "laion") {
            std::vector<json> rows = read_jsonl_rows(args.payload_path, static_cast<size_t>(ds.base_num));
            attributes = build_laion_attributes(rows);
        } else {
            std::vector<json> rows = read_jsonl_rows(args.payload_path, static_cast<size_t>(ds.base_num));
            attributes = build_hnm_attributes(rows);
        }

        if (attributes.empty()) {
            throw std::runtime_error("No attributes selected for FID/TB generation");
        }

        fs::path benchmark_dir = fs::path(args.out_dir) / args.benchmark;
        fs::create_directories(benchmark_dir);

        std::vector<EncodedAttribute> encoded_attrs;
        encoded_attrs.reserve(attributes.size());

        std::vector<std::string> fid_files;
        std::vector<std::string> tb_files;
        fid_files.reserve(attributes.size());
        tb_files.reserve(attributes.size());

        for (const auto& attr : attributes) {
            EncodedAttribute encoded = encode_attribute(attr, args.nfilters, filter_value_map);

            std::vector<std::bitset<256>> tb = build_tb_for_attribute(
                index,
                encoded.fid,
                args.nfilters,
                ds.base_dim,
                args.steiner_factor,
                args.ep_factor,
                0.0f);

            const std::string safe_key = sanitize_key_for_filename(attr.key);
            const std::string fid_name = args.benchmark + "_" + safe_key + "_fid.bin";
            const std::string tb_name = args.benchmark + "_" + safe_key + "_tb.bin";

            const fs::path fid_path = benchmark_dir / fid_name;
            const fs::path tb_path = benchmark_dir / tb_name;

            saveIDs_uint8(encoded.fid, fid_path.string(), args.nfilters);
            saveBitsets256(tb, tb_path.string());

            encoded_attrs.push_back(std::move(encoded));
            fid_files.push_back(fid_name);
            tb_files.push_back(tb_name);

            std::cout << "Built FID/TB for attribute: " << attr.key << std::endl;
        }

        write_manifest(benchmark_dir / "manifest.json", args, ds, encoded_attrs, fid_files, tb_files);
        write_log(benchmark_dir / "build_fid_tb.log", args, ds, encoded_attrs, fid_files, tb_files);

        delete index;

        std::cout << "Done. Outputs under: " << benchmark_dir.string() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
