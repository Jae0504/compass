#include "../../nlohmann/json.hpp"

#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const std::vector<std::string> kLaionKeys = {
    "NSFW",
    "similarity",
    "original_width",
    "original_height",
};

const std::unordered_set<std::string> kHnmExcludedKeys = {
    "detail_desc",
    "prod_name",
};

struct Args {
    std::string dataset_type;
    std::string base_path;
    std::string out_path;
    std::string payload_path;
    std::string metadata_float_key;
    bool metadata_no_bucket = false;
    int nfilters = 100;
    int m = 32;
    int mbeta = 64;
    int ef_search = 64;
    int ef_construction = -1;
    int threads = 64;
};

struct DatasetData {
    std::unique_ptr<float[]> vectors;
    int num = 0;
    int dim = 0;
};

DatasetData read_fvecs_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open fvecs: " + path);
    }

    std::vector<float> values;
    values.reserve(1 << 20);
    int dim = -1;
    int count = 0;

    while (true) {
        int cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (!in) {
            break;
        }
        if (cur_dim <= 0) {
            throw std::runtime_error("Invalid dimension in fvecs: " + path);
        }
        if (dim == -1) {
            dim = cur_dim;
        } else if (cur_dim != dim) {
            throw std::runtime_error("Inconsistent dimension in fvecs: " + path);
        }

        const size_t offset = values.size();
        values.resize(offset + static_cast<size_t>(dim));
        in.read(reinterpret_cast<char*>(values.data() + offset), static_cast<size_t>(dim) * sizeof(float));
        if (!in) {
            throw std::runtime_error("Truncated fvecs payload: " + path);
        }
        ++count;
    }

    if (count <= 0 || dim <= 0) {
        throw std::runtime_error("No vectors found in fvecs: " + path);
    }

    DatasetData out;
    out.num = count;
    out.dim = dim;
    out.vectors = std::make_unique<float[]>(values.size());
    std::copy(values.begin(), values.end(), out.vectors.get());
    return out;
}

DatasetData read_bvecs_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open bvecs: " + path);
    }

    std::vector<float> values;
    values.reserve(1 << 20);
    int dim = -1;
    int count = 0;

    while (true) {
        int cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (!in) {
            break;
        }
        if (cur_dim <= 0) {
            throw std::runtime_error("Invalid dimension in bvecs: " + path);
        }
        if (dim == -1) {
            dim = cur_dim;
        } else if (cur_dim != dim) {
            throw std::runtime_error("Inconsistent dimension in bvecs: " + path);
        }

        std::vector<uint8_t> buf(static_cast<size_t>(dim));
        in.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (!in) {
            throw std::runtime_error("Truncated bvecs payload: " + path);
        }

        for (uint8_t v : buf) {
            values.push_back(static_cast<float>(v));
        }
        ++count;
    }

    if (count <= 0 || dim <= 0) {
        throw std::runtime_error("No vectors found in bvecs: " + path);
    }

    DatasetData out;
    out.num = count;
    out.dim = dim;
    out.vectors = std::make_unique<float[]>(values.size());
    std::copy(values.begin(), values.end(), out.vectors.get());
    return out;
}

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " --dataset-type <sift1m|sift|laion|hnm>"
        << " --base <path(.fvecs|.bvecs)>"
        << " --out <path(.index)>"
        << " [--payload <path(.json|.jsonl)>]"
        << " [--metadata-float-key <json field name>]"
        << " [--metadata-no-bucket]"
        << " [--nfilters 100]"
        << " [--m 32]"
        << " [--mbeta 64]"
        << " [--ef-search 64]"
        << " [--ef-construction -1]"
        << " [--threads 64]\n";
}

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
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

int parse_int_or_throw(const std::string& s, const std::string& name) {
    if (s.empty()) {
        throw std::runtime_error("Missing integer for " + name);
    }
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid integer for " + name + ": " + s);
    }
    return static_cast<int>(v);
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
        } else if (cur == "--base") {
            args.base_path = require_value(cur);
        } else if (cur == "--out") {
            args.out_path = require_value(cur);
        } else if (cur == "--payload") {
            args.payload_path = require_value(cur);
        } else if (cur == "--metadata-float-key") {
            args.metadata_float_key = require_value(cur);
        } else if (cur == "--metadata-no-bucket") {
            args.metadata_no_bucket = true;
        } else if (cur == "--nfilters") {
            args.nfilters = parse_int_or_throw(require_value(cur), "--nfilters");
        } else if (cur == "--m") {
            args.m = parse_int_or_throw(require_value(cur), "--m");
        } else if (cur == "--mbeta") {
            args.mbeta = parse_int_or_throw(require_value(cur), "--mbeta");
        } else if (cur == "--ef-search") {
            args.ef_search = parse_int_or_throw(require_value(cur), "--ef-search");
        } else if (cur == "--ef-construction") {
            args.ef_construction = parse_int_or_throw(require_value(cur), "--ef-construction");
        } else if (cur == "--threads") {
            args.threads = parse_int_or_throw(require_value(cur), "--threads");
        } else if (cur == "--help" || cur == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + cur);
        }
    }

    if (args.dataset_type.empty()) {
        throw std::runtime_error("--dataset-type is required");
    }
    if (args.dataset_type == "sift") {
        args.dataset_type = "sift1m";
    }
    if (args.dataset_type != "sift1m" && args.dataset_type != "laion" && args.dataset_type != "hnm") {
        throw std::runtime_error("--dataset-type must be one of: sift1m, sift, laion, hnm");
    }

    ensure_readable_file(args.base_path, "--base");
    if (args.out_path.empty()) {
        throw std::runtime_error("--out is required");
    }
    if (args.dataset_type != "sift1m") {
        ensure_readable_file(args.payload_path, "--payload");
    }
    if (!args.metadata_float_key.empty() && args.dataset_type == "sift1m") {
        throw std::runtime_error("--metadata-float-key is supported only for laion/hnm");
    }
    if (args.metadata_no_bucket && args.dataset_type == "sift1m") {
        throw std::runtime_error("--metadata-no-bucket is supported only for laion/hnm");
    }

    if (!ends_with(lower_copy(args.base_path), ".fvecs") && !ends_with(lower_copy(args.base_path), ".bvecs")) {
        throw std::runtime_error("--base must end with .fvecs or .bvecs");
    }
    if (args.nfilters <= 0 || args.nfilters > 256) {
        throw std::runtime_error("--nfilters must be in [1, 256]");
    }
    if (args.m <= 0 || args.mbeta <= 0) {
        throw std::runtime_error("--m and --mbeta must be > 0");
    }
    if (args.ef_search <= 0) {
        throw std::runtime_error("--ef-search must be > 0");
    }
    if (args.threads <= 0) {
        throw std::runtime_error("--threads must be > 0");
    }
    return args;
}

DatasetData read_base_vectors(const std::string& base_path) {
    if (ends_with(lower_copy(base_path), ".bvecs")) {
        return read_bvecs_raw(base_path);
    } else {
        return read_fvecs_raw(base_path);
    }
}

std::string trim_copy(const std::string& in) {
    size_t s = 0;
    while (s < in.size() && std::isspace(static_cast<unsigned char>(in[s]))) {
        ++s;
    }
    size_t e = in.size();
    while (e > s && std::isspace(static_cast<unsigned char>(in[e - 1]))) {
        --e;
    }
    return in.substr(s, e - s);
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

uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

int float_to_metadata_id(float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float size");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return static_cast<int32_t>(bits);
}

std::optional<float> json_value_to_float(const json& row, const std::string& key) {
    if (!row.is_object() || !row.contains(key) || row[key].is_null()) {
        return std::nullopt;
    }
    const json& v = row[key];
    if (v.is_number()) {
        return static_cast<float>(v.get<double>());
    }
    if (v.is_string()) {
        const std::string trimmed = trim_copy(strip_outer_quotes(v.get<std::string>()));
        if (trimmed.empty()) {
            return std::nullopt;
        }
        char* end = nullptr;
        errno = 0;
        const float parsed = std::strtof(trimmed.c_str(), &end);
        if (errno != 0 || end == trimmed.c_str() || *end != '\0') {
            return std::nullopt;
        }
        return parsed;
    }
    return std::nullopt;
}

class BucketAssigner {
public:
    explicit BucketAssigner(int nfilters) : nfilters_(nfilters) {}

    int assign(const std::string& token) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }

        int id = 0;
        if (nfilters_ <= 0) {
            id = static_cast<int>(token_to_id_.size());
        } else if (static_cast<int>(token_to_id_.size()) < nfilters_) {
            id = static_cast<int>(token_to_id_.size());
        } else {
            id = static_cast<int>(fnv1a64(token) % static_cast<uint64_t>(nfilters_));
        }
        token_to_id_[token] = id;
        return id;
    }

    size_t unique_tokens() const {
        return token_to_id_.size();
    }

private:
    int nfilters_;
    std::unordered_map<std::string, int> token_to_id_;
};

std::string token_for_laion(const json& row) {
    std::ostringstream oss;
    for (const auto& key : kLaionKeys) {
        bool missing = false;
        std::string value = trim_copy(strip_outer_quotes(json_value_to_string(row, key, &missing)));
        if (missing || value.empty()) {
            value = "__MISSING__";
        }
        oss << key << '=' << value << '|';
    }
    return oss.str();
}

std::string token_for_hnm(const json& row) {
    if (!row.is_object()) {
        return "__EMPTY_ROW__";
    }
    std::vector<std::string> keys;
    keys.reserve(row.size());
    for (auto it = row.begin(); it != row.end(); ++it) {
        if (kHnmExcludedKeys.find(it.key()) == kHnmExcludedKeys.end()) {
            keys.push_back(it.key());
        }
    }
    std::sort(keys.begin(), keys.end());
    if (keys.empty()) {
        return "__EMPTY_KEYS__";
    }

    std::ostringstream oss;
    for (const auto& key : keys) {
        bool missing = false;
        std::string value = trim_copy(strip_outer_quotes(json_value_to_string(row, key, &missing)));
        if (missing || value.empty()) {
            value = "__MISSING__";
        }
        oss << key << '=' << value << '|';
    }
    return oss.str();
}

std::vector<int> build_sift_metadata(size_t n, int nfilters) {
    std::vector<int> metadata(n, 0);
    for (size_t i = 0; i < n; ++i) {
        // Requested behavior:
        // for nfilters=100 -> first 1/100 vectors id=0, next 1/100 id=1, ...
        int gid = static_cast<int>((i * static_cast<size_t>(nfilters)) / n);
        if (gid >= nfilters) {
            gid = nfilters - 1;
        }
        metadata[i] = gid;
    }
    return metadata;
}

std::vector<int> build_metadata_from_jsonl(
        const std::string& payload_path,
        const std::string& dataset_type,
        size_t expected_rows,
        int nfilters,
        bool no_bucket) {
    std::ifstream in(payload_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload: " + payload_path);
    }

    std::vector<int> metadata;
    metadata.reserve(expected_rows);
    BucketAssigner assigner(no_bucket ? -1 : nfilters);

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ++line_no;
        json row;
        try {
            row = json::parse(line);
        } catch (...) {
            row = json::object();
        }

        const std::string token = (dataset_type == "laion") ? token_for_laion(row) : token_for_hnm(row);
        metadata.push_back(assigner.assign(token));
        if (metadata.size() >= expected_rows) {
            break;
        }
    }

    if (metadata.size() < expected_rows) {
        std::ostringstream oss;
        oss << "Payload rows (" << metadata.size() << ") are fewer than base vectors (" << expected_rows << ")";
        throw std::runtime_error(oss.str());
    }

    std::cout << "Loaded JSONL metadata: rows=" << metadata.size()
              << ", unique_tokens=" << assigner.unique_tokens()
              << ", no_bucket=" << (no_bucket ? "true" : "false")
              << std::endl;
    return metadata;
}

std::vector<int> build_metadata_from_jsonl_float_key(
        const std::string& payload_path,
        const std::string& float_key,
        size_t expected_rows) {
    std::ifstream in(payload_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload: " + payload_path);
    }

    std::vector<int> metadata;
    metadata.reserve(expected_rows);

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ++line_no;
        json row;
        try {
            row = json::parse(line);
        } catch (...) {
            std::ostringstream oss;
            oss << "Invalid JSON row at line " << line_no << " in payload: " << payload_path;
            throw std::runtime_error(oss.str());
        }

        const auto f = json_value_to_float(row, float_key);
        if (!f.has_value()) {
            std::ostringstream oss;
            oss << "Missing/non-numeric float field '" << float_key << "' at line " << line_no;
            throw std::runtime_error(oss.str());
        }
        metadata.push_back(float_to_metadata_id(*f));
        if (metadata.size() >= expected_rows) {
            break;
        }
    }

    if (metadata.size() < expected_rows) {
        std::ostringstream oss;
        oss << "Payload rows (" << metadata.size() << ") are fewer than base vectors (" << expected_rows << ")";
        throw std::runtime_error(oss.str());
    }

    std::cout << "Loaded float metadata from JSONL: rows=" << metadata.size()
              << ", field=" << float_key << std::endl;
    return metadata;
}

std::vector<int> build_metadata_from_json_array(
        const std::string& payload_path,
        const std::string& dataset_type,
        size_t expected_rows,
        int nfilters,
        bool no_bucket) {
    std::ifstream in(payload_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload: " + payload_path);
    }

    json root;
    in >> root;
    if (!root.is_array()) {
        throw std::runtime_error("JSON payload is not an array: " + payload_path);
    }
    if (root.size() < expected_rows) {
        std::ostringstream oss;
        oss << "Payload rows (" << root.size() << ") are fewer than base vectors (" << expected_rows << ")";
        throw std::runtime_error(oss.str());
    }

    std::vector<int> metadata;
    metadata.reserve(expected_rows);
    BucketAssigner assigner(no_bucket ? -1 : nfilters);
    for (size_t i = 0; i < expected_rows; ++i) {
        const json& row = root[i];
        const std::string token = (dataset_type == "laion") ? token_for_laion(row) : token_for_hnm(row);
        metadata.push_back(assigner.assign(token));
    }

    std::cout << "Loaded JSON-array metadata: rows=" << metadata.size()
              << ", unique_tokens=" << assigner.unique_tokens()
              << ", no_bucket=" << (no_bucket ? "true" : "false")
              << std::endl;
    return metadata;
}

std::vector<int> build_metadata_from_json_array_float_key(
        const std::string& payload_path,
        const std::string& float_key,
        size_t expected_rows) {
    std::ifstream in(payload_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload: " + payload_path);
    }

    json root;
    in >> root;
    if (!root.is_array()) {
        throw std::runtime_error("JSON payload is not an array: " + payload_path);
    }
    if (root.size() < expected_rows) {
        std::ostringstream oss;
        oss << "Payload rows (" << root.size() << ") are fewer than base vectors (" << expected_rows << ")";
        throw std::runtime_error(oss.str());
    }

    std::vector<int> metadata;
    metadata.reserve(expected_rows);
    for (size_t i = 0; i < expected_rows; ++i) {
        const json& row = root[i];
        const auto f = json_value_to_float(row, float_key);
        if (!f.has_value()) {
            std::ostringstream oss;
            oss << "Missing/non-numeric float field '" << float_key << "' at row index " << i;
            throw std::runtime_error(oss.str());
        }
        metadata.push_back(float_to_metadata_id(*f));
    }

    std::cout << "Loaded float metadata from JSON-array: rows=" << metadata.size()
              << ", field=" << float_key << std::endl;
    return metadata;
}

std::vector<int> build_json_metadata(
        const std::string& payload_path,
        const std::string& dataset_type,
        size_t expected_rows,
        int nfilters,
        const std::string& metadata_float_key,
        bool metadata_no_bucket) {
    if (!metadata_float_key.empty()) {
        const std::string lower = lower_copy(payload_path);
        if (ends_with(lower, ".jsonl")) {
            return build_metadata_from_jsonl_float_key(payload_path, metadata_float_key, expected_rows);
        }
        if (ends_with(lower, ".json")) {
            return build_metadata_from_json_array_float_key(payload_path, metadata_float_key, expected_rows);
        }
        try {
            return build_metadata_from_jsonl_float_key(payload_path, metadata_float_key, expected_rows);
        } catch (...) {
            return build_metadata_from_json_array_float_key(payload_path, metadata_float_key, expected_rows);
        }
    }

    const std::string lower = lower_copy(payload_path);
    if (ends_with(lower, ".jsonl")) {
        return build_metadata_from_jsonl(
                payload_path,
                dataset_type,
                expected_rows,
                nfilters,
                metadata_no_bucket);
    }
    if (ends_with(lower, ".json")) {
        return build_metadata_from_json_array(
                payload_path,
                dataset_type,
                expected_rows,
                nfilters,
                metadata_no_bucket);
    }

    // Fallback: try line-oriented JSON first, then array JSON.
    try {
        return build_metadata_from_jsonl(
                payload_path,
                dataset_type,
                expected_rows,
                nfilters,
                metadata_no_bucket);
    } catch (...) {
        return build_metadata_from_json_array(
                payload_path,
                dataset_type,
                expected_rows,
                nfilters,
                metadata_no_bucket);
    }
}

std::vector<int> build_metadata(const Args& args, size_t n) {
    if (args.dataset_type == "sift1m") {
        return build_sift_metadata(n, args.nfilters);
    }
    return build_json_metadata(
            args.payload_path,
            args.dataset_type,
            n,
            args.nfilters,
            args.metadata_float_key,
            args.metadata_no_bucket);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        DatasetData ds = read_base_vectors(args.base_path);
        const size_t n = static_cast<size_t>(ds.num);

        std::vector<int> metadata = build_metadata(args, n);
        if (metadata.size() != n) {
            throw std::runtime_error("Metadata size does not match number of base vectors");
        }

        if (args.threads > 0) {
            omp_set_num_threads(args.threads);
        }

        fs::path out_path(args.out_path);
        if (!out_path.parent_path().empty()) {
            fs::create_directories(out_path.parent_path());
        }

        std::cout << "Building ACORN index\n"
                  << "  dataset_type: " << args.dataset_type << "\n"
                  << "  base: " << args.base_path << "\n"
                  << "  out: " << args.out_path << "\n"
                  << "  metadata_float_key: "
                  << (args.metadata_float_key.empty() ? std::string("<bucketed-token-mode>") : args.metadata_float_key)
                  << "\n"
                  << "  metadata_no_bucket: " << (args.metadata_no_bucket ? "true" : "false")
                  << "\n"
                  << "  n=" << ds.num << ", d=" << ds.dim << "\n"
                  << "  nfilters=" << args.nfilters
                  << ", M=" << args.m
                  << ", M_beta=" << args.mbeta
                  << ", efSearch=" << args.ef_search
                  << ", efConstruction=" << args.ef_construction
                  << ", threads=" << args.threads << std::endl;

        faiss::IndexACORNFlat index(ds.dim, args.m, args.nfilters, metadata, args.mbeta);
        index.acorn.efSearch = args.ef_search;
        if (args.ef_construction > 0) {
            index.acorn.efConstruction = args.ef_construction;
        }

        index.add(ds.num, ds.vectors.get());
        faiss::write_index(&index, args.out_path.c_str());

        std::cout << "Saved ACORN index: " << args.out_path << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
