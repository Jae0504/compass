#pragma once

#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace filter_search_io {

struct VecFileInfo {
    int num = 0;
    int dim = 0;
};

struct IndexFileInfo {
    size_t cur_element_count = 0;
    size_t data_size_bytes = 0;
    size_t size_data_per_element = 0;
    size_t offset_data = 0;
    size_t label_offset = 0;
};

template <typename T>
struct DenseVectors {
    int num = 0;
    int dim = 0;
    std::vector<T> values;
};

using MetadataRow = std::unordered_map<std::string, std::string>;

struct MetadataTable {
    std::unordered_map<size_t, MetadataRow> rows;
    size_t total_labels = 0;
    size_t populated_rows = 0;
    size_t missing_rows = 0;
    size_t invalid_rows = 0;
    size_t dropped_rows = 0;
};

inline bool ends_with(const std::string& text, const std::string& suffix) {
    if (suffix.size() > text.size()) {
        return false;
    }
    return text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string trim_copy(const std::string& in) {
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

inline bool try_parse_size_t(const std::string& in, size_t* out) {
    if (in.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(in.c_str(), &end, 10);
    if (errno != 0 || end == in.c_str() || *end != '\0') {
        return false;
    }
    if (out != nullptr) {
        *out = static_cast<size_t>(v);
    }
    return true;
}

inline std::string json_value_to_string(const json& v) {
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
        return std::to_string(v.get<double>());
    }
    return v.dump();
}

template <typename T>
inline void read_binary_pod(std::ifstream& in, T* out, const std::string& name) {
    if (!in.read(reinterpret_cast<char*>(out), sizeof(T))) {
        throw std::runtime_error("Failed to read index header field: " + name);
    }
}

inline IndexFileInfo inspect_hnsw_index_file(const std::string& graph_path) {
    std::ifstream in(graph_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open graph index: " + graph_path);
    }

    size_t offset_level0 = 0;
    size_t max_elements = 0;
    size_t cur_element_count = 0;
    size_t size_data_per_element = 0;
    size_t label_offset = 0;
    size_t offset_data = 0;
    int maxlevel = 0;
    unsigned int enterpoint_node = 0;
    size_t maxM = 0;
    size_t maxM0 = 0;
    size_t M = 0;
    double mult = 0.0;
    size_t ef_construction = 0;

    read_binary_pod(in, &offset_level0, "offsetLevel0");
    read_binary_pod(in, &max_elements, "max_elements");
    read_binary_pod(in, &cur_element_count, "cur_element_count");
    read_binary_pod(in, &size_data_per_element, "size_data_per_element");
    read_binary_pod(in, &label_offset, "label_offset");
    read_binary_pod(in, &offset_data, "offsetData");
    read_binary_pod(in, &maxlevel, "maxlevel");
    read_binary_pod(in, &enterpoint_node, "enterpoint_node");
    read_binary_pod(in, &maxM, "maxM");
    read_binary_pod(in, &maxM0, "maxM0");
    read_binary_pod(in, &M, "M");
    read_binary_pod(in, &mult, "mult");
    read_binary_pod(in, &ef_construction, "ef_construction");

    (void)offset_level0;
    (void)max_elements;
    (void)maxlevel;
    (void)enterpoint_node;
    (void)maxM;
    (void)maxM0;
    (void)M;
    (void)mult;
    (void)ef_construction;

    if (label_offset < offset_data) {
        throw std::runtime_error("Corrupted graph index header: label_offset < offset_data");
    }

    const size_t data_size_bytes = label_offset - offset_data;
    if (data_size_bytes == 0) {
        throw std::runtime_error("Corrupted graph index header: data_size_bytes is zero");
    }
    if (size_data_per_element == 0) {
        throw std::runtime_error("Corrupted graph index header: size_data_per_element is zero");
    }

    IndexFileInfo out;
    out.cur_element_count = cur_element_count;
    out.data_size_bytes = data_size_bytes;
    out.size_data_per_element = size_data_per_element;
    out.offset_data = offset_data;
    out.label_offset = label_offset;
    return out;
}

inline VecFileInfo inspect_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open fvecs: " + path);
    }
    int32_t dim = 0;
    if (!in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t)) || dim <= 0) {
        throw std::runtime_error("Invalid fvecs header: " + path);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    const std::streamoff bytes_per = static_cast<std::streamoff>(sizeof(int32_t)) +
                                     static_cast<std::streamoff>(dim) * static_cast<std::streamoff>(sizeof(float));
    if (bytes_per <= 0 || file_size <= 0 || file_size % bytes_per != 0) {
        throw std::runtime_error("Corrupted fvecs size/layout: " + path);
    }
    const int num = static_cast<int>(file_size / bytes_per);
    return VecFileInfo{num, static_cast<int>(dim)};
}

inline VecFileInfo inspect_bvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open bvecs: " + path);
    }
    int32_t dim = 0;
    if (!in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t)) || dim <= 0) {
        throw std::runtime_error("Invalid bvecs header: " + path);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    const std::streamoff bytes_per = static_cast<std::streamoff>(sizeof(int32_t)) +
                                     static_cast<std::streamoff>(dim) * static_cast<std::streamoff>(sizeof(uint8_t));
    if (bytes_per <= 0 || file_size <= 0 || file_size % bytes_per != 0) {
        throw std::runtime_error("Corrupted bvecs size/layout: " + path);
    }
    const int num = static_cast<int>(file_size / bytes_per);
    return VecFileInfo{num, static_cast<int>(dim)};
}

inline VecFileInfo inspect_vector_file(const std::string& path) {
    if (ends_with(path, ".fvecs")) {
        return inspect_fvecs(path);
    }
    if (ends_with(path, ".bvecs")) {
        return inspect_bvecs(path);
    }
    throw std::runtime_error("Vector file must end with .fvecs or .bvecs: " + path);
}

inline DenseVectors<float> read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open fvecs: " + path);
    }

    DenseVectors<float> out;
    int32_t dim = 0;
    while (true) {
        if (!in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) {
            break;
        }
        if (dim <= 0) {
            throw std::runtime_error("Invalid dimension in fvecs: " + path);
        }
        if (out.dim == 0) {
            out.dim = dim;
        } else if (out.dim != dim) {
            throw std::runtime_error("Inconsistent dimensions in fvecs: " + path);
        }

        const size_t old_size = out.values.size();
        out.values.resize(old_size + static_cast<size_t>(dim));
        const size_t bytes = static_cast<size_t>(dim) * sizeof(float);
        if (!in.read(reinterpret_cast<char*>(out.values.data() + old_size), static_cast<std::streamsize>(bytes))) {
            throw std::runtime_error("Truncated fvecs payload: " + path);
        }
        ++out.num;
    }

    if (out.num <= 0 || out.dim <= 0) {
        throw std::runtime_error("No vectors found in fvecs: " + path);
    }
    return out;
}

inline DenseVectors<uint8_t> read_bvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open bvecs: " + path);
    }

    DenseVectors<uint8_t> out;
    int32_t dim = 0;
    while (true) {
        if (!in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) {
            break;
        }
        if (dim <= 0) {
            throw std::runtime_error("Invalid dimension in bvecs: " + path);
        }
        if (out.dim == 0) {
            out.dim = dim;
        } else if (out.dim != dim) {
            throw std::runtime_error("Inconsistent dimensions in bvecs: " + path);
        }

        const size_t old_size = out.values.size();
        out.values.resize(old_size + static_cast<size_t>(dim));
        const size_t bytes = static_cast<size_t>(dim) * sizeof(uint8_t);
        if (!in.read(reinterpret_cast<char*>(out.values.data() + old_size), static_cast<std::streamsize>(bytes))) {
            throw std::runtime_error("Truncated bvecs payload: " + path);
        }
        ++out.num;
    }

    if (out.num <= 0 || out.dim <= 0) {
        throw std::runtime_error("No vectors found in bvecs: " + path);
    }
    return out;
}

inline std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
            continue;
        }

        if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }

    if (in_quotes) {
        throw std::runtime_error("Malformed CSV line with unmatched quote");
    }
    out.push_back(cur);
    return out;
}

inline MetadataTable load_csv_metadata(
    const std::string& path,
    const std::unordered_set<std::string>& referenced_fields,
    const std::string& id_column,
    size_t num_labels) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open metadata CSV: " + path);
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("Empty metadata CSV: " + path);
    }
    std::vector<std::string> header = parse_csv_line(header_line);
    if (!header.empty() && !header[0].empty() && static_cast<unsigned char>(header[0][0]) == 0xEF) {
        if (header[0].size() >= 3 &&
            static_cast<unsigned char>(header[0][1]) == 0xBB &&
            static_cast<unsigned char>(header[0][2]) == 0xBF) {
            header[0] = header[0].substr(3);
        }
    }
    for (std::string& col : header) {
        col = trim_copy(col);
    }

    std::unordered_map<std::string, size_t> column_to_index;
    for (size_t i = 0; i < header.size(); ++i) {
        column_to_index[header[i]] = i;
    }

    auto id_it = column_to_index.find(id_column);
    if (id_it == column_to_index.end()) {
        throw std::runtime_error("CSV missing id column '" + id_column + "'");
    }
    const size_t id_idx = id_it->second;

    std::vector<std::pair<std::string, size_t>> selected_cols;
    selected_cols.reserve(referenced_fields.size());
    for (const std::string& field : referenced_fields) {
        auto it = column_to_index.find(field);
        if (it != column_to_index.end()) {
            selected_cols.push_back(std::make_pair(field, it->second));
        }
    }
    if (selected_cols.empty()) {
        throw std::runtime_error("None of the referenced fields were found in metadata CSV");
    }

    MetadataTable table;
    table.total_labels = num_labels;

    std::string line;
    size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> row;
        try {
            row = parse_csv_line(line);
        } catch (...) {
            ++table.invalid_rows;
            continue;
        }

        if (id_idx >= row.size()) {
            ++table.invalid_rows;
            continue;
        }

        size_t label = 0;
        const std::string id_text = trim_copy(row[id_idx]);
        if (!try_parse_size_t(id_text, &label)) {
            ++table.invalid_rows;
            continue;
        }
        if (label >= num_labels) {
            ++table.dropped_rows;
            continue;
        }

        MetadataRow& dest = table.rows[label];
        for (const auto& kv : selected_cols) {
            const std::string& field = kv.first;
            const size_t col_idx = kv.second;
            if (col_idx < row.size()) {
                dest[field] = trim_copy(row[col_idx]);
            }
        }
    }

    table.populated_rows = table.rows.size();
    table.missing_rows = (num_labels >= table.populated_rows) ? (num_labels - table.populated_rows) : 0;
    return table;
}

inline MetadataTable load_jsonl_metadata(
    const std::string& path,
    const std::unordered_set<std::string>& referenced_fields,
    size_t num_labels) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open payload JSONL: " + path);
    }
    if (referenced_fields.empty()) {
        throw std::runtime_error("Referenced fields cannot be empty");
    }

    MetadataTable table;
    table.total_labels = num_labels;

    std::string line;
    size_t row_id = 0;
    while (row_id < num_labels && std::getline(in, line)) {
        if (line.empty()) {
            ++table.invalid_rows;
            ++row_id;
            continue;
        }
        json j;
        try {
            j = json::parse(line);
        } catch (...) {
            ++table.invalid_rows;
            ++row_id;
            continue;
        }

        if (j.is_object()) {
            MetadataRow dest;
            for (const std::string& field : referenced_fields) {
                auto it = j.find(field);
                if (it != j.end() && !it->is_null()) {
                    dest[field] = json_value_to_string(*it);
                }
            }
            if (!dest.empty()) {
                table.rows[row_id] = std::move(dest);
            }
        } else {
            ++table.invalid_rows;
        }
        ++row_id;
    }

    table.populated_rows = table.rows.size();
    table.missing_rows = (num_labels >= table.populated_rows) ? (num_labels - table.populated_rows) : 0;
    return table;
}

}  // namespace filter_search_io
