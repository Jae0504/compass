#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
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
#include <utility>
#include <vector>

#include <lz4.h>
#include <zlib.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr size_t kChunkSizeBytes = 2 * 1024 * 1024;
const std::vector<std::string> kLaionKeys = {
        "NSFW",
        "similarity",
        "original_width",
        "original_height"};

struct Args {
    std::string hnm_fvecs;
    std::string hnm_jsonl;
    std::string laion_fvecs;
    std::string laion_jsonl;

    int m = 32;
    int ef_construct = 128;
    int acorn_gamma = 12;
    int acorn_mbeta = -1; // defaults to 2 * m
    size_t max_rows = 0;   // 0 means all rows

    std::string out_txt = "profiling.txt";
    std::string out_json = "profiling.json";
};

struct FvecData {
    size_t n = 0;
    size_t d = 0;
    std::vector<float> values;
};

using JsonRow = std::map<std::string, std::string>;

struct MetadataBuildResult {
    std::vector<std::string> selected_keys;
    std::vector<uint8_t> metadata_blob;
    std::vector<int> metadata_ids;
    size_t unique_metadata_ids = 0;
};

struct DatasetResult {
    std::string name;
    size_t rows = 0;
    size_t dim = 0;
    size_t selected_metadata_columns = 0;
    size_t unique_metadata_ids = 0;

    size_t embedding_bytes = 0;
    size_t metadata_original_bytes = 0;
    size_t metadata_lz4_bytes = 0;
    size_t metadata_deflate_bytes = 0;

    size_t hnsw_index_bytes = 0;
    size_t hnsw_graph_bytes = 0;
    size_t acorn_index_bytes = 0;
    size_t acorn_graph_bytes = 0;

    size_t original_plus_hnsw_bytes = 0;
    size_t lz4_plus_hnsw_bytes = 0;
    size_t deflate_plus_hnsw_bytes = 0;
    size_t acorn_metadata_bytes = 0;

    std::vector<std::string> selected_keys;
    std::vector<std::string> notes;
};

void usage(const char* argv0) {
    std::cerr
            << "Usage:\n"
            << "  " << argv0
            << " --hnm-fvecs <path> --hnm-jsonl <path>"
            << " --laion-fvecs <path> --laion-jsonl <path>"
            << " [--m 32] [--ef-construct 128]"
            << " [--acorn-gamma 12] [--acorn-mbeta 64] [--max-rows 0]"
            << " [--out-txt profiling.txt] [--out-json profiling.json]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string cur = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::ostringstream oss;
                oss << "Missing value for " << flag;
                throw std::runtime_error(oss.str());
            }
            ++i;
            return argv[i];
        };

        if (cur == "--hnm-fvecs") {
            args.hnm_fvecs = require_value("--hnm-fvecs");
        } else if (cur == "--hnm-jsonl") {
            args.hnm_jsonl = require_value("--hnm-jsonl");
        } else if (cur == "--laion-fvecs") {
            args.laion_fvecs = require_value("--laion-fvecs");
        } else if (cur == "--laion-jsonl") {
            args.laion_jsonl = require_value("--laion-jsonl");
        } else if (cur == "--m") {
            args.m = std::stoi(require_value("--m"));
        } else if (cur == "--ef-construct") {
            args.ef_construct = std::stoi(require_value("--ef-construct"));
        } else if (cur == "--acorn-gamma") {
            args.acorn_gamma = std::stoi(require_value("--acorn-gamma"));
        } else if (cur == "--acorn-mbeta") {
            args.acorn_mbeta = std::stoi(require_value("--acorn-mbeta"));
        } else if (cur == "--max-rows") {
            args.max_rows = static_cast<size_t>(std::stoull(require_value("--max-rows")));
        } else if (cur == "--out-txt") {
            args.out_txt = require_value("--out-txt");
        } else if (cur == "--out-json") {
            args.out_json = require_value("--out-json");
        } else if (cur == "-h" || cur == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::ostringstream oss;
            oss << "Unknown argument: " << cur;
            throw std::runtime_error(oss.str());
        }
    }

    if (args.hnm_fvecs.empty() || args.hnm_jsonl.empty() ||
        args.laion_fvecs.empty() || args.laion_jsonl.empty()) {
        throw std::runtime_error("Missing required dataset paths.");
    }

    if (args.m <= 0 || args.ef_construct <= 0 || args.acorn_gamma <= 0) {
        throw std::runtime_error("m, ef-construct, and acorn-gamma must be > 0.");
    }
    if (args.acorn_mbeta == 0) {
        throw std::runtime_error("acorn-mbeta cannot be 0.");
    }

    return args;
}

void ensure_file_exists(const std::string& path) {
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
        std::ostringstream oss;
        oss << "File not found: " << path;
        throw std::runtime_error(oss.str());
    }
}

FvecData read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::ostringstream oss;
        oss << "Failed to open fvecs file: " << path;
        throw std::runtime_error(oss.str());
    }

    FvecData out;
    int32_t d_i = 0;
    while (true) {
        if (!in.read(reinterpret_cast<char*>(&d_i), sizeof(int32_t))) {
            break;
        }
        if (d_i <= 0) {
            std::ostringstream oss;
            oss << "Invalid vector dimension " << d_i << " in " << path;
            throw std::runtime_error(oss.str());
        }
        if (out.d == 0) {
            out.d = static_cast<size_t>(d_i);
        } else if (out.d != static_cast<size_t>(d_i)) {
            std::ostringstream oss;
            oss << "Inconsistent vector dimension in " << path;
            throw std::runtime_error(oss.str());
        }

        const size_t old_size = out.values.size();
        out.values.resize(old_size + out.d);
        const size_t bytes = out.d * sizeof(float);
        if (!in.read(reinterpret_cast<char*>(out.values.data() + old_size), static_cast<std::streamsize>(bytes))) {
            std::ostringstream oss;
            oss << "Truncated fvecs payload in " << path;
            throw std::runtime_error(oss.str());
        }
        ++out.n;
    }

    if (out.n == 0 || out.d == 0) {
        std::ostringstream oss;
        oss << "No vectors found in " << path;
        throw std::runtime_error(oss.str());
    }
    return out;
}

JsonRow parse_json_line(const std::string& line) {
    JsonRow result;
    size_t pos = 0;
    while (pos < line.size()) {
        const size_t key_start = line.find('"', pos);
        if (key_start == std::string::npos) {
            break;
        }
        const size_t key_end = line.find('"', key_start + 1);
        if (key_end == std::string::npos) {
            break;
        }
        const std::string key = line.substr(key_start + 1, key_end - key_start - 1);

        const size_t colon = line.find(':', key_end);
        if (colon == std::string::npos) {
            break;
        }

        size_t value_start = colon + 1;
        while (value_start < line.size() &&
               std::isspace(static_cast<unsigned char>(line[value_start]))) {
            ++value_start;
        }

        std::string value;
        if (value_start < line.size() && line[value_start] == '"') {
            size_t value_end = line.find('"', value_start + 1);
            if (value_end == std::string::npos) {
                break;
            }
            value = line.substr(value_start + 1, value_end - value_start - 1);
            pos = value_end + 1;
        } else {
            const size_t value_end = line.find_first_of(",}", value_start);
            if (value_end == std::string::npos) {
                break;
            }
            value = line.substr(value_start, value_end - value_start);
            while (!value.empty() &&
                   std::isspace(static_cast<unsigned char>(value.back()))) {
                value.pop_back();
            }
            pos = value_end + 1;
        }

        result[key] = value;
    }
    return result;
}

std::vector<JsonRow> read_jsonl_rows(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::ostringstream oss;
        oss << "Failed to open jsonl file: " << path;
        throw std::runtime_error(oss.str());
    }

    std::vector<JsonRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        rows.push_back(parse_json_line(line));
    }
    return rows;
}

std::vector<std::string> select_metadata_keys(
        const std::vector<JsonRow>& rows,
        const std::string& dataset_name) {
    if (dataset_name == "laion") {
        return kLaionKeys;
    }

    std::set<std::string> keyset;
    for (const auto& row : rows) {
        for (const auto& kv : row) {
            if (kv.first != "detail_desc" && kv.first != "prod_name") {
                keyset.insert(kv.first);
            }
        }
    }
    return std::vector<std::string>(keyset.begin(), keyset.end());
}

std::string get_value_or_empty(const JsonRow& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) {
        return "";
    }
    return it->second;
}

MetadataBuildResult build_metadata_payload(
        const std::vector<JsonRow>& rows,
        const std::string& dataset_name,
        size_t row_limit) {
    MetadataBuildResult out;
    out.selected_keys = select_metadata_keys(rows, dataset_name);

    out.metadata_ids.reserve(row_limit);
    out.metadata_blob.reserve(row_limit * std::max<size_t>(1, out.selected_keys.size()) * 8);

    std::unordered_map<std::string, int> metadata_to_id;
    metadata_to_id.reserve(row_limit * 2 + 1);

    for (size_t i = 0; i < row_limit; ++i) {
        const JsonRow& row = rows[i];
        std::string token;
        token.reserve(out.selected_keys.size() * 16);

        for (const auto& key : out.selected_keys) {
            const std::string value = get_value_or_empty(row, key);
            out.metadata_blob.insert(out.metadata_blob.end(), value.begin(), value.end());
            out.metadata_blob.push_back('\n');

            token.append(value);
            token.push_back('\x1F');
        }

        auto it = metadata_to_id.find(token);
        if (it == metadata_to_id.end()) {
            const size_t next = metadata_to_id.size();
            if (next > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Too many unique metadata IDs for int encoding.");
            }
            const int id = static_cast<int>(next);
            auto inserted = metadata_to_id.emplace(std::move(token), id);
            out.metadata_ids.push_back(inserted.first->second);
        } else {
            out.metadata_ids.push_back(it->second);
        }
    }

    out.unique_metadata_ids = metadata_to_id.size();
    return out;
}

size_t compress_lz4_chunked(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return 0;
    }
    size_t total = 0;
    size_t offset = 0;
    while (offset < data.size()) {
        const size_t len = std::min(kChunkSizeBytes, data.size() - offset);
        const int len_i = static_cast<int>(len);
        const int bound = LZ4_compressBound(len_i);
        if (bound <= 0) {
            throw std::runtime_error("LZ4_compressBound failed.");
        }
        std::vector<char> dst(static_cast<size_t>(bound));
        const int comp = LZ4_compress_default(
                reinterpret_cast<const char*>(data.data() + offset),
                dst.data(),
                len_i,
                bound);
        if (comp <= 0) {
            throw std::runtime_error("LZ4 compression failed.");
        }
        total += static_cast<size_t>(comp);
        offset += len;
    }
    return total;
}

size_t compress_deflate_chunked(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return 0;
    }
    size_t total = 0;
    size_t offset = 0;
    while (offset < data.size()) {
        const size_t len = std::min(kChunkSizeBytes, data.size() - offset);
        uLongf bound = compressBound(static_cast<uLong>(len));
        std::vector<uint8_t> dst(static_cast<size_t>(bound));

        int ret = compress2(
                dst.data(),
                &bound,
                data.data() + offset,
                static_cast<uLong>(len),
                Z_DEFAULT_COMPRESSION);
        if (ret != Z_OK) {
            throw std::runtime_error("Deflate compression failed.");
        }
        total += static_cast<size_t>(bound);
        offset += len;
    }
    return total;
}

size_t index_serialized_size_bytes(
        const faiss::Index* index,
        const std::string& tag) {
    const pid_t pid = getpid();
    fs::path path = fs::temp_directory_path() /
            ("section3_" + tag + "_" + std::to_string(pid) + ".faissindex");

    faiss::write_index(index, path.string().c_str());
    const size_t size = static_cast<size_t>(fs::file_size(path));
    fs::remove(path);
    return size;
}

double to_mb(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double pct_of(size_t value, size_t base) {
    if (base == 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(value) / static_cast<double>(base);
}

DatasetResult profile_one_dataset(
        const std::string& name,
        const std::string& fvecs_path,
        const std::string& jsonl_path,
        const Args& args) {
    DatasetResult out;
    out.name = name;

    FvecData fvecs = read_fvecs(fvecs_path);
    std::vector<JsonRow> rows = read_jsonl_rows(jsonl_path);
    if (rows.empty()) {
        throw std::runtime_error("No rows found in jsonl: " + jsonl_path);
    }

    size_t n = std::min(fvecs.n, rows.size());
    if (args.max_rows > 0) {
        n = std::min(n, args.max_rows);
        std::ostringstream oss;
        oss << "max_rows=" << args.max_rows << " applied.";
        out.notes.push_back(oss.str());
    }
    if (n == 0) {
        throw std::runtime_error("No overlapping rows between vectors and metadata.");
    }
    if (fvecs.n != rows.size()) {
        std::ostringstream oss;
        oss << "Vector count (" << fvecs.n << ") != metadata rows (" << rows.size()
            << "); truncated to " << n << ".";
        out.notes.push_back(oss.str());
    }

    if (n < fvecs.n) {
        fvecs.values.resize(n * fvecs.d);
    }
    out.rows = n;
    out.dim = fvecs.d;
    out.embedding_bytes = n * fvecs.d * sizeof(float);

    MetadataBuildResult metadata = build_metadata_payload(rows, name, n);
    out.selected_keys = metadata.selected_keys;
    out.selected_metadata_columns = metadata.selected_keys.size();
    out.unique_metadata_ids = metadata.unique_metadata_ids;

    out.metadata_original_bytes = metadata.metadata_blob.size();
    out.metadata_lz4_bytes = compress_lz4_chunked(metadata.metadata_blob);
    out.metadata_deflate_bytes = compress_deflate_chunked(metadata.metadata_blob);

    const int acorn_mbeta = args.acorn_mbeta > 0 ? args.acorn_mbeta : args.m * 2;
    {
        faiss::IndexHNSWFlat hnsw(static_cast<int>(out.dim), args.m, 1);
        hnsw.hnsw.efConstruction = args.ef_construct;
        hnsw.add(static_cast<faiss::idx_t>(out.rows), fvecs.values.data());
        out.hnsw_index_bytes = index_serialized_size_bytes(&hnsw, name + "_hnsw");
    }
    out.hnsw_graph_bytes =
            out.hnsw_index_bytes > out.embedding_bytes
            ? out.hnsw_index_bytes - out.embedding_bytes
            : 0;

    {
        faiss::IndexACORNFlat acorn(
                static_cast<int>(out.dim),
                args.m,
                args.acorn_gamma,
                metadata.metadata_ids,
                acorn_mbeta);
        acorn.acorn.efConstruction = args.ef_construct;
        acorn.add(static_cast<faiss::idx_t>(out.rows), fvecs.values.data());
        out.acorn_index_bytes = index_serialized_size_bytes(&acorn, name + "_acorn");
    }
    out.acorn_graph_bytes =
            out.acorn_index_bytes > out.embedding_bytes
            ? out.acorn_index_bytes - out.embedding_bytes
            : 0;

    out.original_plus_hnsw_bytes = out.metadata_original_bytes + out.hnsw_graph_bytes;
    out.lz4_plus_hnsw_bytes = out.metadata_lz4_bytes + out.hnsw_graph_bytes;
    out.deflate_plus_hnsw_bytes = out.metadata_deflate_bytes + out.hnsw_graph_bytes;
    out.acorn_metadata_bytes = out.acorn_graph_bytes;

    return out;
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '\"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

void write_report_txt(
        const std::vector<DatasetResult>& results,
        const Args& args,
        const std::string& out_path) {
    std::ofstream out(out_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write report: " + out_path);
    }

    out << "Section3 Memory Breakdown (Filtering)\n";
    out << "====================================\n";
    out << "HNSW params: M=" << args.m
        << ", efConstruction=" << args.ef_construct << "\n";
    out << "ACORN params: gamma=" << args.acorn_gamma
        << ", M_beta=" << (args.acorn_mbeta > 0 ? args.acorn_mbeta : args.m * 2) << "\n";
    out << "Max rows: " << (args.max_rows > 0 ? std::to_string(args.max_rows) : "all") << "\n";
    out << "Metadata compression: 2MB blocks (LZ4, Deflate)\n\n";

    for (const auto& ds : results) {
        out << "Dataset: " << ds.name << "\n";
        out << "Rows: " << ds.rows << ", Dim: " << ds.dim << "\n";
        out << "Selected metadata columns: " << ds.selected_metadata_columns
            << " (unique row metadata IDs: " << ds.unique_metadata_ids << ")\n";
        out << "Keys: ";
        for (size_t i = 0; i < ds.selected_keys.size(); ++i) {
            out << ds.selected_keys[i];
            if (i + 1 < ds.selected_keys.size()) {
                out << ", ";
            }
        }
        out << "\n";

        out << "Raw sizes (bytes):\n";
        out << "  embedding: " << ds.embedding_bytes << "\n";
        out << "  metadata original: " << ds.metadata_original_bytes << "\n";
        out << "  metadata LZ4(2MB): " << ds.metadata_lz4_bytes << "\n";
        out << "  metadata Deflate(2MB): " << ds.metadata_deflate_bytes << "\n";
        out << "  HNSW index: " << ds.hnsw_index_bytes
            << ", HNSW graph=index-embedding: " << ds.hnsw_graph_bytes << "\n";
        out << "  ACORN index: " << ds.acorn_index_bytes
            << ", ACORN graph=index-embedding: " << ds.acorn_graph_bytes << "\n";

        out << "Plot categories (MB, and % of embedding):\n";
        const std::vector<std::pair<std::string, size_t>> bars = {
                {"Original metadata + HNSW graph", ds.original_plus_hnsw_bytes},
                {"Compressed metadata (LZ4) + HNSW graph", ds.lz4_plus_hnsw_bytes},
                {"Compressed metadata (Deflate) + HNSW graph", ds.deflate_plus_hnsw_bytes},
                {"ACORN metadata (graph size)", ds.acorn_metadata_bytes},
                {"Embedding", ds.embedding_bytes},
        };
        for (const auto& [label, bytes] : bars) {
            out << "  - " << label << ": "
                << std::fixed << std::setprecision(3) << to_mb(bytes) << " MB"
                << " (" << std::setprecision(2) << pct_of(bytes, ds.embedding_bytes)
                << "% of embedding)\n";
        }

        if (!ds.notes.empty()) {
            out << "Notes:\n";
            for (const auto& note : ds.notes) {
                out << "  - " << note << "\n";
            }
        }
        out << "\n";
    }
}

void write_report_json(
        const std::vector<DatasetResult>& results,
        const Args& args,
        const std::string& out_path) {
    std::ofstream out(out_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write json report: " + out_path);
    }

    out << "{\n";
    out << "  \"params\": {\n";
    out << "    \"m\": " << args.m << ",\n";
    out << "    \"ef_construction\": " << args.ef_construct << ",\n";
    out << "    \"acorn_gamma\": " << args.acorn_gamma << ",\n";
    out << "    \"acorn_mbeta\": " << (args.acorn_mbeta > 0 ? args.acorn_mbeta : args.m * 2) << ",\n";
    out << "    \"max_rows\": " << args.max_rows << ",\n";
    out << "    \"metadata_compression_block_bytes\": " << kChunkSizeBytes << "\n";
    out << "  },\n";
    out << "  \"datasets\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const DatasetResult& ds = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(ds.name) << "\",\n";
        out << "      \"rows\": " << ds.rows << ",\n";
        out << "      \"dim\": " << ds.dim << ",\n";
        out << "      \"selected_metadata_columns\": " << ds.selected_metadata_columns << ",\n";
        out << "      \"unique_metadata_ids\": " << ds.unique_metadata_ids << ",\n";
        out << "      \"selected_keys\": [";
        for (size_t k = 0; k < ds.selected_keys.size(); ++k) {
            out << "\"" << json_escape(ds.selected_keys[k]) << "\"";
            if (k + 1 < ds.selected_keys.size()) {
                out << ", ";
            }
        }
        out << "],\n";

        out << "      \"raw_sizes\": {\n";
        out << "        \"embedding_bytes\": " << ds.embedding_bytes << ",\n";
        out << "        \"metadata_original_bytes\": " << ds.metadata_original_bytes << ",\n";
        out << "        \"metadata_lz4_bytes\": " << ds.metadata_lz4_bytes << ",\n";
        out << "        \"metadata_deflate_bytes\": " << ds.metadata_deflate_bytes << ",\n";
        out << "        \"hnsw_index_bytes\": " << ds.hnsw_index_bytes << ",\n";
        out << "        \"hnsw_graph_bytes\": " << ds.hnsw_graph_bytes << ",\n";
        out << "        \"acorn_index_bytes\": " << ds.acorn_index_bytes << ",\n";
        out << "        \"acorn_graph_bytes\": " << ds.acorn_graph_bytes << "\n";
        out << "      },\n";

        out << "      \"categories\": {\n";
        out << "        \"original_metadata\": " << ds.original_plus_hnsw_bytes << ",\n";
        out << "        \"compressed_metadata_lz4\": " << ds.lz4_plus_hnsw_bytes << ",\n";
        out << "        \"compressed_metadata_deflate\": " << ds.deflate_plus_hnsw_bytes << ",\n";
        out << "        \"acorn_metadata\": " << ds.acorn_metadata_bytes << ",\n";
        out << "        \"embedding\": " << ds.embedding_bytes << "\n";
        out << "      },\n";

        out << "      \"bars\": [\n";
        const std::vector<std::pair<std::string, size_t>> bars = {
                {"Original metadata + HNSW graph", ds.original_plus_hnsw_bytes},
                {"Compressed metadata (LZ4) + HNSW graph", ds.lz4_plus_hnsw_bytes},
                {"Compressed metadata (Deflate) + HNSW graph", ds.deflate_plus_hnsw_bytes},
                {"ACORN metadata (graph size)", ds.acorn_metadata_bytes},
                {"Embedding", ds.embedding_bytes},
        };
        for (size_t b = 0; b < bars.size(); ++b) {
            out << "        {"
                << "\"label\": \"" << json_escape(bars[b].first) << "\", "
                << "\"bytes\": " << bars[b].second << ", "
                << "\"mb\": " << std::fixed << std::setprecision(6) << to_mb(bars[b].second) << ", "
                << "\"pct_of_embedding\": " << std::setprecision(6) << pct_of(bars[b].second, ds.embedding_bytes)
                << "}";
            if (b + 1 < bars.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "      ],\n";

        out << "      \"notes\": [";
        for (size_t n = 0; n < ds.notes.size(); ++n) {
            out << "\"" << json_escape(ds.notes[n]) << "\"";
            if (n + 1 < ds.notes.size()) {
                out << ", ";
            }
        }
        out << "]\n";
        out << "    }";
        if (i + 1 < results.size()) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        ensure_file_exists(args.hnm_fvecs);
        ensure_file_exists(args.hnm_jsonl);
        ensure_file_exists(args.laion_fvecs);
        ensure_file_exists(args.laion_jsonl);

        std::vector<DatasetResult> results;
        results.push_back(profile_one_dataset("hnm", args.hnm_fvecs, args.hnm_jsonl, args));
        results.push_back(profile_one_dataset("laion", args.laion_fvecs, args.laion_jsonl, args));

        write_report_txt(results, args, args.out_txt);
        write_report_json(results, args, args.out_json);

        std::cout << "Wrote " << args.out_txt << " and " << args.out_json << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
