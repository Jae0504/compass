#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <string>
#include <bitset>

#include <cstddef> // for size_t
#include <fstream>
#include <stdexcept>
#include <vector>
#include <array>

#include <unordered_map>
#include <sstream>
#include <filesystem>

#include <sstream>
#include <regex>

// Function to read SIFT base and query data
inline float* read_fvecs(const std::string& filename, int& num_vectors, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    int dim_int;
    file.read((char*)&dim_int, sizeof(int));
    dim = dim_int;

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();  // 여긴 여전히 size_t가 필요
    file.seekg(0, std::ios::beg);

    num_vectors = static_cast<int>(fileSize / (dim * sizeof(float) + sizeof(int)));

    float* data = new float[num_vectors * dim];
    for (int i = 0; i < num_vectors; i++) {
        file.read((char*)&dim_int, sizeof(int));
        file.read((char*)(data + i * dim), dim * sizeof(float));
    }
    return data;
}

inline float* read_bvecs(const std::string& filename, int& num_vectors, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    int dim_int;
    file.read(reinterpret_cast<char*>(&dim_int), sizeof(int));
    dim = dim_int;

    // Determine number of vectors
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    num_vectors = static_cast<int>(fileSize / (dim + sizeof(int)));

    float* data = new float[num_vectors * dim];

    for (int i = 0; i < num_vectors; ++i) {
        file.read(reinterpret_cast<char*>(&dim_int), sizeof(int));  // discard
        std::vector<uint8_t> buffer(dim);
        file.read(reinterpret_cast<char*>(buffer.data()), dim);

        for (int j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<float>(buffer[j]);
        }
    }

    return data;
}

inline int* read_ivecs(const std::string& filename, int& num_vectors, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    int dim_int;
    file.read((char*)&dim_int, sizeof(int));
    dim = dim_int;

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    num_vectors = static_cast<int>(fileSize / (dim * sizeof(int) + sizeof(int)));

    int* data = new int[num_vectors * dim];
    for (int i = 0; i < num_vectors; i++) {
        file.read((char*)&dim_int, sizeof(int));
        file.read((char*)(data + i * dim), dim * sizeof(int));
    }
    return data;
}

template <typename T>
inline void save_ivecs(const std::string& filename, const std::vector<std::vector<T>>& data) {
    /*Generate the directory if it does not exist*/
    std::filesystem::path filePath(filename);
    std::filesystem::path dir = filePath.parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        // throws on failure
        std::filesystem::create_directories(dir);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing");
    }

    for (const auto& vec : data) {
        int dim_int = static_cast<int>(vec.size());  // Write dimension per vector
        file.write(reinterpret_cast<const char*>(&dim_int), sizeof(int));
        for (const auto& val : vec) {
            int value = static_cast<int>(val);  // Cast each element to int
            file.write(reinterpret_cast<const char*>(&value), sizeof(int));
        }
    }

    file.close();
}

void save_fvecs(const std::vector<float>& vectors, int dim, const std::string& filename) {
    /*Generate the directory if it does not exist*/
    std::filesystem::path filePath(filename);
    std::filesystem::path dir = filePath.parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        // throws on failure
        std::filesystem::create_directories(dir);
    }

    std::ofstream out(filename, std::ios::binary);
    for (size_t i = 0; i < vectors.size(); i += dim) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(&vectors[i]), dim * sizeof(float));
    }
}


inline void saveIDs_uint8(const std::vector<uint8_t>& ids, const std::string& save_path, int num_groups) {

    std::filesystem::path outPath = save_path;
    std::filesystem::path parent  = outPath.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        if (!std::filesystem::create_directories(parent, ec) && ec) {
            std::cerr << "Error: could not create directory "
                      << parent << ": " << ec.message() << "\n";
            return;
        }
    }

    std::ofstream outFile(save_path, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file for writing group IDs to: " << save_path << std::endl;
        return;
    }

    size_t size = ids.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    outFile.write(reinterpret_cast<const char*>(ids.data()), size * sizeof(uint8_t));

    outFile.close();
}

inline std::vector<uint8_t> loadIDs_uint8(const std::string& load_path, int num_groups) {
    std::ifstream inFile(load_path, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Unable to open file for reading group IDs from: " << load_path << std::endl;
        return {};
    }

    size_t size = 0;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    std::vector<uint8_t> ids(size);
    inFile.read(reinterpret_cast<char*>(ids.data()), size * sizeof(uint8_t));

    inFile.close();
    return ids;
}

inline void saveBitsets256(const std::vector<std::bitset<256>>& bits, const std::string& path) {

    std::filesystem::path outPath = path;
    std::filesystem::path parent  = outPath.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        if (!std::filesystem::create_directories(parent, ec) && ec) {
            std::cerr << "Error: could not create directory "
                      << parent << ": " << ec.message() << "\n";
            return;
        }
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot open file to save bitsets: " << path << std::endl;
        return;
    }

    size_t size = bits.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& bit : bits) {
        // 비트셋을 32바이트 (= 256비트)로 저장
        std::array<unsigned char, 32> buffer{};
        for (int i = 0; i < 256; ++i) {
            if (bit[i]) {
                buffer[i / 8] |= (1 << (i % 8));
            }
        }
        out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }

    out.close();
}

inline std::vector<std::bitset<256>> loadBitsets256(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file to load bitsets: " << path << std::endl;
        return {};
    }

    size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    std::vector<std::bitset<256>> bits(size);

    for (size_t i = 0; i < size; ++i) {
        std::array<unsigned char, 32> buffer{};
        in.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        std::bitset<256> b;
        for (int j = 0; j < 256; ++j) {
            if (buffer[j / 8] & (1 << (j % 8))) {
                b.set(j);
            }
        }

        bits[i] = b;
    }

    in.close();
    return bits;
}

inline std::unordered_map<int, std::vector<int>> load_ep_extra_map(const std::string& path) {
    std::unordered_map<int, std::vector<int>> ep_map;

    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open file to load EP extra map: " << path << std::endl;
        return ep_map;
    }

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        int ep_id;
        iss >> ep_id;

        int id;
        std::vector<int> extra_ids;
        while (iss >> id) {
            extra_ids.push_back(id);
        }

        ep_map[ep_id] = std::move(extra_ids);
    }

    in.close();
    return ep_map;
}

static std::string to_string_general(float x) {
    std::ostringstream ss;
    ss << std::defaultfloat   // “%g”-style: no trailing zeroes
       << x;
    return ss.str();
}

int countBits(std::uint8_t n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

static std::vector<std::string> parse_csv_fields(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool inq = false;

    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (inq) {
            if (c == '"') {
                if (i + 1 < s.size() && s[i + 1] == '"') { // escaped quote
                    cur.push_back('"'); ++i;
                } else {
                    inq = false; // closing quote
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == '"') {
                inq = true;
            } else if (c == ',') {
                out.emplace_back(std::move(cur));
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
    }
    out.emplace_back(std::move(cur));
    return out;
}

static bool read_csv_record(std::istream& in, std::string& rec) {
    rec.clear();
    std::string line;
    bool inq = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // CRLF
        if (!rec.empty()) rec.push_back('\n');
        rec += line;

        // Toggle in/out of quotes, respecting doubled quotes
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { ++i; } // skip escaped quote
                else { inq = !inq; }
            }
        }
        if (!inq) return true; // complete record
    }
    return !rec.empty(); // last partial record (no newline at EOF)
}

std::vector<std::string> read_csv_column(const std::string& path, const std::string& col_name) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);

    // header (as a full record)
    std::string rec;
    if (!read_csv_record(in, rec)) throw std::runtime_error("Empty CSV: " + path);
    // strip BOM if present
    if (rec.rfind("\xEF\xBB\xBF", 0) == 0) rec.erase(0, 3);

    auto header = parse_csv_fields(rec);

    // find column index
    int col_idx = -1;
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == col_name) { col_idx = static_cast<int>(i); break; }
    }
    if (col_idx == -1) throw std::runtime_error("Column not found: " + col_name);

    std::vector<std::string> values;
    size_t rownum = 1; // data rows start at 1 after header
    while (read_csv_record(in, rec)) {
        ++rownum;
        auto fields = parse_csv_fields(rec);
        if (fields.size() != header.size()) {
            std::cerr << "[WARN] Skipping malformed row " << rownum
                      << " (got " << fields.size() << " fields, expected "
                      << header.size() << ")\n";
            continue; // skip bad row
        }
        values.emplace_back(std::move(fields[col_idx]));
    }
    return values;
}

// Build stable IDs from an unordered_set by sorting the values
static void build_id_maps(
    const std::unordered_set<std::string>& values,
    std::vector<std::string>& id_to_value,
    std::unordered_map<std::string,int>& value_to_id)
{
    id_to_value.assign(values.begin(), values.end());
    std::sort(id_to_value.begin(), id_to_value.end());
    value_to_id.clear();
    for (int i = 0; i < static_cast<int>(id_to_value.size()); ++i) {
        value_to_id.emplace(id_to_value[i], i);
    }
}

// Parse date & chat from a question string
static bool parse_question_bits(const std::string& q, std::string& out_date, std::string& out_chat)
{
    static const std::regex DATE_RE(
        R"(posted\s+on\s+(\d{4}-\d{2}-\d{2}))",
        std::regex::icase
    );
    
    static const std::regex CHAT_RE(
        R"___(on\s+the\s+"(.*?)"\s+group\s+chat\b)___",
        std::regex_constants::icase
    );

    std::smatch m;
    bool ok = true;

    out_date.clear();
    out_chat.clear();

    if (std::regex_search(q, m, DATE_RE)) out_date = m[1].str(); else ok = false;
    if (std::regex_search(q, m, CHAT_RE)) out_chat = m[1].str();
    else ok = false;

    return ok;
}

/**
 * Build query_date_chat_name_ids:
 *   ids[i][0] = date ID for questions[i]
 *   ids[i][1] = chat ID for questions[i]
 * Missing/invalid → -1 and prints the question.
 */
std::vector<std::vector<int>> build_query_ids(
    const std::vector<std::string>& questions,
    const std::unordered_set<std::string>& date_set,
    const std::unordered_set<std::string>& chat_set)
{
    // 2) Output (N x 2), default -1
    std::vector<std::vector<int>> ids(questions.size(), std::vector<int>(2, -1));
    

    // 3) Parse & map
    for (size_t i = 0; i < questions.size(); ++i) {
        const std::string& q = questions[i];
        std::string date, chat;
        bool parsed = parse_question_bits(q, date, chat);

        auto date_it = std::find(date_set.begin(), date_set.end(), date);
        int date_idx = std::distance(date_set.begin(), date_it);
        if (date_idx >= 256) {
            std::cerr << "Error: date_idx exceeds 255, cannot fit in uint8_t" << std::endl;
            return ids;
        }

        auto chat_name_it = std::find(chat_set.begin(), chat_set.end(), chat);
        int chat_name_idx = std::distance(chat_set.begin(), chat_name_it);
        if (chat_name_idx >= 256) {
            std::cerr << "Error: chat_name_idx exceeds 255, cannot fit in uint8_t" << std::endl;
            return ids;
        }

        ids[i][0] = date_idx;
        ids[i][1] = chat_name_idx;
    }

    return ids;
}