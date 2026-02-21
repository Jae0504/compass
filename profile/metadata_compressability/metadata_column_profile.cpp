#include <algorithm>
#include <cctype>
#include <cstdint>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <zlib.h>

struct ColumnStats {
    std::string name;
    bool is_integer = true;
    long long min_val = LLONG_MAX;
    long long max_val = LLONG_MIN;
    size_t unique_count = 0;
    bool included = false;
    std::string reason;
};

static std::map<std::string, std::string> parse_json_line(const std::string& line) {
    std::map<std::string, std::string> result;
    size_t pos = 0;
    while (pos < line.size()) {
        size_t key_start = line.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = line.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = line.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = line.find(':', key_end);
        if (colon == std::string::npos) break;

        size_t value_start = colon + 1;
        while (value_start < line.size() && std::isspace(static_cast<unsigned char>(line[value_start]))) {
            value_start++;
        }

        std::string value;
        if (value_start < line.size() && line[value_start] == '"') {
            size_t value_end = line.find('"', value_start + 1);
            if (value_end == std::string::npos) break;
            value = line.substr(value_start + 1, value_end - value_start - 1);
            pos = value_end + 1;
        } else {
            size_t value_end = line.find_first_of(",}", value_start);
            if (value_end == std::string::npos) break;
            value = line.substr(value_start, value_end - value_start);
            pos = value_end;
        }

        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
        result[key] = value;
        pos++;
    }
    return result;
}

static size_t compressed_size(const std::vector<uint8_t>& data) {
    if (data.empty()) return 0;
    uLongf bound = compressBound(data.size());
    std::vector<uint8_t> out(bound);
    int ret = compress2(out.data(), &bound, data.data(), data.size(), Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK) return 0;
    return static_cast<size_t>(bound);
}

static bool parse_int64(const std::string& s, long long& out) {
    if (s.empty()) return false;
    try {
        size_t pos = 0;
        long long v = std::stoll(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset:hnm|laion> <payloads.jsonl>\n";
        return 1;
    }

    const std::string dataset = argv[1];
    const std::string path = argv[2];

    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }

    std::vector<std::map<std::string, std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) rows.push_back(parse_json_line(line));
    }

    std::vector<std::string> keys;
    if (dataset == "laion") {
        keys = {"NSFW", "similarity", "original_width", "original_height"};
    } else {
        std::set<std::string> keyset;
        for (const auto& r : rows) {
            for (const auto& kv : r) keyset.insert(kv.first);
        }
        for (const auto& k : keyset) {
            if (k != "detail_desc") keys.push_back(k);
        }
    }

    std::vector<uint8_t> raw_text;
    std::vector<uint8_t> binary;
    std::vector<ColumnStats> stats;

    for (const auto& k : keys) {
        std::vector<std::string> vals;
        vals.reserve(rows.size());
        for (const auto& r : rows) {
            auto it = r.find(k);
            vals.push_back(it == r.end() ? "" : it->second);
        }

        ColumnStats s;
        s.name = k;
        std::set<std::string> uniq(vals.begin(), vals.end());
        s.unique_count = uniq.size();

        std::vector<long long> ints;
        ints.reserve(vals.size());
        for (const auto& v : vals) {
            long long iv = 0;
            if (!parse_int64(v, iv)) {
                s.is_integer = false;
                break;
            }
            ints.push_back(iv);
            s.min_val = std::min(s.min_val, iv);
            s.max_val = std::max(s.max_val, iv);
        }

        if (s.is_integer) {
            if (s.unique_count <= 256) {
                s.included = true;
                s.reason = "int unique<=256 mapped";
                std::vector<long long> uniq_ints;
                uniq_ints.reserve(uniq.size());
                for (const auto& u : uniq) {
                    long long iv = 0;
                    parse_int64(u, iv);
                    uniq_ints.push_back(iv);
                }
                std::sort(uniq_ints.begin(), uniq_ints.end());
                std::map<long long, uint8_t> mapping;
                for (size_t i = 0; i < uniq_ints.size(); i++) mapping[uniq_ints[i]] = static_cast<uint8_t>(i);
                for (long long iv : ints) binary.push_back(mapping[iv]);
            } else {
                s.included = true;
                s.reason = "int unique>256 range->0..255";
                long long range = s.max_val - s.min_val;
                for (long long iv : ints) {
                    uint8_t code = 0;
                    if (range > 0) {
                        double v = (iv - s.min_val) * 255.0 / range;
                        if (v < 0) v = 0;
                        if (v > 255) v = 255;
                        code = static_cast<uint8_t>(v);
                    }
                    binary.push_back(code);
                }
            }
        } else if (s.unique_count <= 256) {
            s.included = true;
            s.reason = "string unique<=256 mapped";
            std::map<std::string, uint8_t> mapping;
            uint8_t idx = 0;
            for (const auto& u : uniq) mapping[u] = idx++;
            for (const auto& v : vals) binary.push_back(mapping[v]);
        } else {
            s.included = false;
            s.reason = "string unique>256";
        }

        if (s.included) {
            for (const auto& v : vals) {
                raw_text.insert(raw_text.end(), v.begin(), v.end());
                raw_text.push_back('\n');
            }
        }

        stats.push_back(s);
    }

    size_t included = 0;
    for (const auto& s : stats) {
        if (s.included) included++;
    }

    std::cout << "Dataset: " << dataset << "\n";
    std::cout << "Rows: " << rows.size() << "\n";
    std::cout << "Included columns: " << included << "/" << stats.size() << "\n";
    std::cout << "Metadata raw bytes: " << raw_text.size() << "\n";
    std::cout << "Metadata binary bytes: " << binary.size() << "\n";
    std::cout << "Metadata compressed(raw) bytes: " << compressed_size(raw_text) << "\n";
    std::cout << "Metadata compressed(after binary) bytes: " << compressed_size(binary) << "\n";

    for (const auto& s : stats) {
        std::cout << "  " << s.name << " | unique=" << s.unique_count << " | "
                  << (s.is_integer ? "int" : "string/other") << " | included="
                  << (s.included ? "true" : "false") << " | " << s.reason << "\n";
    }

    return 0;
}
