#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <lz4.h>
#include <zlib.h>

const size_t CHUNK_SIZE = 2 * 1024 * 1024; // 2MB chunks
const size_t COUNT_1M = 1 * 1000 * 1000;   // 1 million data elements
const size_t COUNT_20M = 20 * 1000 * 1000; // 20 million data elements

// Country names for testing
const std::vector<std::string> COUNTRIES = {
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas",
    "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize",
    "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil",
    "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras",
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro",
    "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
    "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine",
    "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia",
    "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan",
    "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste",
    "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan",
    "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe", "Abkhazia", "Artsakh", "Cook Islands",
    "Niue", "Northern Cyprus", "Sahrawi Arab Democratic Republic", "Somaliland", "Transnistria", "South Ossetia",
    "Hong Kong", "Macau", "Puerto Rico", "Greenland", "French Polynesia", "New Caledonia",
    "Bermuda", "Guam", "Faroe Islands", "Gibraltar", "Cayman Islands", "American Samoa",
    "U.S. Virgin Islands", "British Virgin Islands", "Turks and Caicos Islands", "Aruba",
    "Curacao", "Sint Maarten", "Guadeloupe", "Martinique", "French Guiana", "Reunion",
    "Mayotte", "Saint Martin", "Saint Barthelemy", "Saint Pierre and Miquelon", "Wallis and Futuna", "Tokelau",
    "Norfolk Island", "Christmas Island", "Cocos Islands", "Pitcairn Islands", "Falkland Islands", "South Georgia",
    "Bouvet Island", "Heard Island", "Svalbard", "Jan Mayen", "Ascension Island", "Tristan da Cunha",
    "Saint Helena", "Ceuta", "Melilla", "Akrotiri and Dhekelia", "Isle of Man", "Jersey",
    "Guernsey", "Aland Islands", "Scotland", "Wales", "Northern Ireland", "England",
    "Catalonia", "Basque Country", "Galicia", "Bavaria", "Saxony", "Prussia",
    "Brittany", "Corsica", "Sardinia", "Sicily", "Venice", "Tuscany"
};

struct CompressionResult {
    std::string name;
    size_t original_size;
    size_t compressed_size;
    double ratio;
};

void print_result(const CompressionResult& result) {
    std::cout << std::left << std::setw(40) << result.name
              << " Original: " << std::setw(12) << result.original_size
              << " Compressed: " << std::setw(12) << result.compressed_size
              << " Ratio: " << std::fixed << std::setprecision(4) << result.ratio << std::endl;
}

// Compress data using LZ4 with 2MB chunking
CompressionResult compress_lz4(const std::vector<uint8_t>& data, const std::string& name) {
    size_t total_compressed = 0;
    size_t offset = 0;

    while (offset < data.size()) {
        size_t chunk_len = std::min(CHUNK_SIZE, data.size() - offset);

        int max_compressed_size = LZ4_compressBound(chunk_len);
        std::vector<char> compressed(max_compressed_size);

        int compressed_size = LZ4_compress_default(
            reinterpret_cast<const char*>(data.data() + offset),
            compressed.data(),
            chunk_len,
            max_compressed_size
        );

        if (compressed_size <= 0) {
            std::cerr << "LZ4 compression failed!" << std::endl;
            return {name + " (LZ4)", data.size(), 0, 0.0};
        }

        total_compressed += compressed_size;
        offset += chunk_len;
    }

    return {
        name + " (LZ4)",
        data.size(),
        total_compressed,
        static_cast<double>(total_compressed) / data.size()
    };
}

// Compress data using Deflate with 2MB chunking
CompressionResult compress_deflate(const std::vector<uint8_t>& data, const std::string& name) {
    size_t total_compressed = 0;
    size_t offset = 0;

    while (offset < data.size()) {
        size_t chunk_len = std::min(CHUNK_SIZE, data.size() - offset);

        uLongf compressed_size = compressBound(chunk_len);
        std::vector<uint8_t> compressed(compressed_size);

        int result = compress2(
            compressed.data(),
            &compressed_size,
            data.data() + offset,
            chunk_len,
            Z_DEFAULT_COMPRESSION
        );

        if (result != Z_OK) {
            std::cerr << "Deflate compression failed!" << std::endl;
            return {name + " (Deflate)", data.size(), 0, 0.0};
        }

        total_compressed += compressed_size;
        offset += chunk_len;
    }

    return {
        name + " (Deflate)",
        data.size(),
        total_compressed,
        static_cast<double>(total_compressed) / data.size()
    };
}

// Generate random int data (0-255)
std::vector<int> generate_random_ints(size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<int> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = dis(gen);
    }
    return data;
}

// Convert int vector to uint8_t vector
std::vector<uint8_t> int_to_binary(const std::vector<int>& data) {
    std::vector<uint8_t> binary(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        binary[i] = static_cast<uint8_t>(data[i]);
    }
    return binary;
}

// Convert int vector to bytes (as-is representation)
std::vector<uint8_t> int_to_bytes(const std::vector<int>& data) {
    std::vector<uint8_t> bytes(data.size() * sizeof(int));
    std::memcpy(bytes.data(), data.data(), data.size() * sizeof(int));
    return bytes;
}

// Generate random country names data (text format) - count is number of country names
std::vector<uint8_t> generate_country_text_data(size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, COUNTRIES.size() - 1);

    std::vector<uint8_t> data;
    // Estimate size: average country name ~10 chars + newline
    data.reserve(count * 11);

    for (size_t i = 0; i < count; i++) {
        const std::string& country = COUNTRIES[dis(gen)];
        data.insert(data.end(), country.begin(), country.end());
        data.push_back('\n'); // Add newline separator
    }

    return data;
}

// Generate random country indices as uint8_t (binary format) - count is number of indices
std::vector<uint8_t> generate_country_indices_data(size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<uint8_t> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<uint8_t>(dis(gen));
    }
    return data;
}

void test_random_int_data(size_t count, const std::string& count_label) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing Random Int Data (" << count_label << " elements)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Generate random int data (0-255)
    std::vector<int> int_data = generate_random_ints(count);

    // Test 1: Compress int data as-is (4 bytes per int)
    std::cout << "\n--- Int data (as 4-byte integers) ---" << std::endl;
    std::vector<uint8_t> int_bytes = int_to_bytes(int_data);
    std::cout << "Data size: " << int_bytes.size() << " bytes ("
              << (int_bytes.size() / (1024.0 * 1024.0)) << " MB)" << std::endl;
    CompressionResult r1 = compress_lz4(int_bytes, "Random Int");
    CompressionResult r2 = compress_deflate(int_bytes, "Random Int");
    print_result(r1);
    print_result(r2);

    // Test 2: Compress binary data (uint8_t representation)
    std::cout << "\n--- Binary data (as uint8_t, 1 byte per value) ---" << std::endl;
    std::vector<uint8_t> binary_data = int_to_binary(int_data);
    std::cout << "Data size: " << binary_data.size() << " bytes ("
              << (binary_data.size() / (1024.0 * 1024.0)) << " MB)" << std::endl;
    CompressionResult r3 = compress_lz4(binary_data, "Binary (uint8_t)");
    CompressionResult r4 = compress_deflate(binary_data, "Binary (uint8_t)");
    print_result(r3);
    print_result(r4);
}

// Simple JSON value extractor (for JSONL with simple key-value pairs)
std::map<std::string, std::string> parse_json_line(const std::string& line) {
    std::map<std::string, std::string> result;
    size_t pos = 0;

    while (pos < line.size()) {
        // Find key
        size_t key_start = line.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = line.find('"', key_start + 1);
        if (key_end == std::string::npos) break;

        std::string key = line.substr(key_start + 1, key_end - key_start - 1);

        // Find colon
        size_t colon = line.find(':', key_end);
        if (colon == std::string::npos) break;

        // Find value
        size_t value_start = colon + 1;
        while (value_start < line.size() && (line[value_start] == ' ' || line[value_start] == '\t')) {
            value_start++;
        }

        std::string value;
        if (line[value_start] == '"') {
            // String value
            size_t value_end = line.find('"', value_start + 1);
            if (value_end == std::string::npos) break;
            value = line.substr(value_start + 1, value_end - value_start - 1);
            pos = value_end + 1;
        } else {
            // Numeric value
            size_t value_end = line.find_first_of(",}", value_start);
            if (value_end == std::string::npos) break;
            value = line.substr(value_start, value_end - value_start);
            pos = value_end;
        }

        result[key] = value;
        pos++;
    }

    return result;
}

// Column data structure
struct ColumnData {
    std::string name;
    std::vector<std::string> values;
    bool is_numeric;
    long long min_val;
    long long max_val;
    size_t unique_count;
};

// Analyze column and determine conversion strategy
ColumnData analyze_column(const std::string& col_name, const std::vector<std::string>& values) {
    ColumnData col;
    col.name = col_name;
    col.values = values;
    col.is_numeric = true;
    col.min_val = LLONG_MAX;
    col.max_val = LLONG_MIN;

    std::set<std::string> unique_values;

    for (const auto& val : values) {
        unique_values.insert(val);

        // Try to parse as number
        if (col.is_numeric) {
            try {
                size_t pos;
                long long num = std::stoll(val, &pos);
                if (pos == val.length()) {
                    col.min_val = std::min(col.min_val, num);
                    col.max_val = std::max(col.max_val, num);
                } else {
                    col.is_numeric = false;
                }
            } catch (...) {
                col.is_numeric = false;
            }
        }
    }

    col.unique_count = unique_values.size();
    return col;
}

// Convert column to binary format
std::vector<uint8_t> convert_column_to_binary(const ColumnData& col) {
    std::vector<uint8_t> binary(col.values.size());

    if (col.unique_count <= 256) {
        // Map unique values to indices
        std::map<std::string, uint8_t> value_map;
        std::set<std::string> unique_vals(col.values.begin(), col.values.end());
        uint8_t idx = 0;
        for (const auto& val : unique_vals) {
            value_map[val] = idx++;
        }

        for (size_t i = 0; i < col.values.size(); i++) {
            binary[i] = value_map[col.values[i]];
        }
    } else if (col.is_numeric) {
        long long range = col.max_val - col.min_val;

        if (range < 256) {
            // Direct offset encoding
            for (size_t i = 0; i < col.values.size(); i++) {
                long long num = std::stoll(col.values[i]);
                binary[i] = static_cast<uint8_t>(num - col.min_val);
            }
        } else {
            // Bucketing with step
            double step = static_cast<double>(range) / 256.0;
            for (size_t i = 0; i < col.values.size(); i++) {
                long long num = std::stoll(col.values[i]);
                uint8_t bucket = static_cast<uint8_t>(std::min(255.0, (num - col.min_val) / step));
                binary[i] = bucket;
            }
        }
    } else {
        // Fallback: hash to uint8_t
        for (size_t i = 0; i < col.values.size(); i++) {
            std::hash<std::string> hasher;
            binary[i] = static_cast<uint8_t>(hasher(col.values[i]) % 256);
        }
    }

    return binary;
}

// Convert column values to byte stream (text format)
std::vector<uint8_t> column_to_text_bytes(const std::vector<std::string>& values) {
    std::vector<uint8_t> result;
    for (const auto& val : values) {
        result.insert(result.end(), val.begin(), val.end());
        result.push_back('\n');
    }
    return result;
}

void test_jsonl_columns(const std::string& filename) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing JSONL Column Compression" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "========================================" << std::endl;

    // Read JSONL file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Parse all rows and extract columns
    std::map<std::string, std::vector<std::string>> columns;
    std::string line;
    size_t row_count = 0;

    std::cout << "Reading and parsing JSONL file..." << std::endl;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto row = parse_json_line(line);
        for (const auto& [key, value] : row) {
            columns[key].push_back(value);
        }
        row_count++;

        if (row_count % 10000 == 0) {
            std::cout << "  Processed " << row_count << " rows..." << std::endl;
        }
    }
    file.close();

    std::cout << "Total rows: " << row_count << std::endl;
    std::cout << "Total columns: " << columns.size() << std::endl;
    std::cout << "\n" << std::endl;

    // Process each column
    for (const auto& [col_name, col_values] : columns) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Column: " << col_name << std::endl;

        // Analyze column
        ColumnData col_data = analyze_column(col_name, col_values);

        std::cout << "  Rows: " << col_values.size()
                  << " | Unique: " << col_data.unique_count
                  << " | Type: " << (col_data.is_numeric ? "Numeric" : "String") << std::endl;

        if (col_data.is_numeric) {
            std::cout << "  Range: [" << col_data.min_val << " - " << col_data.max_val
                      << "] (diff: " << (col_data.max_val - col_data.min_val) << ")" << std::endl;
        }

        // Convert to text bytes
        std::vector<uint8_t> text_bytes = column_to_text_bytes(col_values);
        std::cout << "  Text size: " << text_bytes.size() << " bytes ("
                  << (text_bytes.size() / (1024.0 * 1024.0)) << " MB)" << std::endl;

        // Compress text format
        CompressionResult text_lz4 = compress_lz4(text_bytes, col_name + " (Text)");
        CompressionResult text_deflate = compress_deflate(text_bytes, col_name + " (Text)");

        std::cout << "  Text compression:" << std::endl;
        std::cout << "    ";
        print_result(text_lz4);
        std::cout << "    ";
        print_result(text_deflate);

        // Convert to binary and compress
        std::vector<uint8_t> binary_bytes = convert_column_to_binary(col_data);
        std::cout << "  Binary size: " << binary_bytes.size() << " bytes ("
                  << (binary_bytes.size() / (1024.0 * 1024.0)) << " MB)" << std::endl;

        CompressionResult binary_lz4 = compress_lz4(binary_bytes, col_name + " (Binary)");
        CompressionResult binary_deflate = compress_deflate(binary_bytes, col_name + " (Binary)");

        std::cout << "  Binary compression:" << std::endl;
        std::cout << "    ";
        print_result(binary_lz4);
        std::cout << "    ";
        print_result(binary_deflate);

        // Show improvement
        double size_reduction = (1.0 - (double)binary_bytes.size() / text_bytes.size()) * 100.0;
        std::cout << "  >>> Text to binary size reduction: "
                  << std::fixed << std::setprecision(2) << size_reduction << "%" << std::endl;

        std::cout << std::endl;
    }
}

void test_country_data(size_t count, const std::string& count_label) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing Country Names Data (" << count_label << " elements)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test 1: Country names as text (variable length strings)
    std::cout << "\n--- Country names (text format) ---" << std::endl;
    std::vector<uint8_t> country_text = generate_country_text_data(count);
    std::cout << "Data size: " << country_text.size() << " bytes ("
              << (country_text.size() / (1024.0 * 1024.0)) << " MB)" << std::endl;
    CompressionResult r1 = compress_lz4(country_text, "Country Names Text");
    CompressionResult r2 = compress_deflate(country_text, "Country Names Text");
    print_result(r1);
    print_result(r2);

    // Test 2: Country indices as uint8_t (binary format: Afghanistan=0, Albania=1, etc.)
    std::cout << "\n--- Country indices (uint8_t binary, 0-255) ---" << std::endl;
    std::vector<uint8_t> country_indices = generate_country_indices_data(count);
    std::cout << "Data size: " << country_indices.size() << " bytes ("
              << (country_indices.size() / (1024.0 * 1024.0)) << " MB)" << std::endl;
    CompressionResult r3 = compress_lz4(country_indices, "Country Indices Binary");
    CompressionResult r4 = compress_deflate(country_indices, "Country Indices Binary");
    print_result(r3);
    print_result(r4);

    // Show size reduction from text to binary
    double size_reduction = (1.0 - (double)country_indices.size() / country_text.size()) * 100.0;
    std::cout << "\n>>> Converting text to binary reduced size by "
              << std::fixed << std::setprecision(2) << size_reduction << "%" << std::endl;
}

int main() {
    std::cout << "Compression Ratio Evaluation" << std::endl;
    std::cout << "Chunk Size: 2MB" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test random int data - 1M elements
    test_random_int_data(COUNT_1M, "1M");

    // Test random int data - 20M elements
    test_random_int_data(COUNT_20M, "20M");

    // Test country names data - 1M elements
    test_country_data(COUNT_1M, "1M");

    // Test country names data - 20M elements
    test_country_data(COUNT_20M, "20M");

    // Test JSONL column compression
    test_jsonl_columns("/home/jykang5/profile/compression_motivation/payloads.jsonl");

    std::cout << "\n========================================" << std::endl;
    std::cout << "Compression ratio = compressed_size / original_size" << std::endl;
    std::cout << "Lower ratio = better compression" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
