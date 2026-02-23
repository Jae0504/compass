#include "../../hnswlib/build_hnsw/hnswlib.h"

#include <bitset>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// Global definitions required by globals.h / hnswalg.h in this fork.
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
bool ena_algorithm = false;
bool ena_cnt_distcal = false;
int nElements = 0;
int nFilters = 256;
int nThreads = 1;
int cnt_distcal_lv0 = 0;
int cnt_distcal_upper = 0;
std::vector<int> target_filter_ids;
std::vector<int> target_filter_ids_m1;
std::vector<int> target_filter_ids_m2;
std::ofstream RunResultFile;
std::string dataset_type;
int isolated_connection_factor = 0;
int steiner_factor = 0;
int ep_factor = 0;
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

struct Args {
    std::string base_path;
    std::string graph_out;
    int m = 32;
    int ef_construct = 128;
};

struct FvecData {
    int num = 0;
    int dim = 0;
    std::vector<float> values;
};

struct BvecData {
    int num = 0;
    int dim = 0;
    std::vector<uint8_t> values;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " --base <path(.fvecs|.bvecs)>"
        << " --M <int>"
        << " --efconstruct <int>"
        << " --graph-out <path_or_dir>\n";
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

        if (cur == "--base") {
            args.base_path = require_value(cur);
        } else if (cur == "--M") {
            args.m = std::stoi(require_value(cur));
        } else if (cur == "--efconstruct") {
            args.ef_construct = std::stoi(require_value(cur));
        } else if (cur == "--graph-out") {
            args.graph_out = require_value(cur);
        } else if (cur == "-h" || cur == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + cur);
        }
    }

    if (args.base_path.empty()) {
        throw std::runtime_error("--base is required");
    }
    if (args.graph_out.empty()) {
        throw std::runtime_error("--graph-out is required");
    }
    if (args.m <= 0) {
        throw std::runtime_error("--M must be > 0");
    }
    if (args.ef_construct <= 0) {
        throw std::runtime_error("--efconstruct must be > 0");
    }
    if (!fs::exists(args.base_path) || !fs::is_regular_file(args.base_path)) {
        throw std::runtime_error("Base file not found: " + args.base_path);
    }

    return args;
}

FvecData read_fvecs_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open fvecs: " + path);
    }

    FvecData out;
    int32_t d = 0;
    while (true) {
        if (!in.read(reinterpret_cast<char*>(&d), sizeof(int32_t))) {
            break;
        }
        if (d <= 0) {
            throw std::runtime_error("Invalid dimension in fvecs: " + path);
        }

        if (out.dim == 0) {
            out.dim = d;
        } else if (out.dim != d) {
            throw std::runtime_error("Inconsistent dimensions in fvecs: " + path);
        }

        const size_t old_size = out.values.size();
        out.values.resize(old_size + static_cast<size_t>(d));

        const size_t bytes = static_cast<size_t>(d) * sizeof(float);
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

BvecData read_bvecs_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open bvecs: " + path);
    }

    BvecData out;
    int32_t d = 0;
    while (true) {
        if (!in.read(reinterpret_cast<char*>(&d), sizeof(int32_t))) {
            break;
        }
        if (d <= 0) {
            throw std::runtime_error("Invalid dimension in bvecs: " + path);
        }

        if (out.dim == 0) {
            out.dim = d;
        } else if (out.dim != d) {
            throw std::runtime_error("Inconsistent dimensions in bvecs: " + path);
        }

        const size_t old_size = out.values.size();
        out.values.resize(old_size + static_cast<size_t>(d));

        const size_t bytes = static_cast<size_t>(d) * sizeof(uint8_t);
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

std::string derive_graph_path(const Args& args) {
    fs::path out_path(args.graph_out);
    if (fs::exists(out_path) && fs::is_directory(out_path)) {
        fs::path base_name = fs::path(args.base_path).stem();
        std::string file_name = base_name.string() + "_m" + std::to_string(args.m) +
                                "_efc" + std::to_string(args.ef_construct) + ".bin";
        return (out_path / file_name).string();
    }

    if (!out_path.parent_path().empty()) {
        fs::create_directories(out_path.parent_path());
    }
    return out_path.string();
}

void build_from_fvecs(const Args& args, const std::string& out_graph) {
    FvecData data = read_fvecs_raw(args.base_path);
    hnswlib::L2Space space(static_cast<size_t>(data.dim));
    hnswlib::HierarchicalNSW<float> index(
        &space,
        static_cast<size_t>(data.num),
        static_cast<size_t>(args.m),
        static_cast<size_t>(args.ef_construct));

    for (int i = 0; i < data.num; ++i) {
        index.addPoint(
            static_cast<const void*>(data.values.data() + static_cast<size_t>(i) * data.dim),
            static_cast<hnswlib::labeltype>(i));
    }

    index.saveIndex(out_graph);

    std::cout << "Built HNSW from fvecs\n";
    std::cout << "  vectors: " << data.num << "\n";
    std::cout << "  dim: " << data.dim << "\n";
    std::cout << "  M: " << args.m << "\n";
    std::cout << "  efConstruct: " << args.ef_construct << "\n";
    std::cout << "  graph: " << out_graph << "\n";
}

void build_from_bvecs(const Args& args, const std::string& out_graph) {
    BvecData data = read_bvecs_raw(args.base_path);
    hnswlib::L2SpaceI space(static_cast<size_t>(data.dim));
    hnswlib::HierarchicalNSW<int> index(
        &space,
        static_cast<size_t>(data.num),
        static_cast<size_t>(args.m),
        static_cast<size_t>(args.ef_construct));

    for (int i = 0; i < data.num; ++i) {
        index.addPoint(
            static_cast<const void*>(data.values.data() + static_cast<size_t>(i) * data.dim),
            static_cast<hnswlib::labeltype>(i));
    }

    index.saveIndex(out_graph);

    std::cout << "Built HNSW from bvecs (uint8_t kept)\n";
    std::cout << "  vectors: " << data.num << "\n";
    std::cout << "  dim: " << data.dim << "\n";
    std::cout << "  M: " << args.m << "\n";
    std::cout << "  efConstruct: " << args.ef_construct << "\n";
    std::cout << "  graph: " << out_graph << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::string out_graph = derive_graph_path(args);

        if (args.base_path.size() >= 6 && args.base_path.substr(args.base_path.size() - 6) == ".fvecs") {
            build_from_fvecs(args, out_graph);
        } else if (args.base_path.size() >= 6 && args.base_path.substr(args.base_path.size() - 6) == ".bvecs") {
            build_from_bvecs(args, out_graph);
        } else {
            throw std::runtime_error("--base must end with .fvecs or .bvecs");
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
