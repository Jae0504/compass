#include "../../end_to_end/vectordb/hnswlib/build_hnsw/hnswlib.h"
#include "../../end_to_end/vectordb/hnswlib/build_hnsw/io_utils.h"

#include <lz4.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

// Global definitions required by build_hnsw/globals.h referenced from hnswalg.h.
std::vector<std::bitset<256>> connector_bits;
std::vector<std::uint8_t> filter_ids;
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

using Clock = std::chrono::steady_clock;
using Ns = std::chrono::nanoseconds;

constexpr size_t kNumBuckets = 256;
constexpr size_t kBlockSize = 4096;

struct Args {
    std::string graph_path;
    std::string query_path;
    size_t k = 0;
    size_t num_queries = 0;
};

template <typename T>
struct DenseVectors {
    int num = 0;
    int dim = 0;
    std::vector<T> values;
};

using DenseFvecs = DenseVectors<float>;
using DenseBvecs = DenseVectors<uint8_t>;

struct CompressedMetadata {
    std::vector<size_t> raw_block_sizes;
    std::vector<std::vector<char>> lz4_blocks;
    std::vector<std::vector<uint8_t>> deflate_blocks;
};

struct QueryTiming {
    uint64_t lz4_naive_ns = 0;
    uint64_t lz4_common_ns = 0;
    uint64_t deflate_naive_ns = 0;
    uint64_t deflate_common_ns = 0;

    uint64_t upper_traversal_ns = 0;
    uint64_t upper_distance_ns = 0;
    uint64_t level0_traversal_ns = 0;
    uint64_t level0_distance_ns = 0;
    uint64_t candidate_update_ns = 0;

    size_t expanded_nodes_upper = 0;
    size_t expanded_nodes_level0 = 0;
};

struct AggregateTiming {
    QueryTiming total;
    size_t queries = 0;
    size_t num_elements = 0;
    size_t query_dim = 0;
    size_t graph_dim = 0;
    std::string query_format;
    size_t k = 0;
    size_t ef_search = 0;
    size_t block_size = kBlockSize;
};

void usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " <graph(.bin|.index)> <query(.fvecs|.bvecs)> <k> <num_queries>\n";
}

bool parse_size_t(const std::string& s, size_t* out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<size_t>(v);
    return true;
}

Args parse_args(int argc, char** argv) {
    if (argc != 5) {
        usage(argv[0]);
        throw std::runtime_error("Expected exactly 4 positional arguments");
    }

    Args args;
    args.graph_path = argv[1];
    args.query_path = argv[2];

    if (!parse_size_t(argv[3], &args.k) || args.k == 0) {
        throw std::runtime_error("k must be a positive integer");
    }
    if (!parse_size_t(argv[4], &args.num_queries) || args.num_queries == 0) {
        throw std::runtime_error("num_queries must be a positive integer");
    }

    if (!fs::exists(args.graph_path) || !fs::is_regular_file(args.graph_path)) {
        throw std::runtime_error("Graph file does not exist: " + args.graph_path);
    }
    if (!fs::exists(args.query_path) || !fs::is_regular_file(args.query_path)) {
        throw std::runtime_error("Query file does not exist: " + args.query_path);
    }

    return args;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

template <typename DistT, typename SpaceT>
std::unique_ptr<hnswlib::HierarchicalNSW<DistT>> load_graph_index_or_throw(
        SpaceT* space,
        const std::string& graph_path,
        const std::string& query_format) {
    try {
        return std::make_unique<hnswlib::HierarchicalNSW<DistT>>(space, graph_path);
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Failed to load graph index: " << graph_path
            << ". Expected an HNSWlib serialized graph file (commonly .bin or .index) "
            << "compatible with " << query_format << " queries. "
            << "Underlying error: " << e.what();
        throw std::runtime_error(oss.str());
    }
}

DenseFvecs load_queries_fvecs(const std::string& path) {
    int num = 0;
    int dim = 0;
    float* raw = read_fvecs(path, num, dim);
    if (num <= 0 || dim <= 0 || raw == nullptr) {
        delete[] raw;
        throw std::runtime_error("Failed to load query fvecs: " + path);
    }

    DenseFvecs out;
    out.num = num;
    out.dim = dim;
    out.values.assign(raw, raw + static_cast<size_t>(num) * static_cast<size_t>(dim));
    delete[] raw;
    return out;
}

DenseBvecs load_queries_bvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open bvecs query file: " + path);
    }

    DenseBvecs out;
    int32_t d = 0;
    while (true) {
        if (!in.read(reinterpret_cast<char*>(&d), sizeof(int32_t))) {
            break;
        }
        if (d <= 0) {
            throw std::runtime_error("Invalid dimension in bvecs query file: " + path);
        }

        if (out.dim == 0) {
            out.dim = d;
        } else if (out.dim != d) {
            throw std::runtime_error("Inconsistent dimensions in bvecs query file: " + path);
        }

        const size_t old_size = out.values.size();
        out.values.resize(old_size + static_cast<size_t>(d));
        const size_t bytes = static_cast<size_t>(d) * sizeof(uint8_t);
        if (!in.read(reinterpret_cast<char*>(out.values.data() + old_size), static_cast<std::streamsize>(bytes))) {
            throw std::runtime_error("Truncated bvecs query payload: " + path);
        }

        ++out.num;
    }

    if (out.num <= 0 || out.dim <= 0) {
        throw std::runtime_error("No query vectors found in bvecs file: " + path);
    }

    return out;
}

std::vector<uint8_t> build_synthetic_metadata(size_t n) {
    if (n == 0) {
        throw std::runtime_error("Index has zero elements");
    }
    const size_t group_size = (n + kNumBuckets - 1) / kNumBuckets;
    std::vector<uint8_t> metadata(n, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t gid = i / group_size;
        if (gid >= kNumBuckets) {
            gid = kNumBuckets - 1;
        }
        metadata[i] = static_cast<uint8_t>(gid);
    }
    return metadata;
}

CompressedMetadata compress_metadata_blocks(const std::vector<uint8_t>& metadata) {
    CompressedMetadata out;
    const size_t nblocks = (metadata.size() + kBlockSize - 1) / kBlockSize;

    out.raw_block_sizes.resize(nblocks, 0);
    out.lz4_blocks.resize(nblocks);
    out.deflate_blocks.resize(nblocks);

    for (size_t bid = 0; bid < nblocks; ++bid) {
        const size_t start = bid * kBlockSize;
        const size_t raw_len = std::min(kBlockSize, metadata.size() - start);
        out.raw_block_sizes[bid] = raw_len;

        const int raw_len_i = static_cast<int>(raw_len);
        const int lz4_bound = LZ4_compressBound(raw_len_i);
        if (lz4_bound <= 0) {
            throw std::runtime_error("LZ4_compressBound failed");
        }

        out.lz4_blocks[bid].resize(static_cast<size_t>(lz4_bound));
        const int lz4_size = LZ4_compress_default(
            reinterpret_cast<const char*>(metadata.data() + start),
            out.lz4_blocks[bid].data(),
            raw_len_i,
            lz4_bound);
        if (lz4_size <= 0) {
            throw std::runtime_error("LZ4 compression failed at block " + std::to_string(bid));
        }
        out.lz4_blocks[bid].resize(static_cast<size_t>(lz4_size));

        uLongf deflate_bound = compressBound(static_cast<uLong>(raw_len));
        out.deflate_blocks[bid].resize(static_cast<size_t>(deflate_bound));
        int zret = compress2(
            out.deflate_blocks[bid].data(),
            &deflate_bound,
            metadata.data() + start,
            static_cast<uLong>(raw_len),
            Z_DEFAULT_COMPRESSION);
        if (zret != Z_OK) {
            throw std::runtime_error("Deflate compression failed at block " + std::to_string(bid));
        }
        out.deflate_blocks[bid].resize(static_cast<size_t>(deflate_bound));
    }

    return out;
}

inline uint64_t elapsed_ns(const Clock::time_point& t0, const Clock::time_point& t1) {
    return static_cast<uint64_t>(std::chrono::duration_cast<Ns>(t1 - t0).count());
}

uint64_t decompress_lz4_block(
    const CompressedMetadata& cm,
    size_t block_id,
    std::vector<uint8_t>& scratch) {
    const size_t raw_len = cm.raw_block_sizes[block_id];
    if (scratch.size() < raw_len) {
        throw std::runtime_error("Scratch buffer is smaller than raw block size");
    }

    const auto t0 = Clock::now();
    const int ret = LZ4_decompress_safe(
        cm.lz4_blocks[block_id].data(),
        reinterpret_cast<char*>(scratch.data()),
        static_cast<int>(cm.lz4_blocks[block_id].size()),
        static_cast<int>(raw_len));
    const auto t1 = Clock::now();

    if (ret != static_cast<int>(raw_len)) {
        throw std::runtime_error("LZ4 decompression failed at block " + std::to_string(block_id));
    }

    return elapsed_ns(t0, t1);
}

uint64_t decompress_deflate_block(
    const CompressedMetadata& cm,
    size_t block_id,
    std::vector<uint8_t>& scratch) {
    const size_t raw_len = cm.raw_block_sizes[block_id];
    if (scratch.size() < raw_len) {
        throw std::runtime_error("Scratch buffer is smaller than raw block size");
    }

    uLongf dst_len = static_cast<uLongf>(raw_len);
    const auto t0 = Clock::now();
    const int ret = uncompress(
        scratch.data(),
        &dst_len,
        cm.deflate_blocks[block_id].data(),
        static_cast<uLong>(cm.deflate_blocks[block_id].size()));
    const auto t1 = Clock::now();

    if (ret != Z_OK || dst_len != raw_len) {
        throw std::runtime_error("Deflate decompression failed at block " + std::to_string(block_id));
    }

    return elapsed_ns(t0, t1);
}

size_t block_id_for_node(size_t node_id) {
    return node_id / kBlockSize;
}

template <typename DistT, typename QueryT>
QueryTiming profile_single_query(
    const hnswlib::HierarchicalNSW<DistT>& index,
    const QueryT* query,
    size_t ef,
    size_t k,
    const CompressedMetadata& cm,
    std::vector<uint8_t>& lz4_scratch,
    std::vector<uint8_t>& deflate_scratch) {
    QueryTiming qt;

    if (index.cur_element_count == 0) {
        return qt;
    }

    hnswlib::tableint curr_obj = index.enterpoint_node_;
    if (static_cast<int>(curr_obj) == -1) {
        return qt;
    }

    DistT curdist = std::numeric_limits<DistT>::max();
    {
        const auto t0 = Clock::now();
        const DistT dist = index.fstdistfunc_(query, index.getDataByInternalId(curr_obj), index.dist_func_param_);
        const auto t1 = Clock::now();
        qt.upper_distance_ns += elapsed_ns(t0, t1);
        curdist = dist;
    }

    for (int level = index.maxlevel_; level > 0; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            ++qt.expanded_nodes_upper;
            uint64_t loop_dist_ns = 0;
            const auto loop_t0 = Clock::now();

            auto* data = reinterpret_cast<unsigned int*>(index.get_linklist(curr_obj, level));
            const int size = index.getListCount(data);
            auto* datal = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int i = 0; i < size; ++i) {
                const hnswlib::tableint cand = datal[i];
                if (cand >= index.cur_element_count) {
                    throw std::runtime_error("Corrupted graph link: candidate id out of range");
                }

                const auto td0 = Clock::now();
                const DistT d = index.fstdistfunc_(query, index.getDataByInternalId(cand), index.dist_func_param_);
                const auto td1 = Clock::now();
                loop_dist_ns += elapsed_ns(td0, td1);

                if (d < curdist) {
                    curdist = d;
                    curr_obj = cand;
                    changed = true;
                }
            }

            const auto loop_t1 = Clock::now();
            const uint64_t loop_total_ns = elapsed_ns(loop_t0, loop_t1);
            qt.upper_distance_ns += loop_dist_ns;
            qt.upper_traversal_ns += (loop_total_ns >= loop_dist_ns) ? (loop_total_ns - loop_dist_ns) : 0;
        }
    }

    using Pair = std::pair<DistT, hnswlib::tableint>;
    using PQ = std::priority_queue<Pair, std::vector<Pair>, typename hnswlib::HierarchicalNSW<DistT>::CompareByFirst>;

    hnswlib::VisitedList* vl = index.visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type* visited = vl->mass;
    const hnswlib::vl_type tag = vl->curV;

    PQ top_candidates;
    PQ candidate_set;

    DistT lower_bound = std::numeric_limits<DistT>::max();

    {
        const auto td0 = Clock::now();
        const DistT d = index.fstdistfunc_(query, index.getDataByInternalId(curr_obj), index.dist_func_param_);
        const auto td1 = Clock::now();
        qt.level0_distance_ns += elapsed_ns(td0, td1);

        lower_bound = d;

        const auto tc0 = Clock::now();
        candidate_set.emplace(-d, curr_obj);
        const auto tc1 = Clock::now();
        qt.candidate_update_ns += elapsed_ns(tc0, tc1);

        visited[curr_obj] = tag;
    }

    while (!candidate_set.empty()) {
        uint64_t loop_dist_ns = 0;
        uint64_t loop_candidate_ns = 0;
        uint64_t loop_lz4_naive_ns = 0;
        uint64_t loop_lz4_common_ns = 0;
        uint64_t loop_deflate_naive_ns = 0;
        uint64_t loop_deflate_common_ns = 0;

        const auto loop_t0 = Clock::now();

        const Pair current = candidate_set.top();
        const DistT candidate_dist = -current.first;

        bool stop = false;
        {
            if (candidate_dist > lower_bound && top_candidates.size() >= ef) {
                stop = true;
            }
        }
        if (stop) {
            const auto loop_t1 = Clock::now();
            const uint64_t loop_total_ns = elapsed_ns(loop_t0, loop_t1);
            const uint64_t loop_decomp_ns = loop_lz4_naive_ns + loop_lz4_common_ns +
                                            loop_deflate_naive_ns + loop_deflate_common_ns;
            qt.level0_distance_ns += loop_dist_ns;
            qt.candidate_update_ns += loop_candidate_ns;
            qt.lz4_naive_ns += loop_lz4_naive_ns;
            qt.lz4_common_ns += loop_lz4_common_ns;
            qt.deflate_naive_ns += loop_deflate_naive_ns;
            qt.deflate_common_ns += loop_deflate_common_ns;
            const uint64_t non_trav_ns = loop_dist_ns + loop_candidate_ns + loop_decomp_ns;
            qt.level0_traversal_ns += (loop_total_ns >= non_trav_ns) ? (loop_total_ns - non_trav_ns) : 0;
            break;
        }

        {
            const auto tc0 = Clock::now();
            candidate_set.pop();
            const auto tc1 = Clock::now();
            loop_candidate_ns += elapsed_ns(tc0, tc1);
        }
        ++qt.expanded_nodes_level0;

        const hnswlib::tableint current_id = current.second;
        auto* data = reinterpret_cast<int*>(index.get_linklist0(current_id));
        const size_t size = index.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));

        std::unordered_set<size_t> unique_blocks;
        unique_blocks.reserve(size);

        for (size_t j = 1; j <= size; ++j) {
            const hnswlib::tableint nid = static_cast<hnswlib::tableint>(*(data + j));
            if (nid >= index.cur_element_count) {
                throw std::runtime_error("Corrupted graph link at level0: candidate id out of range");
            }

            const size_t bid = block_id_for_node(static_cast<size_t>(nid));
            unique_blocks.insert(bid);

            loop_lz4_naive_ns += decompress_lz4_block(cm, bid, lz4_scratch);
            loop_deflate_naive_ns += decompress_deflate_block(cm, bid, deflate_scratch);
        }

        for (size_t bid : unique_blocks) {
            loop_lz4_common_ns += decompress_lz4_block(cm, bid, lz4_scratch);
            loop_deflate_common_ns += decompress_deflate_block(cm, bid, deflate_scratch);
        }

        for (size_t j = 1; j <= size; ++j) {
            const hnswlib::tableint candidate_id = static_cast<hnswlib::tableint>(*(data + j));

            if (visited[candidate_id] == tag) {
                continue;
            }
            visited[candidate_id] = tag;

            const auto td0 = Clock::now();
            const DistT dist = index.fstdistfunc_(query, index.getDataByInternalId(candidate_id), index.dist_func_param_);
            const auto td1 = Clock::now();
            loop_dist_ns += elapsed_ns(td0, td1);

            const bool consider = (top_candidates.size() < ef) || (lower_bound > dist);
            if (consider) {
                const auto tc0 = Clock::now();
                candidate_set.emplace(-dist, candidate_id);
                if (!index.isMarkedDeleted(candidate_id)) {
                    top_candidates.emplace(dist, candidate_id);
                }
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
                const auto tc1 = Clock::now();
                loop_candidate_ns += elapsed_ns(tc0, tc1);
            }
        }

        const auto loop_t1 = Clock::now();
        const uint64_t loop_total_ns = elapsed_ns(loop_t0, loop_t1);
        const uint64_t loop_decomp_ns = loop_lz4_naive_ns + loop_lz4_common_ns +
                                        loop_deflate_naive_ns + loop_deflate_common_ns;

        qt.level0_distance_ns += loop_dist_ns;
        qt.candidate_update_ns += loop_candidate_ns;
        qt.lz4_naive_ns += loop_lz4_naive_ns;
        qt.lz4_common_ns += loop_lz4_common_ns;
        qt.deflate_naive_ns += loop_deflate_naive_ns;
        qt.deflate_common_ns += loop_deflate_common_ns;

        const uint64_t non_trav_ns = loop_dist_ns + loop_candidate_ns + loop_decomp_ns;
        qt.level0_traversal_ns += (loop_total_ns >= non_trav_ns) ? (loop_total_ns - non_trav_ns) : 0;
    }

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }

    index.visited_list_pool_->releaseVisitedList(vl);
    return qt;
}

std::string resolve_log_path() {
    std::array<char, 4096> exe_buf{};
    const ssize_t len = ::readlink("/proc/self/exe", exe_buf.data(), exe_buf.size() - 1);
    if (len > 0) {
        exe_buf[static_cast<size_t>(len)] = '\0';
        fs::path p(exe_buf.data());
        if (!p.parent_path().empty()) {
            return (p.parent_path() / "profile.log").string();
        }
    }
    return (fs::current_path() / "profile.log").string();
}

inline double pct(uint64_t part, uint64_t whole) {
    if (whole == 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

inline double ns_to_ms(uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

void write_log(const std::string& preferred_path, const AggregateTiming& agg) {
    std::ofstream out(preferred_path);
    std::string used_path = preferred_path;

    if (!out.is_open()) {
        used_path = (fs::current_path() / "profile.log").string();
        out.open(used_path);
    }
    if (!out.is_open()) {
        throw std::runtime_error("Failed to create profile.log in executable directory and current directory");
    }

    const uint64_t traversal_ns = agg.total.upper_traversal_ns + agg.total.level0_traversal_ns;
    const uint64_t distance_ns = agg.total.upper_distance_ns + agg.total.level0_distance_ns;
    const uint64_t candidate_ns = agg.total.candidate_update_ns;
    const uint64_t baseline_ns = traversal_ns + distance_ns + candidate_ns;

    out << std::fixed << std::setprecision(6);
    out << "Section 3 Decompression vs Traversal Profiling\n";
    out << "log_path: " << used_path << "\n\n";

    out << "[Run Config]\n";
    out << "queries_used: " << agg.queries << "\n";
    out << "num_elements: " << agg.num_elements << "\n";
    out << "query_format: " << agg.query_format << "\n";
    out << "query_dim: " << agg.query_dim << "\n";
    out << "graph_dim: " << agg.graph_dim << "\n";
    out << "k: " << agg.k << "\n";
    out << "ef_search_effective: " << agg.ef_search << "\n";
    out << "block_size: " << agg.block_size << "\n\n";

    out << "[Totals - Decompression]\n";
    out << "(1) naive_lz4_ns: " << agg.total.lz4_naive_ns << " (" << ns_to_ms(agg.total.lz4_naive_ns) << " ms)\n";
    out << "(1) naive_deflate_ns: " << agg.total.deflate_naive_ns << " (" << ns_to_ms(agg.total.deflate_naive_ns) << " ms)\n";
    out << "(2) common_lz4_ns: " << agg.total.lz4_common_ns << " (" << ns_to_ms(agg.total.lz4_common_ns) << " ms)\n";
    out << "(2) common_deflate_ns: " << agg.total.deflate_common_ns << " (" << ns_to_ms(agg.total.deflate_common_ns) << " ms)\n\n";

    out << "[Totals - Traversal Pipeline]\n";
    out << "(3) graph_traversal_ns: " << traversal_ns << " (" << ns_to_ms(traversal_ns) << " ms)\n";
    out << "    upper_traversal_ns: " << agg.total.upper_traversal_ns << " (" << ns_to_ms(agg.total.upper_traversal_ns) << " ms)\n";
    out << "    level0_traversal_ns: " << agg.total.level0_traversal_ns << " (" << ns_to_ms(agg.total.level0_traversal_ns) << " ms)\n";
    out << "(4) distance_calculation_ns: " << distance_ns << " (" << ns_to_ms(distance_ns) << " ms)\n";
    out << "    upper_distance_ns: " << agg.total.upper_distance_ns << " (" << ns_to_ms(agg.total.upper_distance_ns) << " ms)\n";
    out << "    level0_distance_ns: " << agg.total.level0_distance_ns << " (" << ns_to_ms(agg.total.level0_distance_ns) << " ms)\n";
    out << "(5) candidate_update_ns: " << candidate_ns << " (" << ns_to_ms(candidate_ns) << " ms)\n";
    out << "baseline_sum_ns=(3)+(4)+(5): " << baseline_ns << " (" << ns_to_ms(baseline_ns) << " ms)\n\n";

    out << "[Node Access Counts]\n";
    out << "expanded_nodes_upper: " << agg.total.expanded_nodes_upper << "\n";
    out << "expanded_nodes_level0: " << agg.total.expanded_nodes_level0 << "\n\n";

    out << "[Portion of Decompression vs Baseline Sum]\n";
    out << "naive_lz4 / baseline_sum: " << pct(agg.total.lz4_naive_ns, baseline_ns) << "%\n";
    out << "naive_deflate / baseline_sum: " << pct(agg.total.deflate_naive_ns, baseline_ns) << "%\n";
    out << "common_lz4 / baseline_sum: " << pct(agg.total.lz4_common_ns, baseline_ns) << "%\n";
    out << "common_deflate / baseline_sum: " << pct(agg.total.deflate_common_ns, baseline_ns) << "%\n\n";

    out << "[Traversal/Distance/Candidate Percentage of Baseline Sum]\n";
    out << "(3) graph_traversal_pct: " << pct(traversal_ns, baseline_ns) << "%\n";
    out << "(4) distance_calculation_pct: " << pct(distance_ns, baseline_ns) << "%\n";
    out << "(5) candidate_update_pct: " << pct(candidate_ns, baseline_ns) << "%\n\n";

    const size_t node_accesses = std::max<size_t>(agg.total.expanded_nodes_level0, 1);
    out << "[Per-Node Access Averages (Level-0)]\n";
    out << "node_access_denominator: " << agg.total.expanded_nodes_level0 << "\n";
    out << "(1) naive_lz4_ns_per_node: " << (agg.total.lz4_naive_ns / node_accesses) << "\n";
    out << "(1) naive_deflate_ns_per_node: " << (agg.total.deflate_naive_ns / node_accesses) << "\n";
    out << "(2) common_lz4_ns_per_node: " << (agg.total.lz4_common_ns / node_accesses) << "\n";
    out << "(2) common_deflate_ns_per_node: " << (agg.total.deflate_common_ns / node_accesses) << "\n";
    out << "(3) graph_traversal_ns_per_node: " << (agg.total.level0_traversal_ns / node_accesses) << "\n";
    out << "(4) distance_calculation_ns_per_node: " << (agg.total.level0_distance_ns / node_accesses) << "\n";
    out << "(5) candidate_update_ns_per_node: " << (agg.total.candidate_update_ns / node_accesses) << "\n\n";

    const size_t q = std::max<size_t>(agg.queries, 1);
    out << "[Per-Query Averages]\n";
    out << "avg_naive_lz4_ns: " << (agg.total.lz4_naive_ns / q) << "\n";
    out << "avg_naive_deflate_ns: " << (agg.total.deflate_naive_ns / q) << "\n";
    out << "avg_common_lz4_ns: " << (agg.total.lz4_common_ns / q) << "\n";
    out << "avg_common_deflate_ns: " << (agg.total.deflate_common_ns / q) << "\n";
    out << "avg_upper_traversal_ns: " << (agg.total.upper_traversal_ns / q) << "\n";
    out << "avg_level0_traversal_ns: " << (agg.total.level0_traversal_ns / q) << "\n";
    out << "avg_upper_distance_ns: " << (agg.total.upper_distance_ns / q) << "\n";
    out << "avg_level0_distance_ns: " << (agg.total.level0_distance_ns / q) << "\n";
    out << "avg_candidate_update_ns: " << (agg.total.candidate_update_ns / q) << "\n";
    out << "avg_expanded_nodes_upper: " << (agg.total.expanded_nodes_upper / q) << "\n";
    out << "avg_expanded_nodes_level0: " << (agg.total.expanded_nodes_level0 / q) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        AggregateTiming agg;

        if (ends_with(args.query_path, ".fvecs")) {
            DenseFvecs queries = load_queries_fvecs(args.query_path);
            if (queries.num <= 0 || queries.dim <= 0) {
                throw std::runtime_error("Query file has no valid vectors");
            }

            hnswlib::L2Space space(static_cast<size_t>(queries.dim));
            auto index = load_graph_index_or_throw<float>(&space, args.graph_path, ".fvecs");

            const size_t graph_data_bytes = index->label_offset_ - index->offsetData_;
            const size_t expected_data_bytes = static_cast<size_t>(queries.dim) * sizeof(float);
            if (graph_data_bytes != expected_data_bytes) {
                throw std::runtime_error(
                    "Graph/query type mismatch for fvecs. Expected graph vector bytes " +
                    std::to_string(expected_data_bytes) + " (dim=" + std::to_string(queries.dim) +
                    "), got " + std::to_string(graph_data_bytes) + ". Use a graph built from .fvecs.");
            }
            const size_t graph_dim = graph_data_bytes / sizeof(float);

            const size_t n = index->getCurrentElementCount();
            nElements = static_cast<int>(n);
            std::vector<uint8_t> metadata = build_synthetic_metadata(n);
            CompressedMetadata cm = compress_metadata_blocks(metadata);

            const size_t effective_ef = std::max<size_t>(64, args.k);
            index->setEf(effective_ef);

            const size_t queries_to_run = std::min<size_t>(args.num_queries, static_cast<size_t>(queries.num));
            if (queries_to_run == 0) {
                throw std::runtime_error("No queries to run after clamping to available query vectors");
            }

            agg.queries = queries_to_run;
            agg.num_elements = n;
            agg.query_format = "fvecs";
            agg.query_dim = static_cast<size_t>(queries.dim);
            agg.graph_dim = graph_dim;
            agg.k = args.k;
            agg.ef_search = effective_ef;

            std::vector<uint8_t> lz4_scratch(kBlockSize, 0);
            std::vector<uint8_t> deflate_scratch(kBlockSize, 0);

            for (size_t qid = 0; qid < queries_to_run; ++qid) {
                const float* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);
                QueryTiming qt = profile_single_query(
                    *index,
                    qptr,
                    effective_ef,
                    args.k,
                    cm,
                    lz4_scratch,
                    deflate_scratch);

                agg.total.lz4_naive_ns += qt.lz4_naive_ns;
                agg.total.lz4_common_ns += qt.lz4_common_ns;
                agg.total.deflate_naive_ns += qt.deflate_naive_ns;
                agg.total.deflate_common_ns += qt.deflate_common_ns;
                agg.total.upper_traversal_ns += qt.upper_traversal_ns;
                agg.total.upper_distance_ns += qt.upper_distance_ns;
                agg.total.level0_traversal_ns += qt.level0_traversal_ns;
                agg.total.level0_distance_ns += qt.level0_distance_ns;
                agg.total.candidate_update_ns += qt.candidate_update_ns;
                agg.total.expanded_nodes_upper += qt.expanded_nodes_upper;
                agg.total.expanded_nodes_level0 += qt.expanded_nodes_level0;
            }
        } else if (ends_with(args.query_path, ".bvecs")) {
            DenseBvecs queries = load_queries_bvecs(args.query_path);
            if (queries.num <= 0 || queries.dim <= 0) {
                throw std::runtime_error("Query file has no valid vectors");
            }

            hnswlib::L2SpaceI space(static_cast<size_t>(queries.dim));
            auto index = load_graph_index_or_throw<int>(&space, args.graph_path, ".bvecs");

            const size_t graph_data_bytes = index->label_offset_ - index->offsetData_;
            const size_t expected_data_bytes = static_cast<size_t>(queries.dim) * sizeof(uint8_t);
            if (graph_data_bytes != expected_data_bytes) {
                throw std::runtime_error(
                    "Graph/query type mismatch for bvecs. Expected graph vector bytes " +
                    std::to_string(expected_data_bytes) + " (dim=" + std::to_string(queries.dim) +
                    "), got " + std::to_string(graph_data_bytes) + ". Use a graph built from .bvecs.");
            }
            const size_t graph_dim = graph_data_bytes / sizeof(uint8_t);

            const size_t n = index->getCurrentElementCount();
            nElements = static_cast<int>(n);
            std::vector<uint8_t> metadata = build_synthetic_metadata(n);
            CompressedMetadata cm = compress_metadata_blocks(metadata);

            const size_t effective_ef = std::max<size_t>(64, args.k);
            index->setEf(effective_ef);

            const size_t queries_to_run = std::min<size_t>(args.num_queries, static_cast<size_t>(queries.num));
            if (queries_to_run == 0) {
                throw std::runtime_error("No queries to run after clamping to available query vectors");
            }

            agg.queries = queries_to_run;
            agg.num_elements = n;
            agg.query_format = "bvecs";
            agg.query_dim = static_cast<size_t>(queries.dim);
            agg.graph_dim = graph_dim;
            agg.k = args.k;
            agg.ef_search = effective_ef;

            std::vector<uint8_t> lz4_scratch(kBlockSize, 0);
            std::vector<uint8_t> deflate_scratch(kBlockSize, 0);

            for (size_t qid = 0; qid < queries_to_run; ++qid) {
                const uint8_t* qptr = queries.values.data() + qid * static_cast<size_t>(queries.dim);
                QueryTiming qt = profile_single_query(
                    *index,
                    qptr,
                    effective_ef,
                    args.k,
                    cm,
                    lz4_scratch,
                    deflate_scratch);

                agg.total.lz4_naive_ns += qt.lz4_naive_ns;
                agg.total.lz4_common_ns += qt.lz4_common_ns;
                agg.total.deflate_naive_ns += qt.deflate_naive_ns;
                agg.total.deflate_common_ns += qt.deflate_common_ns;
                agg.total.upper_traversal_ns += qt.upper_traversal_ns;
                agg.total.upper_distance_ns += qt.upper_distance_ns;
                agg.total.level0_traversal_ns += qt.level0_traversal_ns;
                agg.total.level0_distance_ns += qt.level0_distance_ns;
                agg.total.candidate_update_ns += qt.candidate_update_ns;
                agg.total.expanded_nodes_upper += qt.expanded_nodes_upper;
                agg.total.expanded_nodes_level0 += qt.expanded_nodes_level0;
            }
        } else {
            throw std::runtime_error("query file must end with .fvecs or .bvecs");
        }

        const std::string preferred_log = resolve_log_path();
        write_log(preferred_log, agg);

        std::cout << "Profiling completed. queries_used=" << agg.queries
                  << ", ef_search=" << agg.ef_search
                  << ", query_format=" << agg.query_format
                  << ", log=" << preferred_log << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
