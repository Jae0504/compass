// FID/TB compression benchmark.
//
// Datasets:
//   sift1m  – /storage/jykang5/fid_tb/n_filter_100/sift1m/  (1% selectivity, 100 filters)
//   hnm     – /storage/jykang5/fid_tb/hnm/
//   laion   – /storage/jykang5/fid_tb/laion/
//
// For each dataset the benchmark:
//   1. Reads manifest.json to enumerate attributes.
//   2. For each attribute: reads FID (1 byte/node), scans for max bucket value
//      to determine used_bits, reads TB (32 bytes/node), reorders TB from
//      node-major to bucket-major keeping only the first used_bits bucket layers.
//   3. Compresses each attribute FID independently (chunk_size per attribute).
//   4. Concatenates all reordered TB slices and compresses the combined buffer.
//   5. Measures CPU cycles (compression) and reports compression ratios.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <x86intrin.h>

#include <lz4.h>
#include <qpl/c_api/dictionary.h>
#include <qpl/c_api/huffman_table.h>
#include <qpl/qpl.h>

// nlohmann/json – for reading manifest.json
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// ── Constants ─────────────────────────────────────────────────────────────────

constexpr std::size_t kOneKB = 1024ULL;
constexpr std::size_t kOneMB = 1024ULL * kOneKB;
constexpr std::size_t kTbBitsPerNode  = 256;
constexpr std::size_t kTbBytesPerNode = kTbBitsPerNode / 8;  // 32 bytes

// ── Dataset configuration ─────────────────────────────────────────────────────

struct DatasetConfig {
    std::string name;
    std::string dir;
};

static const std::vector<DatasetConfig> kDatasets = {
    {"sift1m", "/storage/jykang5/fid_tb/n_filter_100/sift1b/"},
    {"hnm",    "/storage/jykang5/fid_tb/hnm/"},
    {"laion",  "/storage/jykang5/fid_tb/laion/"},
};

// ── Chunk-size policy ─────────────────────────────────────────────────────────
// chunk_size = smallest power-of-two >= (data_bytes / 8), clamped to [1KB, 2MB].

std::size_t next_chunk_size(std::size_t data_bytes) {
    const std::size_t base = data_bytes / 8;
    std::size_t chunk = kOneKB;                 // minimum 1 KB
    while (chunk < base && chunk < 2 * kOneMB)
        chunk *= 2;
    return std::min(chunk, 2 * kOneMB);         // maximum 2 MB
}

// ── FID utilities ─────────────────────────────────────────────────────────────

// Scan the FID byte array for the maximum bucket id; used_bits = max + 1.
std::size_t fid_used_bits(const std::vector<uint8_t>& fid) {
    const uint8_t mx = *std::max_element(fid.begin(), fid.end());
    return static_cast<std::size_t>(mx) + 1;
}

// ── TB reordering ─────────────────────────────────────────────────────────────
//
// Node-major (on disk):
//   src[i * 32 + b/8] bit(b%8)  = connector bit for (node i, bucket b)
//
// Bucket-major (output, only first used_bits buckets):
//   dst[(b * N + i) / 8] bit((b*N+i)%8) = connector bit for (node i, bucket b)
//   total size = used_bits * ceil(N/8) bytes

std::vector<uint8_t> reorder_tb_partial(const std::vector<uint8_t>& src,
                                         std::size_t N,
                                         std::size_t used_bits) {
    const std::size_t bytes_per_bucket = (N + 7) / 8;
    std::vector<uint8_t> dst(used_bits * bytes_per_bucket, 0);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t b = 0; b < used_bits; ++b) {
            const uint8_t bit =
                (src[i * kTbBytesPerNode + b / 8] >> (b % 8)) & 1u;
            if (bit) {
                const std::size_t pos = b * N + i;
                dst[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
            }
        }
    }
    return dst;
}

// ── JSONL payload helpers ─────────────────────────────────────────────────────

std::vector<json> load_jsonl_rows(const std::string& path, std::size_t n_rows) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open payload file: " + path);
    std::vector<json> rows;
    rows.reserve(n_rows);
    std::string line;
    while (rows.size() < n_rows && std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            rows.push_back(json::parse(line));
        } catch (...) {
            rows.push_back(json::object());
        }
    }
    return rows;
}

// Extract attribute values from payloads and serialize as raw float32 bytes.
std::vector<uint8_t> attr_to_float32_bytes(const std::vector<json>& rows,
                                            const std::string& key) {
    std::vector<uint8_t> out;
    out.reserve(rows.size() * 4);
    for (const auto& row : rows) {
        float val = 0.0f;
        if (row.is_object() && row.contains(key) && !row[key].is_null()) {
            if (row[key].is_number())
                val = static_cast<float>(row[key].get<double>());
        }
        uint8_t bytes[4];
        std::memcpy(bytes, &val, 4);
        out.insert(out.end(), bytes, bytes + 4);
    }
    return out;
}

// Compute actual metadata storage bytes from payloads.
// Only counts attributes listed in manifest["attributes"] (the FID-indexed ones):
//   number → 4 bytes, string → string length bytes, null/missing → 4 bytes.
std::size_t compute_metadata_bytes(const std::vector<json>& rows,
                                    const json& attributes) {
    std::size_t total = 0;
    for (const auto& row : rows) {
        if (!row.is_object()) { total += 4 * attributes.size(); continue; }
        for (const auto& attr : attributes) {
            const std::string key = attr["key"].get<std::string>();
            if (!row.contains(key) || row[key].is_null()) {
                total += 4;
            } else if (row[key].is_number()) {
                total += 4;
            } else if (row[key].is_string()) {
                total += row[key].get<std::string>().size();
            } else {
                total += row[key].dump().size();
            }
        }
    }
    return total;
}

// Resolve payload path: prefer manifest's payload_path field, else fallback.
std::string resolve_payload_path(const json& manifest,
                                  const std::string& dataset_name) {
    const std::string from_manifest = manifest.value("payload_path", std::string{});
    if (!from_manifest.empty() && fs::exists(from_manifest)) return from_manifest;
    return "/home/jykang5/compass/json_dataset/" + dataset_name + "_payloads.jsonl";
}

// ── File I/O ──────────────────────────────────────────────────────────────────
// Binary files: 8-byte uint64_t header (element count) followed by raw data.

std::vector<uint8_t> read_binary_file(const std::string& path,
                                       std::size_t expected_data_bytes = 0) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    const std::size_t file_sz = static_cast<std::size_t>(f.tellg());
    constexpr std::size_t kHdr = 8;
    if (file_sz < kHdr)
        throw std::runtime_error("File too small: " + path);

    f.seekg(0);
    uint64_t hdr_n = 0;
    f.read(reinterpret_cast<char*>(&hdr_n), static_cast<std::streamsize>(kHdr));

    const std::size_t data_sz = file_sz - kHdr;
    if (expected_data_bytes > 0 && data_sz != expected_data_bytes) {
        std::ostringstream oss;
        oss << "Size mismatch for " << path
            << ": expected " << expected_data_bytes
            << " bytes, got " << data_sz
            << " (header N=" << hdr_n << ")";
        throw std::runtime_error(oss.str());
    }
    std::vector<uint8_t> buf(data_sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(data_sz));
    if (!f)
        throw std::runtime_error("Read failed: " + path);
    return buf;
}

// ── Chunk splitting ───────────────────────────────────────────────────────────

using RawBlocks = std::vector<std::vector<uint8_t>>;

RawBlocks split_into_chunks(const std::vector<uint8_t>& data, std::size_t chunk_sz) {
    RawBlocks out;
    if (chunk_sz == 0 || data.empty()) return out;
    for (std::size_t off = 0; off < data.size(); off += chunk_sz) {
        const std::size_t len = std::min(chunk_sz, data.size() - off);
        out.emplace_back(data.begin() + static_cast<std::ptrdiff_t>(off),
                         data.begin() + static_cast<std::ptrdiff_t>(off + len));
    }
    return out;
}

// ── Compressed storage ────────────────────────────────────────────────────────

struct ChunkedCompressed {
    std::vector<std::vector<uint8_t>> chunks;
    std::vector<uint32_t> original_chunk_sizes;
    std::vector<uint8_t> dictionary_raw;
    bool iaa_dynamic = false;
    std::size_t compressed_bytes = 0;
};

static std::size_t total_original_bytes(const ChunkedCompressed& c) {
    std::size_t total = 0;
    for (uint32_t s : c.original_chunk_sizes) total += s;
    return total;
}

// ── LZ4 ───────────────────────────────────────────────────────────────────────

ChunkedCompressed compress_lz4(const RawBlocks& raw_blocks, uint64_t& cycles) {
    ChunkedCompressed out;
    out.chunks.resize(raw_blocks.size());
    out.original_chunk_sizes.resize(raw_blocks.size(), 0);

    const uint64_t t0 = __rdtsc();
    for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
        const auto& blk = raw_blocks[i];
        const int max_comp = LZ4_compressBound(static_cast<int>(blk.size()));
        if (max_comp <= 0) throw std::runtime_error("LZ4_compressBound failed");
        out.chunks[i].resize(static_cast<std::size_t>(max_comp));
        const int comp = LZ4_compress_default(
            reinterpret_cast<const char*>(blk.data()),
            reinterpret_cast<char*>(out.chunks[i].data()),
            static_cast<int>(blk.size()), max_comp);
        if (comp <= 0) throw std::runtime_error("LZ4_compress_default failed");
        out.chunks[i].resize(static_cast<std::size_t>(comp));
        out.compressed_bytes += static_cast<std::size_t>(comp);
        out.original_chunk_sizes[i] = static_cast<uint32_t>(blk.size());
    }
    const uint64_t t1 = __rdtsc();
    cycles = t1 - t0;
    return out;
}

bool decompress_lz4(const ChunkedCompressed& comp, std::vector<uint8_t>& out,
                    std::string& err) {
    out.assign(total_original_bytes(comp), 0);
    std::size_t off = 0;
    for (std::size_t i = 0; i < comp.chunks.size(); ++i) {
        const int expected = static_cast<int>(comp.original_chunk_sizes[i]);
        const int got = LZ4_decompress_safe(
            reinterpret_cast<const char*>(comp.chunks[i].data()),
            reinterpret_cast<char*>(out.data() + off),
            static_cast<int>(comp.chunks[i].size()), expected);
        if (got != expected) {
            err = "LZ4 decomp failed at chunk " + std::to_string(i);
            return false;
        }
        off += static_cast<std::size_t>(expected);
    }
    return true;
}

// ── Deflate (QPL software) ────────────────────────────────────────────────────

ChunkedCompressed compress_deflate(const RawBlocks& raw_blocks, uint64_t& cycles) {
    ChunkedCompressed out;
    out.chunks.resize(raw_blocks.size());
    out.original_chunk_sizes.resize(raw_blocks.size(), 0);
    if (raw_blocks.empty()) { cycles = 0; return out; }

    const qpl_path_t path = qpl_path_software;
    uint32_t job_size = 0;
    if (qpl_get_job_size(path, &job_size) != QPL_STS_OK)
        throw std::runtime_error("Deflate: qpl_get_job_size failed");
    std::vector<uint8_t> jbuf(job_size, 0);
    qpl_job* job = reinterpret_cast<qpl_job*>(jbuf.data());
    if (qpl_init_job(path, job) != QPL_STS_OK)
        throw std::runtime_error("Deflate: qpl_init_job failed");

    struct HuffGuard {
        qpl_huffman_table_t t = nullptr;
        ~HuffGuard() { if (t) qpl_huffman_table_destroy(t); }
    } hg;
    allocator_t alloc = {malloc, free};
    if (qpl_deflate_huffman_table_create(compression_table_type, path, alloc, &hg.t)
            != QPL_STS_OK) {
        qpl_fini_job(job);
        throw std::runtime_error("Deflate: Huffman table create failed");
    }

    const uint64_t t0 = __rdtsc();
    for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
        const auto& blk = raw_blocks[i];
        const uint32_t sz = static_cast<uint32_t>(blk.size());
        out.original_chunk_sizes[i] = sz;

        qpl_histogram hist{};
        if (qpl_gather_deflate_statistics(const_cast<uint8_t*>(blk.data()), sz,
                                          &hist, qpl_default_level, path)
                != QPL_STS_OK) {
            qpl_fini_job(job);
            throw std::runtime_error("Deflate: gather stats failed at chunk " +
                                     std::to_string(i));
        }
        if (qpl_huffman_table_init_with_histogram(hg.t, &hist) != QPL_STS_OK) {
            qpl_fini_job(job);
            throw std::runtime_error("Deflate: Huffman init failed at chunk " +
                                     std::to_string(i));
        }

        const uint32_t cap = qpl_get_safe_deflate_compression_buffer_size(sz);
        out.chunks[i].resize(cap);
        job->op            = qpl_op_compress;
        job->level         = qpl_default_level;
        job->next_in_ptr   = const_cast<uint8_t*>(blk.data());
        job->available_in  = static_cast<uint32_t>(blk.size());
        job->next_out_ptr  = out.chunks[i].data();
        job->available_out = cap;
        job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_OMIT_VERIFY;
        job->huffman_table = hg.t;
        job->dictionary    = nullptr;

        if (qpl_execute_job(job) != QPL_STS_OK) {
            qpl_fini_job(job);
            throw std::runtime_error("Deflate: compress failed at chunk " +
                                     std::to_string(i));
        }
        const std::size_t produced = static_cast<std::size_t>(job->total_out);
        out.chunks[i].resize(produced);
        out.compressed_bytes += produced;
    }
    const uint64_t t1 = __rdtsc();
    cycles = t1 - t0;
    qpl_fini_job(job);
    return out;
}

bool decompress_deflate(const ChunkedCompressed& comp, std::vector<uint8_t>& out,
                        std::string& err) {
    out.assign(total_original_bytes(comp), 0);
    if (comp.chunks.empty()) return true;

    const qpl_path_t path = qpl_path_software;
    uint32_t job_size = 0;
    if (qpl_get_job_size(path, &job_size) != QPL_STS_OK) {
        err = "Deflate decomp: qpl_get_job_size failed"; return false;
    }
    std::vector<uint8_t> jbuf(job_size, 0);
    qpl_job* job = reinterpret_cast<qpl_job*>(jbuf.data());
    if (qpl_init_job(path, job) != QPL_STS_OK) {
        err = "Deflate decomp: qpl_init_job failed"; return false;
    }

    std::size_t off = 0;
    for (std::size_t i = 0; i < comp.chunks.size(); ++i) {
        const uint32_t expected = comp.original_chunk_sizes[i];
        job->op            = qpl_op_decompress;
        job->next_in_ptr   = const_cast<uint8_t*>(comp.chunks[i].data());
        job->available_in  = static_cast<uint32_t>(comp.chunks[i].size());
        job->next_out_ptr  = out.data() + off;
        job->available_out = expected;
        job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST;
        job->dictionary    = nullptr;
        job->huffman_table = nullptr;

        if (qpl_execute_job(job) != QPL_STS_OK ||
            static_cast<uint32_t>(job->total_out) != expected) {
            err = "Deflate decomp failed at chunk " + std::to_string(i);
            qpl_fini_job(job);
            return false;
        }
        off += expected;
    }
    qpl_fini_job(job);
    return true;
}

// ── IAA (QPL hardware, async pipelined) ──────────────────────────────────────

class IaaJobPool {
public:
    explicit IaaJobPool(std::size_t n) : n_(n) {
        uint32_t job_size = 0;
        if (qpl_get_job_size(qpl_path_hardware, &job_size) != QPL_STS_OK)
            throw std::runtime_error("IAA: qpl_get_job_size failed");
        bufs_.resize(n_);
        jobs_.resize(n_, nullptr);
        for (std::size_t i = 0; i < n_; ++i) {
            bufs_[i] = std::make_unique<uint8_t[]>(job_size);
            jobs_[i] = reinterpret_cast<qpl_job*>(bufs_[i].get());
            if (qpl_init_job(qpl_path_hardware, jobs_[i]) != QPL_STS_OK)
                throw std::runtime_error("IAA: qpl_init_job failed at slot " +
                                         std::to_string(i));
        }
    }
    ~IaaJobPool() { for (auto* j : jobs_) if (j) qpl_fini_job(j); }
    qpl_job* get(std::size_t slot) { return jobs_[slot % n_]; }
    std::size_t size() const { return n_; }

private:
    std::size_t n_;
    std::vector<std::unique_ptr<uint8_t[]>> bufs_;
    std::vector<qpl_job*> jobs_;
};

ChunkedCompressed compress_iaa(const RawBlocks& raw_blocks, uint64_t& cycles) {
    ChunkedCompressed out;
    const std::size_t nchunks = raw_blocks.size();
    out.chunks.resize(nchunks);
    out.original_chunk_sizes.resize(nchunks, 0);
    if (nchunks == 0) { cycles = 0; return out; }

    constexpr std::size_t kQueueSize  = 128;
    constexpr std::size_t kWaitBatch  = 64;
    constexpr std::size_t kOutBufSize = 2ULL * kOneMB;
    constexpr std::size_t kDictRawSz  = 4ULL * kOneKB;

    const std::size_t pool_sz = std::min(kQueueSize, nchunks);
    IaaJobPool pool(pool_sz);

    for (std::size_t i = 0; i < nchunks; ++i) {
        if (raw_blocks[i].size() > kOutBufSize)
            throw std::runtime_error("IAA: raw block > 2MB at chunk " +
                                     std::to_string(i));
        out.original_chunk_sizes[i] =
            static_cast<uint32_t>(raw_blocks[i].size());
    }

    struct HuffGuard {
        qpl_huffman_table_t t = nullptr;
        ~HuffGuard() { if (t) qpl_huffman_table_destroy(t); }
    } hg;
    struct DictGuard {
        std::unique_ptr<uint8_t[]> buf;
        qpl_dictionary* dict = nullptr;
    } dg;

    allocator_t alloc = {malloc, free};
    if (qpl_deflate_huffman_table_create(
            compression_table_type, qpl_path_hardware, alloc, &hg.t) != QPL_STS_OK)
        throw std::runtime_error("IAA: Huffman table create failed");

    // Build dictionary from first kDictRawSz bytes
    {
        std::size_t collected = 0;
        for (const auto& blk : raw_blocks) {
            if (collected >= kDictRawSz) break;
            const std::size_t take = std::min(kDictRawSz - collected, blk.size());
            out.dictionary_raw.insert(out.dictionary_raw.end(),
                                      blk.begin(),
                                      blk.begin() + static_cast<std::ptrdiff_t>(take));
            collected += take;
        }
    }
    if (!out.dictionary_raw.empty()) {
        const sw_compression_level sw_lvl = SW_NONE;
        const hw_compression_level hw_lvl = HW_LEVEL_1;
        const std::size_t dsz =
            qpl_get_dictionary_size(sw_lvl, hw_lvl, out.dictionary_raw.size());
        if (dsz > 0) {
            dg.buf  = std::make_unique<uint8_t[]>(dsz);
            dg.dict = reinterpret_cast<qpl_dictionary*>(dg.buf.get());
            if (qpl_build_dictionary(dg.dict, sw_lvl, hw_lvl,
                                     out.dictionary_raw.data(),
                                     out.dictionary_raw.size()) != QPL_STS_OK) {
                dg.dict = nullptr;
                out.dictionary_raw.clear();
            }
        }
    }
    bool use_dict = (dg.dict != nullptr);

    std::vector<std::vector<uint8_t>> slot_bufs(pool_sz,
                                                  std::vector<uint8_t>(kOutBufSize));

    auto do_submit = [&](std::size_t ci) {
        const std::size_t slot = ci % pool_sz;
        qpl_job* job = pool.get(slot);
        const auto& blk = raw_blocks[ci];
        const uint32_t raw_sz = out.original_chunk_sizes[ci];

        qpl_histogram hist{};
        if (qpl_gather_deflate_statistics(const_cast<uint8_t*>(blk.data()), raw_sz,
                                          &hist, qpl_default_level,
                                          qpl_path_hardware) != QPL_STS_OK)
            throw std::runtime_error("IAA: gather stats failed at chunk " +
                                     std::to_string(ci));
        if (qpl_huffman_table_init_with_histogram(hg.t, &hist) != QPL_STS_OK)
            throw std::runtime_error("IAA: Huffman init failed at chunk " +
                                     std::to_string(ci));

        job->op            = qpl_op_compress;
        job->level         = qpl_default_level;
        job->next_in_ptr   = const_cast<uint8_t*>(blk.data());
        job->available_in  = raw_sz;
        job->next_out_ptr  = slot_bufs[slot].data();
        job->available_out = static_cast<uint32_t>(kOutBufSize);
        job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_OMIT_VERIFY;
        job->dictionary    = use_dict ? dg.dict : nullptr;
        job->huffman_table = hg.t;

        const uint64_t t0 = __rdtsc();
        qpl_status st = qpl_submit_job(job);
        const uint64_t t1 = __rdtsc();
        cycles += (t1 - t0);

        if (st == QPL_STS_NOT_SUPPORTED_MODE_ERR && use_dict) {
            use_dict = false; dg.dict = nullptr; out.dictionary_raw.clear();
            job->dictionary = nullptr;
            const uint64_t t2 = __rdtsc();
            st = qpl_submit_job(job);
            const uint64_t t3 = __rdtsc();
            cycles += (t3 - t2);
        }
        if (st != QPL_STS_OK)
            throw std::runtime_error("IAA: submit failed at chunk " +
                                     std::to_string(ci) + " status=" +
                                     std::to_string(st));
    };

    auto do_wait = [&](std::size_t ci) {
        const std::size_t slot = ci % pool_sz;
        qpl_job* job = pool.get(slot);
        const uint64_t t0 = __rdtsc();
        const qpl_status st = qpl_wait_job(job);
        const uint64_t t1 = __rdtsc();
        cycles += (t1 - t0);
        if (st != QPL_STS_OK)
            throw std::runtime_error("IAA: wait failed at chunk " +
                                     std::to_string(ci) + " status=" +
                                     std::to_string(st));
        const std::size_t produced = static_cast<std::size_t>(job->total_out);
        out.chunks[ci].assign(slot_bufs[slot].begin(),
                              slot_bufs[slot].begin() +
                              static_cast<std::ptrdiff_t>(produced));
        out.compressed_bytes += produced;
    };

    cycles = 0;
    std::size_t next_sub = 0, next_wait = 0;
    for (; next_sub < std::min(pool_sz, nchunks); ++next_sub)
        do_submit(next_sub);
    while (next_wait < nchunks) {
        const std::size_t wn = std::min(kWaitBatch, nchunks - next_wait);
        for (std::size_t k = 0; k < wn; ++k) do_wait(next_wait + k);
        next_wait += wn;
        const std::size_t sn = std::min(wn, nchunks - next_sub);
        for (std::size_t k = 0; k < sn; ++k) do_submit(next_sub + k);
        next_sub += sn;
    }
    return out;
}

bool decompress_iaa(const ChunkedCompressed& comp, std::vector<uint8_t>& out,
                    std::string& err) {
    out.assign(total_original_bytes(comp), 0);
    if (comp.chunks.empty()) return true;

    constexpr std::size_t kQueueSize = 128;
    constexpr std::size_t kWaitBatch = 64;
    const std::size_t pool_sz = std::min(kQueueSize, comp.chunks.size());
    IaaJobPool pool(pool_sz);

    struct HuffGuard {
        qpl_huffman_table_t t = nullptr;
        ~HuffGuard() { if (t) qpl_huffman_table_destroy(t); }
    } hg;
    struct DictGuard {
        std::unique_ptr<uint8_t[]> buf;
        qpl_dictionary* dict = nullptr;
    } dg;

    allocator_t alloc = {malloc, free};
    if (qpl_deflate_huffman_table_create(
            decompression_table_type, qpl_path_hardware, alloc, &hg.t) != QPL_STS_OK) {
        err = "IAA decomp: Huffman table create failed"; return false;
    }

    if (!comp.dictionary_raw.empty()) {
        const sw_compression_level sw_lvl = SW_NONE;
        const hw_compression_level hw_lvl = HW_LEVEL_1;
        const std::size_t dsz =
            qpl_get_dictionary_size(sw_lvl, hw_lvl, comp.dictionary_raw.size());
        if (dsz > 0) {
            dg.buf  = std::make_unique<uint8_t[]>(dsz);
            dg.dict = reinterpret_cast<qpl_dictionary*>(dg.buf.get());
            if (qpl_build_dictionary(dg.dict, sw_lvl, hw_lvl,
                                     comp.dictionary_raw.data(),
                                     comp.dictionary_raw.size()) != QPL_STS_OK)
                dg.dict = nullptr;
        }
    }

    std::vector<std::size_t> offsets(comp.chunks.size(), 0);
    for (std::size_t i = 1; i < offsets.size(); ++i)
        offsets[i] = offsets[i - 1] + comp.original_chunk_sizes[i - 1];

    bool ok = true;
    auto do_submit = [&](std::size_t ci) -> bool {
        qpl_job* job = pool.get(ci % pool_sz);
        job->op            = qpl_op_decompress;
        job->next_in_ptr   = const_cast<uint8_t*>(comp.chunks[ci].data());
        job->available_in  = static_cast<uint32_t>(comp.chunks[ci].size());
        job->next_out_ptr  = out.data() + offsets[ci];
        job->available_out = comp.original_chunk_sizes[ci];
        job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST;
        job->dictionary    = dg.dict;
        job->huffman_table = hg.t;
        if (qpl_submit_job(job) != QPL_STS_OK) {
            err = "IAA decomp submit failed at chunk " + std::to_string(ci);
            return false;
        }
        return true;
    };
    auto do_wait = [&](std::size_t ci) -> bool {
        qpl_job* job = pool.get(ci % pool_sz);
        const uint32_t expected = comp.original_chunk_sizes[ci];
        if (qpl_wait_job(job) != QPL_STS_OK ||
            static_cast<uint32_t>(job->total_out) != expected) {
            err = "IAA decomp wait failed at chunk " + std::to_string(ci);
            return false;
        }
        return true;
    };

    std::size_t next_sub = 0, next_wait = 0;
    for (; next_sub < std::min(pool_sz, comp.chunks.size()) && ok; ++next_sub)
        ok = do_submit(next_sub);
    while (next_wait < comp.chunks.size() && ok) {
        const std::size_t wn = std::min(kWaitBatch, comp.chunks.size() - next_wait);
        for (std::size_t k = 0; k < wn && ok; ++k) ok = do_wait(next_wait + k);
        next_wait += wn;
        const std::size_t sn = std::min(wn, comp.chunks.size() - next_sub);
        for (std::size_t k = 0; k < sn && ok; ++k) ok = do_submit(next_sub + k);
        next_sub += sn;
    }
    return ok;
}

// ── Result row ────────────────────────────────────────────────────────────────

struct ResultRow {
    std::string dataset;
    std::string data_type;       // "fid" | "tb"
    std::string algorithm;       // "LZ4" | "Deflate" | "IAA"
    std::size_t chunk_size_kb = 0;
    std::size_t n_elements    = 0;
    std::size_t original_bytes    = 0;
    std::size_t compressed_bytes  = 0;
    double      compression_ratio = 0.0;
    uint64_t    compression_cycles = 0;
    bool        verified = false;
};

// ── Algo enum ─────────────────────────────────────────────────────────────────

enum class Algo { LZ4, Deflate, IAA };

static const std::vector<std::pair<std::string, Algo>> kAlgos = {
    {"LZ4",    Algo::LZ4},
    {"Deflate", Algo::Deflate},
    {"IAA",    Algo::IAA},
};

// ── Single-buffer benchmark ───────────────────────────────────────────────────

ResultRow run_benchmark(const std::string& dataset,
                        const std::string& data_type,
                        const std::vector<uint8_t>& raw,
                        std::size_t chunk_size,
                        std::size_t n_elements,
                        Algo algo) {
    const RawBlocks blocks = split_into_chunks(raw, chunk_size);

    ResultRow row;
    row.dataset        = dataset;
    row.data_type      = data_type;
    row.chunk_size_kb  = chunk_size / kOneKB;
    row.n_elements     = n_elements;
    row.original_bytes = raw.size();

    uint64_t cycles = 0;
    ChunkedCompressed comp;

    switch (algo) {
    case Algo::LZ4:
        row.algorithm = "LZ4";
        comp = compress_lz4(blocks, cycles);
        break;
    case Algo::Deflate:
        row.algorithm = "Deflate";
        comp = compress_deflate(blocks, cycles);
        break;
    case Algo::IAA:
        row.algorithm = "IAA";
        comp = compress_iaa(blocks, cycles);
        break;
    }

    row.compressed_bytes   = comp.compressed_bytes;
    row.compression_ratio  = static_cast<double>(comp.compressed_bytes) /
                             static_cast<double>(raw.size());
    row.compression_cycles = cycles;

    std::vector<uint8_t> decompressed;
    std::string err;
    bool ok = false;
    switch (algo) {
    case Algo::LZ4:    ok = decompress_lz4(comp, decompressed, err);    break;
    case Algo::Deflate: ok = decompress_deflate(comp, decompressed, err); break;
    case Algo::IAA:    ok = decompress_iaa(comp, decompressed, err);    break;
    }
    if (!ok) {
        std::cerr << "[WARN] " << dataset << "/" << data_type << "/"
                  << row.algorithm << " decompression failed: " << err << "\n";
    } else {
        row.verified = (decompressed == raw);
        if (!row.verified)
            std::cerr << "[WARN] " << dataset << "/" << data_type << "/"
                      << row.algorithm << " round-trip mismatch!\n";
    }
    return row;
}

// ── Process one dataset ───────────────────────────────────────────────────────

std::vector<ResultRow> process_dataset(const DatasetConfig& cfg) {
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "Dataset: " << cfg.name << "  dir: " << cfg.dir << "\n";

    // Read manifest
    const std::string manifest_path = cfg.dir + "manifest.json";
    std::ifstream mf(manifest_path);
    if (!mf.is_open())
        throw std::runtime_error("Cannot open manifest: " + manifest_path);
    json manifest;
    mf >> manifest;

    const std::size_t N = manifest["n_elements"].get<std::size_t>();
    std::cout << "  n_elements = " << N << "\n";

    // Per-algo FID accumulators
    struct FidAccum {
        std::size_t orig_bytes  = 0;
        std::size_t comp_bytes  = 0;
        uint64_t    cycles      = 0;
        bool        verified    = true;
        std::size_t chunk_kb    = 0;
    };
    std::map<std::string, FidAccum> fid_acc;
    for (const auto& [aname, _] : kAlgos) fid_acc[aname] = {};

    // Combined (concatenated) TB buffer
    std::vector<uint8_t> combined_tb;

    // Pre-load payload rows for datasets that have payloads.jsonl.
    // Used for: (a) LAION numeric FID replacement, (b) metadata size computation.
    std::vector<json> payload_rows;
    if (cfg.name == "laion" || cfg.name == "hnm") {
        const std::string payload_path = resolve_payload_path(manifest, cfg.name);
        if (fs::exists(payload_path)) {
            std::cout << "  Loading payloads (" << N << " rows) from: "
                      << payload_path << "\n";
            payload_rows = load_jsonl_rows(payload_path, N);
            std::cout << "  Loaded " << payload_rows.size() << " rows.\n";
        } else {
            std::cerr << "[WARN] No payload file found for " << cfg.name
                      << "; metadata size will be estimated as N*4.\n";
        }
    }

    for (const auto& attr_json : manifest["attributes"]) {
        const std::string fid_file = attr_json["fid_file"].get<std::string>();
        const std::string tb_file  = attr_json["tb_file"].get<std::string>();
        const std::string key      = attr_json["key"].get<std::string>();

        std::cout << "  [attr] " << key << "\n";

        // ── FID ──────────────────────────────────────────────────────────────
        const auto fid = read_binary_file(cfg.dir + fid_file, N);
        const std::size_t used_bits = fid_used_bits(fid);

        // For LAION numeric attrs (used_bits >= 255): compress raw float32
        // payload values instead of quantized bucket IDs.
        std::vector<uint8_t> fid_to_compress;
        if (cfg.name == "laion" && used_bits >= 255 && !payload_rows.empty()) {
            fid_to_compress = attr_to_float32_bytes(payload_rows, key);
            std::cout << "    FID (payload float32): " << fid_to_compress.size()
                      << " B  used_bits=" << used_bits << "\n";
        } else {
            fid_to_compress = fid;
            std::cout << "    FID (quantized):       " << fid_to_compress.size()
                      << " B  used_bits=" << used_bits << "\n";
        }

        const std::size_t cs_fid = next_chunk_size(fid_to_compress.size());
        std::cout << "    chunk=" << cs_fid / kOneKB << " KB\n";

        for (const auto& [aname, algo] : kAlgos) {
            std::cout << "    FID " << aname << " ...\r" << std::flush;
            const auto row =
                run_benchmark(cfg.name, "fid", fid_to_compress, cs_fid, N, algo);
            auto& acc = fid_acc[aname];
            if (acc.chunk_kb == 0) acc.chunk_kb = cs_fid / kOneKB;
            acc.orig_bytes  += row.original_bytes;
            acc.comp_bytes  += row.compressed_bytes;
            acc.cycles      += row.compression_cycles;
            acc.verified    &= row.verified;
        }

        // ── TB ───────────────────────────────────────────────────────────────
        const auto tb_raw = read_binary_file(cfg.dir + tb_file, N * kTbBytesPerNode);
        std::cout << "    Reordering TB (" << N * kTbBytesPerNode / kOneMB
                  << " MB → " << used_bits << " buckets)...\n";
        auto tb_partial = reorder_tb_partial(tb_raw, N, used_bits);
        std::cout << "    TB partial: " << tb_partial.size() << " B\n";

        combined_tb.insert(combined_tb.end(),
                           tb_partial.begin(), tb_partial.end());
    }

    const std::size_t cs_tb = next_chunk_size(combined_tb.size());
    std::cout << "  Combined TB: " << combined_tb.size() << " B ("
              << static_cast<double>(combined_tb.size()) / kOneMB << " MB)"
              << "  chunk=" << cs_tb / kOneKB << " KB\n";

    // ── Compress combined TB ──────────────────────────────────────────────────
    std::vector<ResultRow> rows;

    for (const auto& [aname, algo] : kAlgos) {
        std::cout << "  TB " << aname << " ...\n";
        rows.push_back(run_benchmark(cfg.name, "tb", combined_tb, cs_tb, N, algo));
    }

    // ── Assemble FID rows from accumulators ───────────────────────────────────
    for (const auto& [aname, algo] : kAlgos) {
        const auto& acc = fid_acc[aname];
        ResultRow r;
        r.dataset           = cfg.name;
        r.data_type         = "fid";
        r.algorithm         = aname;
        r.chunk_size_kb     = acc.chunk_kb;
        r.n_elements        = N;
        r.original_bytes    = acc.orig_bytes;
        r.compressed_bytes  = acc.comp_bytes;
        r.compression_ratio = acc.orig_bytes > 0
            ? static_cast<double>(acc.comp_bytes) /
              static_cast<double>(acc.orig_bytes)
            : 0.0;
        r.compression_cycles = acc.cycles;
        r.verified           = acc.verified;
        rows.push_back(r);
    }

    // ── Metadata size row ─────────────────────────────────────────────────────
    {
        std::size_t meta_bytes;
        if (!payload_rows.empty() && manifest.contains("attributes")) {
            meta_bytes = compute_metadata_bytes(payload_rows, manifest["attributes"]);
        } else {
            // SIFT1M: 1 numeric attr × 4 bytes/element
            meta_bytes = N * 4;
        }
        ResultRow r;
        r.dataset            = cfg.name;
        r.data_type          = "metadata";
        r.algorithm          = "none";
        r.chunk_size_kb      = 0;
        r.n_elements         = N;
        r.original_bytes     = meta_bytes;
        r.compressed_bytes   = meta_bytes;
        r.compression_ratio  = 1.0;
        r.compression_cycles = 0;
        r.verified           = true;
        rows.push_back(r);
        std::cout << "  Metadata bytes: " << meta_bytes << " B ("
                  << static_cast<double>(meta_bytes) / kOneMB << " MB)\n";
    }

    return rows;
}

// ── Output ────────────────────────────────────────────────────────────────────

void print_table(const std::vector<ResultRow>& rows) {
    std::cout << "\n"
              << std::left
              << std::setw(10) << "Dataset"
              << std::setw(8)  << "Type"
              << std::setw(10) << "Algo"
              << std::setw(12) << "Chunk(KB)"
              << std::setw(14) << "Orig(MB)"
              << std::setw(14) << "Comp(MB)"
              << std::setw(10) << "Ratio"
              << std::setw(20) << "Cycles"
              << "Verify\n"
              << std::string(110, '-') << "\n";

    for (const auto& r : rows) {
        std::cout
            << std::left
            << std::setw(10) << r.dataset
            << std::setw(8)  << r.data_type
            << std::setw(10) << r.algorithm
            << std::setw(12) << r.chunk_size_kb
            << std::setw(14) << std::fixed << std::setprecision(3)
                             << (static_cast<double>(r.original_bytes) / kOneMB)
            << std::setw(14) << std::fixed << std::setprecision(3)
                             << (static_cast<double>(r.compressed_bytes) / kOneMB)
            << std::setw(10) << std::fixed << std::setprecision(4)
                             << r.compression_ratio
            << std::setw(20) << r.compression_cycles
            << (r.verified ? "OK" : "FAIL")
            << "\n";
    }
}

void write_csv(const std::vector<ResultRow>& rows, const std::string& path) {
    fs::path p(path);
    fs::create_directories(p.parent_path());
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write CSV: " + path);

    f << "dataset,data_type,algorithm,chunk_size_kb,n_elements,"
         "original_bytes,compressed_bytes,compression_ratio,"
         "compression_cycles,verified\n";
    for (const auto& r : rows) {
        f << r.dataset << ","
          << r.data_type << ","
          << r.algorithm << ","
          << r.chunk_size_kb << ","
          << r.n_elements << ","
          << r.original_bytes << ","
          << r.compressed_bytes << ","
          << std::fixed << std::setprecision(6) << r.compression_ratio << ","
          << r.compression_cycles << ","
          << (r.verified ? "true" : "false") << "\n";
    }
    std::cout << "CSV written to: " << path << "\n";
}

} // namespace

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string out_csv = "out/fid_tb_compression.csv";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--out" || arg == "-o") && i + 1 < argc)
            out_csv = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--out <csv>]\n";
            return 0;
        }
    }

    std::vector<ResultRow> all_rows;

    for (const auto& cfg : kDatasets) {
        try {
            auto rows = process_dataset(cfg);
            for (auto& r : rows)
                all_rows.push_back(std::move(r));
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Dataset '" << cfg.name << "': " << e.what() << "\n";
        }
    }

    print_table(all_rows);
    write_csv(all_rows, out_csv);
    return 0;
}
