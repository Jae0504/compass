// iaa_decomp.cpp — compiled separately so it can include the REAL <qpl/qpl.h>
// without conflicting with the stub qpl/qpl.h pulled in by hnswlib headers.
#include "iaa_decomp.h"

#include <qpl/qpl.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::nanoseconds;

static inline uint64_t elapsed_ns(const Clock::time_point& t0,
                                   const Clock::time_point& t1) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Ns>(t1 - t0).count());
}

// ── IaaDecompHandle ──────────────────────────────────────────────────────────

struct IaaDecompHandle {
    std::vector<uint8_t> buf;
    qpl_job*             job = nullptr;
};

IaaDecompHandle* iaa_decomp_create() {
    uint32_t job_size = 0;
    if (qpl_get_job_size(qpl_path_hardware, &job_size) != QPL_STS_OK)
        return nullptr;

    auto* h = new IaaDecompHandle();
    h->buf.resize(job_size, 0);
    h->job = reinterpret_cast<qpl_job*>(h->buf.data());
    if (qpl_init_job(qpl_path_hardware, h->job) != QPL_STS_OK) {
        delete h;
        return nullptr;
    }
    return h;
}

void iaa_decomp_destroy(IaaDecompHandle* h) {
    if (!h) return;
    if (h->job) qpl_fini_job(h->job);
    delete h;
}

uint64_t iaa_decomp_block(IaaDecompHandle* h,
                            const uint8_t* in,  uint32_t in_sz,
                            uint8_t*       out, uint32_t out_sz) {
    if (!h || !h->job) return 0;

    qpl_job* job    = h->job;
    job->op         = qpl_op_decompress;
    job->next_in_ptr   = const_cast<uint8_t*>(in);
    job->available_in  = in_sz;
    job->next_out_ptr  = out;
    job->available_out = out_sz;
    job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST;
    job->huffman_table = nullptr;
    job->dictionary    = nullptr;

    const auto t0 = Clock::now();
    qpl_status st = qpl_submit_job(job);
    if (st != QPL_STS_OK)
        throw std::runtime_error("IAA decomp submit failed st=" + std::to_string(st));
    st = qpl_wait_job(job);
    const auto t1 = Clock::now();

    if (st != QPL_STS_OK)
        throw std::runtime_error("IAA decomp wait failed st=" + std::to_string(st));
    return elapsed_ns(t0, t1);
}

// ── IAA compression (used during setup) ─────────────────────────────────────

bool iaa_compress_blocks(const std::vector<const uint8_t*>& data_ptrs,
                          const std::vector<uint32_t>&       data_sizes,
                          std::vector<std::vector<uint8_t>>& out_blocks) {
    if (data_ptrs.size() != data_sizes.size()) return false;
    const size_t n = data_ptrs.size();
    out_blocks.resize(n);

    uint32_t job_size = 0;
    if (qpl_get_job_size(qpl_path_hardware, &job_size) != QPL_STS_OK)
        return false;

    std::vector<uint8_t> jbuf(job_size, 0);
    qpl_job* job = reinterpret_cast<qpl_job*>(jbuf.data());
    if (qpl_init_job(qpl_path_hardware, job) != QPL_STS_OK)
        return false;

    bool ok = true;
    for (size_t i = 0; i < n && ok; ++i) {
        const uint32_t raw_sz = data_sizes[i];
        const uint32_t cap    = raw_sz + raw_sz / 8 + 128;
        out_blocks[i].resize(cap);

        job->op            = qpl_op_compress;
        job->level         = qpl_default_level;
        job->next_in_ptr   = const_cast<uint8_t*>(data_ptrs[i]);
        job->available_in  = raw_sz;
        job->next_out_ptr  = out_blocks[i].data();
        job->available_out = cap;
        job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST |
                             QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;
        job->huffman_table = nullptr;
        job->dictionary    = nullptr;

        qpl_status st = qpl_submit_job(job);
        if (st == QPL_STS_OK) st = qpl_wait_job(job);
        if (st == QPL_STS_OK) {
            out_blocks[i].resize(job->total_out);
        } else {
            ok = false;
        }
    }

    qpl_fini_job(job);
    return ok;
}
