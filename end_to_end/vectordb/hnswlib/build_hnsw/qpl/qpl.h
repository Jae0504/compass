#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t qpl_status;
typedef int32_t qpl_path_t;
typedef int32_t qpl_operation;
typedef int32_t qpl_out_bit_width;
typedef int32_t qpl_compression_levels;
typedef int32_t qpl_huffman_table_type_e;
typedef void* qpl_huffman_table_t;

typedef struct {
    uint32_t dummy;
} qpl_histogram;

typedef struct qpl_job {
    qpl_operation op;
    qpl_compression_levels level;
    uint8_t* next_in_ptr;
    uint8_t* next_out_ptr;
    uint32_t available_in;
    uint32_t available_out;
    uint32_t src1_bit_width;
    qpl_out_bit_width out_bit_width;
    uint32_t param_low;
    uint32_t param_high;
    uint32_t num_input_elements;
    uint32_t flags;
    uint32_t total_out;
    uint32_t sum_value;
    qpl_huffman_table_t huffman_table;
} qpl_job;

static const qpl_status QPL_STS_OK = 0;

static const qpl_path_t qpl_path_hardware = 0;

static const qpl_operation qpl_op_compress = 1;
static const qpl_operation qpl_op_scan_eq = 2;
static const qpl_operation qpl_op_extract = 3;
static const qpl_operation qpl_op_scan_range = 4;

static const qpl_out_bit_width qpl_ow_nom = 0;

static const qpl_compression_levels qpl_default_level = 1;

static const uint32_t QPL_FLAG_FIRST = 1u << 0;
static const uint32_t QPL_FLAG_LAST = 1u << 1;
static const uint32_t QPL_FLAG_OMIT_VERIFY = 1u << 2;
static const uint32_t QPL_FLAG_DECOMPRESS_ENABLE = 1u << 3;

static const qpl_huffman_table_type_e compression_table_type = 0;
static const void* DEFAULT_ALLOCATOR_C = nullptr;

static inline qpl_status qpl_get_job_size(qpl_path_t, uint32_t* size) {
    if (size) {
        *size = sizeof(qpl_job);
    }
    return QPL_STS_OK;
}

static inline qpl_status qpl_init_job(qpl_path_t, qpl_job* job) {
    if (job) {
        job->total_out = 0;
        job->sum_value = 0;
    }
    return QPL_STS_OK;
}

static inline qpl_status qpl_fini_job(qpl_job*) {
    return QPL_STS_OK;
}

static inline qpl_status qpl_submit_job(qpl_job* job) {
    if (job) {
        job->total_out = job->available_in;
    }
    return QPL_STS_OK;
}

static inline qpl_status qpl_wait_job(qpl_job*) {
    return QPL_STS_OK;
}

static inline qpl_status qpl_check_job(qpl_job*) {
    return QPL_STS_OK;
}

static inline qpl_status qpl_deflate_huffman_table_create(
    qpl_huffman_table_type_e,
    qpl_path_t,
    const void*,
    qpl_huffman_table_t* table) {
    if (table) {
        *table = nullptr;
    }
    return QPL_STS_OK;
}

static inline qpl_status qpl_huffman_table_destroy(qpl_huffman_table_t) {
    return QPL_STS_OK;
}

static inline qpl_status qpl_gather_deflate_statistics(
    const uint8_t*,
    uint32_t,
    qpl_histogram*,
    qpl_compression_levels,
    qpl_path_t) {
    return QPL_STS_OK;
}

static inline qpl_status qpl_huffman_table_init_with_histogram(
    qpl_huffman_table_t,
    const qpl_histogram*) {
    return QPL_STS_OK;
}

#ifdef __cplusplus
}
#endif
