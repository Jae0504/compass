#pragma once
#include <cstdint>
#include <vector>

// Opaque handle for an IAA hardware decompression job.
// Create once, reuse across many decompress_block calls.
struct IaaDecompHandle;

// Returns nullptr if IAA hardware is unavailable.
IaaDecompHandle* iaa_decomp_create();
void             iaa_decomp_destroy(IaaDecompHandle*);

// Decompress one IAA-compressed deflate block synchronously.
// Returns elapsed nanoseconds (submit + wait).
// Returns 0 and leaves out_buf untouched if handle is null.
uint64_t iaa_decomp_block(IaaDecompHandle* h,
                           const uint8_t* in,  uint32_t in_sz,
                           uint8_t*       out, uint32_t out_sz);

// Compress each block in data_ptrs/data_sizes with IAA hardware deflate.
// Fills out_blocks; returns false if IAA is unavailable or any block fails.
bool iaa_compress_blocks(const std::vector<const uint8_t*>& data_ptrs,
                          const std::vector<uint32_t>&       data_sizes,
                          std::vector<std::vector<uint8_t>>& out_blocks);
