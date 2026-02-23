#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <lz4hc.h>         // from https://github.com/lz4/lz4
#include <snappy.h>     // from https://github.com/google/snappy
#include <zstd.h>       // from https://github.com/facebook/zstd

// Compress with LZ4 High Compression (HC)
int lz4_compression(std::size_t chunk_size,
                    const std::vector<uint8_t>& original_data,
                    std::vector<std::vector<uint8_t>>& compressed_data,
                    int compression_level=1)
{
    if (chunk_size == 0) return 1;
    if (compression_level < 1 || compression_level > 12) return 1;  // validate

    const uint8_t* src = original_data.data();
    std::size_t N = original_data.size();

    int idx = 0;
    for (std::size_t offset = 0; offset < N; offset += chunk_size) {
        int this_size = (int)std::min(chunk_size, N - offset);
        int maxC = LZ4_compressBound(this_size);
        if (maxC <= 0) return 1;

        std::vector<uint8_t> dst(maxC);
        int cSize = LZ4_compress_HC(
            reinterpret_cast<const char*>(src + offset),
            reinterpret_cast<char*>(dst.data()),
            this_size, maxC,
            compression_level
        );
        if (cSize <= 0) return 1;

        dst.resize(cSize);
        compressed_data[idx] = std::move(dst);
        idx++;
    }

    return 0;
}

// Compress with Snappy
int snappy_compression(std::size_t chunk_size,
                    const std::vector<uint8_t>& original_data,
                    std::vector<std::vector<uint8_t>>& compressed_data)
{
    if (chunk_size == 0) return 1;

    const char* src = reinterpret_cast<const char*>(original_data.data());
    std::size_t N   = original_data.size();

    int idx = 0;
    for (std::size_t offset = 0; offset < N; offset += chunk_size) {
        std::size_t this_size = std::min(chunk_size, N - offset);
        // max compressed size:
        size_t maxC = snappy::MaxCompressedLength(this_size);
        std::vector<uint8_t> dst(maxC);
        size_t cSize = 0;

        snappy::RawCompress(
            src + offset, this_size,
            reinterpret_cast<char*>(dst.data()), &cSize
        );
        if (cSize == 0 || cSize > maxC) return 1;

        dst.resize(cSize);
        compressed_data[idx] = (std::move(dst));
        idx++;
    }
    return 0;
}

// Compress with Zstd
int zstd_compression(std::size_t chunk_size,
                  const std::vector<uint8_t>& original_data,
                  std::vector<std::vector<uint8_t>>& compressed_data)
{
    if (chunk_size == 0) return 1;

    const void* src = original_data.data();
    std::size_t N   = original_data.size();
    const int   level = 1;  // choose compression level 1–22

    int idx = 0;
    for (std::size_t offset = 0; offset < N; offset += chunk_size) {
        std::size_t this_size = std::min(chunk_size, N - offset);
        size_t maxC = ZSTD_compressBound(this_size);
        if (ZSTD_isError(maxC)) return 1;

        std::vector<uint8_t> dst(maxC);
        size_t cSize = ZSTD_compress(
            dst.data(), maxC,
            reinterpret_cast<const uint8_t*>(src) + offset, this_size,
            level
        );
        if (ZSTD_isError(cSize) || cSize == 0) return 1;

        dst.resize(cSize);
        compressed_data[idx] = (std::move(dst));
        idx++;
    }
    return 0;
}

int lz4_decompression(const std::vector<uint8_t>& compressed_data,
                      uint8_t* decompressed_data,
                      std::size_t decompressed_size)
{
    int result = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed_data.data()),
        reinterpret_cast<char*>(decompressed_data),
        static_cast<int>(compressed_data.size()),
        static_cast<int>(decompressed_size)
    );

    return (result < 0) ? 1 : 0;
}

int snappy_decompression(const std::vector<uint8_t>& compressed_data,
                         uint8_t* decompressed_data,
                         std::size_t decompressed_size)
{
    // Snappy doesn't need the output size for RawUncompress, but we provide it for safety/context
    bool ok = snappy::RawUncompress(
        reinterpret_cast<const char*>(compressed_data.data()),
        compressed_data.size(),
        reinterpret_cast<char*>(decompressed_data)
    );
    return ok ? 0 : 1;
}

int zstd_decompression(const std::vector<uint8_t>& compressed_data,
                       uint8_t* decompressed_data,
                       std::size_t decompressed_size)
{
    size_t result = ZSTD_decompress(
        decompressed_data, decompressed_size,
        compressed_data.data(), compressed_data.size()
    );

    return ZSTD_isError(result) ? 1 : 0;
}

int generate_bitmap(const std::vector<uint8_t>& data,
                    int scan_value,
                    std::vector<uint8_t>& bitmap)
{
    if (scan_value < 0 || scan_value > 255) return 1;
    if (bitmap.size() < (data.size() + 7) / 8) return 1;  // ensure bitmap is big enough

    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] == static_cast<uint8_t>(scan_value)) {
            bitmap[i / 8] |= (1 << (i % 8));
        }
    }

    return 0;
}


int extract_data(const std::vector<uint8_t>& data,
                 int start_range, int end_range,
                 std::vector<uint8_t>& extracted_data)
{
    if (start_range < 0 || end_range < 0 ||
        start_range > end_range ||
        static_cast<std::size_t>(end_range) >= data.size() ||
        extracted_data.size() != static_cast<std::size_t>(end_range - start_range + 1)) {
        return 1;
    }

    std::copy(data.begin() + start_range,
              data.begin() + end_range + 1,
              extracted_data.begin());

    return 0;
}
