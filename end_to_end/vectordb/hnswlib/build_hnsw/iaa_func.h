#include <vector>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <memory>
#include <algorithm>
#include <bitset>
#include <thread>
#include <x86intrin.h> 
#include <atomic>

#include "globals.h"
#include "qpl/qpl.h"

std::chrono::high_resolution_clock::time_point top_layer_end;

int iaa_initialization(std::vector<qpl_job *>& job, std::vector<std::unique_ptr<uint8_t[]>>& job_buffer, uint32_t queue_size) {
    // std::cout << "Job Intialization" << std::endl;
    qpl_status                              status;
    uint32_t                                size = 0;
    qpl_path_t execution_path = qpl_path_hardware;

    job_buffer.resize(queue_size);
    job.resize(queue_size);
    status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job size getting." << std::endl;
        return 1;
    }

    for (int i = 0; i < queue_size; ++i) {
        job_buffer[i] = std::make_unique<uint8_t[]>(size);
        job[i] = reinterpret_cast<qpl_job *>(job_buffer[i].get());
        status = qpl_init_job(execution_path, job[i]);
        if (status != QPL_STS_OK) {
            std::cout << "An error " << status << " acquired during job initializing." << std::endl;
            return 1;
        }
    }
    return 0;
}

int iaa_free(std::vector<qpl_job *>& job, uint32_t queue_size) {
    // Freeing resources
    qpl_status                              status;
    for (int i = 0; i < queue_size; ++i) {
        status = qpl_fini_job(job[i]);
        if (status != QPL_STS_OK) {
            std::cout << "An error " << status << " acquired during job finalization." << std::endl;
            return 1;
        }
    }
    return 0;
}

int iaa_wait_jobs(std::vector<qpl_job *>* job, int job_idx) {
    int start_idx = 64 < job_idx ? 64 : 0;
    for (int i = start_idx; i < job_idx; i++) {
        qpl_status status = qpl_wait_job((*job)[i]);
        if (status != QPL_STS_OK) {
            std::cout << "Error during " << i << "th job waiting: " << status << std::endl;
            return 1;
        }
    }
    return 0;
}

int iaa_wait_connector_jobs(std::vector<qpl_job *>* job, int job_idx, int* num) {
    int start_idx = 64 < job_idx ? 64 : 0;
    int sum = 0;
    for (int i = start_idx; i < job_idx; i++) {
        qpl_status status = qpl_wait_job((*job)[i]);
        if (status != QPL_STS_OK) {
            return 1;
        }
        sum += (*job)[i]->total_out;
        *num += (*job)[i]->sum_value;
    }
    return 0;
}

int iaa_refresh_jobs(std::vector<qpl_job *>* job, int job_idx) {
    int start_idx = 128 - job_idx;
    int end_idx = start_idx + 64;
    for (int i = start_idx; i < end_idx; i++) {
        qpl_status status = qpl_wait_job((*job)[i]);
        if (status != QPL_STS_OK) {
            std::cout << "Error during " << i << "th job waiting: " << status << std::endl;
            return 1;
        }
    }
    return 0;
}

int iaa_compression(std::size_t chunk_size, std::vector<uint8_t>& src_vector, std::vector<std::vector<uint8_t>>& compressed_vector)
{
    std::cout << "[IAA Compression]" << std::endl;
    iaa_cpu_cycles = 0;

    // Job initialization
    std::vector<std::unique_ptr<uint8_t[]>> job_buffer;
    std::vector<qpl_job *>                  job;
    qpl_status                              status;
    uint32_t                                size = 0;
    qpl_path_t execution_path = qpl_path_hardware;
    uint32_t queue_size = 128;
    std::vector<std::vector<uint8_t>> dest_vector;
    dest_vector.resize(queue_size);
    job_buffer.resize(queue_size);
    job.resize(queue_size);

    status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job size getting." << std::endl;
        return 1;
    }
    for (int i = 0; i < queue_size; ++i) {
        job_buffer[i] = std::make_unique<uint8_t[]>(size);
        job[i] = reinterpret_cast<qpl_job *>(job_buffer[i].get());
        status = qpl_init_job(execution_path, job[i]);
        if (status != QPL_STS_OK) {
            std::cout << "An error " << status << " acquired during job initializing." << std::endl;
            return 1;
        }
    }

    std::size_t src_file_left = src_vector.size();
    std::size_t compressed_size = 0;
    std::size_t vector_size = 0;
    std::size_t iteration = 0;
    std::size_t current_idx = 0;
    std::size_t max_chunk_size = 2*1024*1024;

    qpl_huffman_table_t huffman_table = nullptr;
    status = qpl_deflate_huffman_table_create(compression_table_type, execution_path, DEFAULT_ALLOCATOR_C, &huffman_table);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during Huffman table creation.\n";
        return 1;
    }
    qpl_histogram              deflate_histogram {};
    unsigned long long t0 = 0, t1 = 0;
    // Compression
    while(src_file_left > 0) {
        int enqueue_cnt = 0;
        for (int i = 0; i < queue_size; ++i) {
            // Resizing source and destination vectors
            if (src_file_left <= chunk_size) {
                vector_size = src_file_left;
            } else {
                vector_size = chunk_size;
            }
            if (current_idx % max_chunk_size == 0) {
                status = qpl_gather_deflate_statistics(src_vector.data() + current_idx, std::min(max_chunk_size, src_file_left), &deflate_histogram, qpl_default_level, execution_path);
                if (status != QPL_STS_OK) {
                    std::cout << "An error " << status << " acquired during gathering statistics for Huffman table.\n";
                    qpl_huffman_table_destroy(huffman_table);
                    return 1;
                }
                status = qpl_huffman_table_init_with_histogram(huffman_table, &deflate_histogram);
                if (status != QPL_STS_OK) {
                    std::cout << "An error " << status << " acquired during Huffman table initialization.\n";
                    qpl_huffman_table_destroy(huffman_table);
                    return 1;
                }
            }
            dest_vector[i].resize(max_chunk_size);
            t0 = __rdtsc();
            // Loading data from source file to source vector
            // src_file.read(reinterpret_cast<char *>(&src_vector[i].front()), vector_size);
            // Performing a operation
            job[i]->op             = qpl_op_compress;
            job[i]->level          = qpl_default_level;
            job[i]->next_in_ptr    = src_vector.data() + current_idx; // src_vector[i].data();
            job[i]->next_out_ptr   = dest_vector[i].data();
            job[i]->available_in   = static_cast<uint32_t>(vector_size);
            job[i]->available_out  = static_cast<uint32_t>(max_chunk_size);
            // job[i]->flags          = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_OMIT_VERIFY |QPL_FLAG_DYNAMIC_HUFFMAN;
            job[i]->flags          = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_OMIT_VERIFY;
            job[i]->huffman_table  = huffman_table;
            status = qpl_submit_job(job[i]);
            if (status != QPL_STS_OK) {
                std::cout << "An error " << status << " acquired during job execution." << std::endl;
                return 1;
            }

            current_idx += vector_size;
            src_file_left -= vector_size;
            enqueue_cnt = i + 1;
            t1 = __rdtsc();
            iaa_cpu_cycles += t1 - t0;
            if (src_file_left == 0) { break; }
        }
        t0 = __rdtsc();
        for (int i = 0; i < enqueue_cnt; ++i) {
            if (qpl_check_job(job[i]) == QPL_STS_OK) {
                continue;
            }
            status = qpl_wait_job(job[i]);
            if (status != QPL_STS_OK) {
                std::cout << "An error " << status << " acquired during job waiting." << std::endl;
                return 1;
            }
        }
        t1 = __rdtsc();
        iaa_cpu_cycles += t1 - t0;
        for (int i = 0; i < enqueue_cnt; ++i) {
            compressed_size += static_cast<std::size_t>(job[i]->total_out);
            std::size_t out_size = static_cast<std::size_t>(job[i]->total_out);
            compressed_vector[iteration + i].resize(out_size);
            std::copy_n(dest_vector[i].begin(), out_size, compressed_vector[iteration + i].begin());
        }
        iteration += enqueue_cnt;
    }

    // Freeing resources
    status = qpl_huffman_table_destroy(huffman_table);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during destroying Huffman table.\n";
        return 1;
    }
    for (int i = 0; i < queue_size; ++i) {
        status = qpl_fini_job(job[i]);
        if (status != QPL_STS_OK) {
            std::cout << "An error " << status << " acquired during job finalization." << std::endl;
            return 1;
        }
    }

    std::cout << std::endl;
    std::cout << "Content was successfully compressed." << std::endl;
    std::cout << "Input size       = " << src_vector.size() << " Bytes" << std::endl;
    std::cout << "Output size      = " << compressed_size << " Bytes" << std::endl;
    std::cout << "Ratio            = " << static_cast<double>(compressed_size) / static_cast<double>(src_vector.size()) << std::endl << std::endl;

    return 0;
}

int iaa_decompression_metadata(std::vector<qpl_job *>* job, std::vector<std::vector<uint8_t>>& compressed_metadata, std::vector<std::vector<uint8_t>>& decompressed_metadata, std::size_t chunk_size, std::vector<int> target_filter_ids, int& job_idx, bool group_bit, bool& first_run)
{
    int num_candidates;
    if (group_bit) {
        num_candidates = (nElements + chunk_size -1) / chunk_size;
        int job_offset = job_idx;
        // unsigned long long t0 = __rdtsc();
        for (int target_filter_id_idx = 0; target_filter_id_idx < target_filter_ids.size(); target_filter_id_idx++) {
            int dest_idx = 0;
            for (int candidate_idx = 0; candidate_idx < num_candidates; candidate_idx++) {
                (*job)[job_offset    ]->op                  = qpl_op_scan_eq;
                (*job)[job_offset    ]->next_in_ptr         = compressed_metadata[candidate_idx].data();
                (*job)[job_offset    ]->next_out_ptr        = decompressed_metadata[target_filter_id_idx].data() + dest_idx;
                (*job)[job_offset    ]->available_in        = compressed_metadata[candidate_idx].size();
                (*job)[job_offset    ]->available_out       = std::min(chunk_size, nElements - chunk_size * candidate_idx);
                (*job)[job_offset    ]->src1_bit_width      = 8;
                (*job)[job_offset    ]->out_bit_width       = qpl_ow_nom;
                (*job)[job_offset    ]->param_low           = target_filter_ids[target_filter_id_idx];
                (*job)[job_offset    ]->num_input_elements  = static_cast<uint32_t>(std::min(chunk_size, nElements - chunk_size * candidate_idx));
                (*job)[job_offset    ]->flags               = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
                dest_idx += (std::min(chunk_size, nElements - chunk_size * candidate_idx)) / 8;
                qpl_status status = qpl_submit_job((*job)[job_offset]);
                if (status != QPL_STS_OK) {
                    std::cout << "Error during job " << job_offset << "th submission: " << status << std::endl;
                    return 1;
                }
                job_offset++;
                if (job_offset == 128 || (!first_run && job_offset == 64))
                {
                    iaa_refresh_jobs(job, job_offset);
                    if (job_offset == 128) {
                        job_offset = 0;
                        first_run = false;  
                    }
                    else {
                        job_offset = 64;
                    }
                }
            }
        }
        job_idx = job_offset;
        // unsigned long long t1 = __rdtsc();
        // iaa_cpu_cycles += t1 - t0;
    } else {
        std::vector<int> organized_target_filter_ids;
        for (int target_filter_id : target_filter_ids) {
            if (organized_target_filter_ids.size() == 0) {
                organized_target_filter_ids.push_back(target_filter_id / 8);
            }
            if (organized_target_filter_ids.size() > 0 && organized_target_filter_ids[organized_target_filter_ids.size() - 1] != target_filter_id / 8) {
                organized_target_filter_ids.push_back(target_filter_id / 8);
            }
        }

        int job_offset = job_idx;
        for (int target_filter_id_idx = 0; target_filter_id_idx < organized_target_filter_ids.size(); target_filter_id_idx++) {            
            int start_range = organized_target_filter_ids[target_filter_id_idx] * nElements;
            int end_range = start_range + nElements;

            int start_chunk_idx = start_range / chunk_size;
            int end_chunk_idx = (end_range + chunk_size - 1) / chunk_size;
            // unsigned long long t0 = __rdtsc();
            int dest_idx = 0;
            for (int candidate_idx = start_chunk_idx; candidate_idx < end_chunk_idx; candidate_idx++) {
                int chunk_start = candidate_idx * chunk_size;
                int chunk_end = chunk_start + chunk_size - 1;
                uint32_t low_range = std::max(start_range, chunk_start) - chunk_start;
                uint32_t high_range = std::min(end_range - 1, chunk_end) - chunk_start;
                (*job)[job_offset    ]->op                  = qpl_op_extract;
                (*job)[job_offset    ]->next_in_ptr         = compressed_metadata[candidate_idx].data();
                (*job)[job_offset    ]->next_out_ptr        = decompressed_metadata[target_filter_id_idx].data() + dest_idx;
                (*job)[job_offset    ]->available_in        = compressed_metadata[candidate_idx].size();
                (*job)[job_offset    ]->available_out       = high_range - low_range + 1;
                (*job)[job_offset    ]->src1_bit_width      = 8;
                (*job)[job_offset    ]->out_bit_width       = qpl_ow_nom;
                (*job)[job_offset    ]->param_low           = low_range;
                (*job)[job_offset    ]->param_high          = high_range;
                (*job)[job_offset    ]->num_input_elements  = std::min(chunk_size, 32 * nElements - chunk_size * candidate_idx);
                (*job)[job_offset    ]->flags               = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;
                dest_idx += high_range - low_range + 1;
                qpl_status status = qpl_submit_job((*job)[job_offset]);
                if (status != QPL_STS_OK) {
                    std::cout << "Error during job " << job_offset << "th submission: " << status << std::endl;
                    return 1;
                }
                job_offset++;
                if (job_offset == 128 || (!first_run && job_offset == 64))
                {
                    iaa_refresh_jobs(job, job_offset);
                    if (job_offset == 128) {
                        job_offset = 0;
                        first_run = false;  
                    }
                    else {
                        job_offset = 64;
                    }
                }
            }
            // unsigned long long t1 = __rdtsc();
            // iaa_cpu_cycles += t1 - t0;
        }
        job_idx = job_offset;
    }
    return 0;
}

// int iaa_group_decompression(std::vector<qpl_job *>* job, std::vector<std::vector<uint8_t>>& compressed_metadata, std::vector<std::vector<uint8_t>>& decompressed_metadata, std::size_t chunk_size, std::vector<int> target_filter_ids, int& job_idx, bool& first_run, std::vector<int>& group_bit_decompression_plan, int num_iaa_decompression)
// {
//     int num_candidates;
//     // num_candidates = (nElements + chunk_size -1) / chunk_size;
//     int job_offset = job_idx;
//     int dest_idx = 0;
//     for (int target_filter_id_idx = 0; target_filter_id_idx < target_filter_ids.size(); target_filter_id_idx++) {
//         for (int unique_id_idx = 0 ; unique_id_idx < num_iaa_decompression; unique_id_idx++) {            
//             int candidate_idx = group_bit_decompression_plan[unique_id_idx];
//             dest_idx = chunk_size * candidate_idx / 8;
//             (*job)[job_offset    ]->op                  = qpl_op_scan_eq;
//             (*job)[job_offset    ]->next_in_ptr         = compressed_metadata[candidate_idx].data();
//             (*job)[job_offset    ]->next_out_ptr        = decompressed_metadata[target_filter_id_idx].data() + dest_idx;
//             (*job)[job_offset    ]->available_in        = compressed_metadata[candidate_idx].size();
//             (*job)[job_offset    ]->available_out       = std::min(chunk_size, nElements - chunk_size * candidate_idx);
//             (*job)[job_offset    ]->src1_bit_width      = 8;
//             (*job)[job_offset    ]->out_bit_width       = qpl_ow_nom;
//             (*job)[job_offset    ]->param_low           = target_filter_ids[target_filter_id_idx];
//             (*job)[job_offset    ]->num_input_elements  = static_cast<uint32_t>(std::min(chunk_size, nElements - chunk_size * candidate_idx));
//             (*job)[job_offset    ]->flags               = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;                                
            
//             qpl_status status = qpl_submit_job((*job)[job_offset]);
//             if (status != QPL_STS_OK) {
//                 std::cout << "Error during job " << job_offset << "th submission: " << status << std::endl;
//                 return 1;
//             }                                
            
//             job_offset++;
//             if (job_offset == 128 || (!first_run && job_offset == 64))
//             {
//                 iaa_refresh_jobs(job, job_offset);
//                 if (job_offset == 128) {
//                     job_offset = 0;
//                     first_run = false;  
//                 }
//                 else {
//                     job_offset = 64;
//                 }
//             }                                
//         }
//     }
//     job_idx = job_offset;
//     return 0;
// }

int iaa_group_decompression_multi_attribute(std::vector<qpl_job *>* job, std::vector<std::vector<uint8_t>>& compressed_metadata, std::vector<std::vector<uint8_t>>& decompressed_metadata, std::size_t chunk_size, std::vector<int> target_filter_ids, int& job_idx, bool& first_run, std::vector<int>& group_bit_decompression_plan, int num_iaa_decompression)
{
    int job_offset = job_idx;

    // for (int target_filter_id_idx = 0; target_filter_id_idx < static_cast<int>(target_filter_ids.size()); ++target_filter_id_idx) {
        const int start_filter_id = target_filter_ids[0];

        for (int unique_id_idx = 0; unique_id_idx < num_iaa_decompression; ++unique_id_idx) {
            // unsigned long long t0 = __rdtsc();
            const int candidate_idx = group_bit_decompression_plan[unique_id_idx];
            const std::size_t candidate_offset_bits = chunk_size * candidate_idx;
            const std::size_t candidate_offset_bytes = candidate_offset_bits / 8;
            const std::size_t remaining_elements = nElements - chunk_size * candidate_idx;
            const std::size_t input_elements = std::min(chunk_size, remaining_elements);
            const std::size_t available_out = input_elements;

            qpl_job* current_job = (*job)[job_offset];

            current_job->op                 = qpl_op_scan_eq;
            current_job->next_in_ptr        = compressed_metadata[candidate_idx].data();
            current_job->next_out_ptr       = decompressed_metadata[0].data() + candidate_offset_bytes;
            current_job->available_in       = compressed_metadata[candidate_idx].size();
            current_job->available_out      = available_out;
            current_job->src1_bit_width     = 8;
            current_job->out_bit_width      = qpl_ow_nom;
            current_job->param_low          = start_filter_id;
            current_job->num_input_elements = static_cast<uint32_t>(input_elements);
            current_job->flags              = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;

            // unsigned long long t1 = __rdtsc();

            qpl_status status = qpl_submit_job(current_job);
            if (status != QPL_STS_OK) {
                std::cout << "Error during job " << job_offset << "th submission: " << status << std::endl;
                return 1;
            }
            // unsigned long long t2 = __rdtsc();
            ++job_offset;

            if (job_offset == 128 || (!first_run && job_offset == 64)) {
                iaa_refresh_jobs(job, job_offset);
                if (job_offset == 128) {
                    job_offset = 0;
                    first_run = false;
                } else {
                    job_offset = 64;
                }
            }
            // unsigned long long t3 = __rdtsc();
            // std::cout << "(Decompress Group) job construciton : " << t1 - t0 <<  std::endl;
            // std::cout << "(Decompress Group) job submission : " << t2 - t1 <<  std::endl;
            // std::cout << "(Decompress Group) job refresh : " << t3 - t2 <<  std::endl;
        }
    // }

    job_idx = job_offset;
    return 0;
}

int iaa_group_decompression(std::vector<qpl_job *>* job, std::vector<std::vector<uint8_t>>& compressed_metadata, std::vector<std::vector<uint8_t>>& decompressed_metadata, std::size_t chunk_size, std::vector<int> target_filter_ids, int& job_idx, bool& first_run, std::vector<int>& group_bit_decompression_plan, int num_iaa_decompression)
{
    int job_offset = job_idx;

    // for (int target_filter_id_idx = 0; target_filter_id_idx < static_cast<int>(target_filter_ids.size()); ++target_filter_id_idx) {
        const int start_filter_id = target_filter_ids[0];
        const int end_filter_id = target_filter_ids[target_filter_ids.size()-1];

        for (int unique_id_idx = 0; unique_id_idx < num_iaa_decompression; ++unique_id_idx) {
            // unsigned long long t0 = __rdtsc();
            const int candidate_idx = group_bit_decompression_plan[unique_id_idx];
            const std::size_t candidate_offset_bits = chunk_size * candidate_idx;
            const std::size_t candidate_offset_bytes = candidate_offset_bits / 8;
            const std::size_t remaining_elements = nElements - chunk_size * candidate_idx;
            const std::size_t input_elements = std::min(chunk_size, remaining_elements);
            const std::size_t available_out = input_elements;

            qpl_job* current_job = (*job)[job_offset];

            // current_job->op                 = qpl_op_scan_eq;
            current_job->op                 = qpl_op_scan_range;
            current_job->next_in_ptr        = compressed_metadata[candidate_idx].data();
            current_job->next_out_ptr       = decompressed_metadata[0].data() + candidate_offset_bytes;
            current_job->available_in       = compressed_metadata[candidate_idx].size();
            current_job->available_out      = available_out;
            current_job->src1_bit_width     = 8;
            current_job->out_bit_width      = qpl_ow_nom;
            current_job->param_low          = start_filter_id;
            current_job->param_high          = end_filter_id;
            current_job->num_input_elements = static_cast<uint32_t>(input_elements);
            current_job->flags              = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_DECOMPRESS_ENABLE;

            // unsigned long long t1 = __rdtsc();

            qpl_status status = qpl_submit_job(current_job);
            if (status != QPL_STS_OK) {
                // std::cout << "Error during job " << job_offset << "th submission: " << status << std::endl;
                return 1;
            }
            // unsigned long long t2 = __rdtsc();
            ++job_offset;

            if (job_offset == 128 || (!first_run && job_offset == 64)) {
                iaa_refresh_jobs(job, job_offset);
                if (job_offset == 128) {
                    job_offset = 0;
                    first_run = false;
                } else {
                    job_offset = 64;
                }
            }
            // unsigned long long t3 = __rdtsc();
            // std::cout << "(Decompress Group) job construciton : " << t1 - t0 <<  std::endl;
            // std::cout << "(Decompress Group) job submission : " << t2 - t1 <<  std::endl;
            // std::cout << "(Decompress Group) job refresh : " << t3 - t2 <<  std::endl;
        }
    // }

    job_idx = job_offset;
    return 0;
}

template<class Function>
inline void ParallelFor_IAA_pooling(
    size_t start, 
    size_t end, 
    size_t numThreads, 
    std::vector<qpl_job *>* job, 
    std::vector<std::vector<uint8_t>> compressed_filter_ids, 
    std::vector<std::vector<uint8_t>> compressed_connector_bits,
    std::size_t chunk_size,
    std::vector<int> target_filter_ids, 
    Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    int job_idx = 0;
    bool first_run = true;
    iaa_cpu_cycles = 0;
    // auto start_time = std::chrono::high_resolution_clock::now();
    // unsigned long long t0 = __rdtsc();
    // if (iaa_decompression_metadata(job, compressed_filter_ids, parsed_filter_ids, group_chunk_size, target_filter_ids, job_idx, true, first_run) == 1) {
    //     std::cout << "Error on decompressing Group bit" << std::endl;
    //     return;
    // }

    // if (iaa_decompression_metadata(job, compressed_connector_bits, parsed_connector_bits, chunk_size, target_filter_ids, job_idx, false, first_run) == 1) {
    //     std::cout << "Error on decompressing Connection bit" << std::endl;
    //     return;
    // }
    // if (iaa_wait_jobs(job, job_idx)) {
    //     std::cout << "Error on waiting jobs" << std::endl;
    //     return; 
    // }
    // unsigned long long t1 = __rdtsc();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    // std::cout << "iaa decompression Time taken: " << duration_ns << " ns" << std::endl;
    // iaa_cpu_cycles += t1 - t0;
    // std::cout << "iaa decompression Cycles: " << iaa_cpu_cycles << std::endl;

    // if (numThreads == 1) {
    //     IAA_wait_job
        for (size_t id = start; id < end; id++) {
            int job_idx = 0;
            bool first_run = true;
            iaa_cpu_cycles = 0;

            // auto single_start_time = std::chrono::high_resolution_clock::now();
            // auto start_time = std::chrono::high_resolution_clock::now();
            // if (iaa_decompression_metadata(job, compressed_filter_ids, parsed_filter_ids, group_chunk_size, target_filter_ids, job_idx, true, first_run) == 1) {
            //     std::cout << "Error on decompressing Group bit" << std::endl;
            //     return;
            // }

            if (iaa_decompression_metadata(job, compressed_connector_bits, parsed_connector_bits, chunk_size, target_filter_ids, job_idx, false, first_run) == 1) {
                std::cout << "Error on decompressing Connection bit" << std::endl;
                return;
            }
            // if (iaa_wait_jobs(job, job_idx)) {
            //     std::cout << "Error on waiting jobs to decompress group/connector bits" << std::endl;
            //     return; 
            // }
            // auto end_time = std::chrono::high_resolution_clock::now();
            // auto iaa_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            // std::cout << "iaa decompression Time taken: " << iaa_duration_ns << " ns" << std::endl;

            // auto top_layer_start = std::chrono::high_resolution_clock::now();
            fn(id, 0, job_idx);
            // auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(top_layer_end - top_layer_start).count();
            // std::cout << "Time taken on top layer: " << duration_ns << " ns" << std::endl;
            // auto single_end_time = std::chrono::high_resolution_clock::now();
            // auto single_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(single_end_time - single_start_time).count();
            // std::cout << "Single query latency: " << single_duration_ns << " ns" << std::endl;
        }
    // } 
    // else {
    //     if (iaa_decompression_metadata(job, compressed_filter_ids, parsed_filter_ids, group_chunk_size, target_filter_ids, job_idx, true, first_run) == 1) {
    //         std::cout << "Error on decompressing Group bit" << std::endl;
    //         return;
    //     }

    //     if (iaa_decompression_metadata(job, compressed_connector_bits, parsed_connector_bits, chunk_size, target_filter_ids, job_idx, false, first_run) == 1) {
    //         std::cout << "Error on decompressing Connection bit" << std::endl;
    //         return;
    //     }
    //     if (iaa_wait_jobs(job, job_idx)) {
    //         std::cout << "Error on waiting jobs" << std::endl;
    //         return; 
    //     }

    //     std::vector<std::thread> threads;
    //     std::atomic<size_t> current(start);

    //     // keep track of exceptions in threads
    //     // https://stackoverflow.com/a/32428427/1713196
    //     std::exception_ptr lastException = nullptr;
    //     std::mutex lastExceptMutex;

    //     for (size_t threadId = 0; threadId < numThreads; ++threadId) {
    //         threads.push_back(std::thread([&, threadId] {
    //             while (true) {
    //                 size_t id = current.fetch_add(1);

    //                 if (id >= end) {
    //                     break;
    //                 }

    //                 try {
    //                     fn(id, threadId, job_idx);
    //                 } catch (...) {
    //                     std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
    //                     lastException = std::current_exception();
    //                     /*
    //                      * This will work even when current is the largest value that
    //                      * size_t can fit, because fetch_add returns the previous value
    //                      * before the increment (what will result in overflow
    //                      * and produce 0 instead of current + 1).
    //                      */
    //                     current = end;
    //                     break;
    //                 }
    //             }
    //         }));
    //     }
    //     // IAA_wait_job

    //     for (auto &thread : threads) {
    //         thread.join();
    //     }
    //     if (lastException) {
    //         std::rethrow_exception(lastException);
    //     }
    // }
}


template<class Function>
inline void ParallelFor_IAA_pooling_multiple_attribute(
    size_t start, 
    size_t end, 
    size_t numThreads, 
    std::vector<qpl_job *>* job, 
    std::vector<std::vector<uint8_t>> compressed_connector_bits_m1, 
    std::vector<std::vector<uint8_t>> compressed_connector_bits_m2,
    std::size_t chunk_size,
    std::vector<int> target_filter_ids_m1, 
    std::vector<int> target_filter_ids_m2, 
    Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    int job_idx = 0;
    bool first_run = true;
    iaa_cpu_cycles = 0;
    for (size_t id = start; id < end; id++) {
        int job_idx = 0;
        bool first_run = true;
        iaa_cpu_cycles = 0;
        if (iaa_decompression_metadata(job, compressed_connector_bits_m1, parsed_connector_bits_m1, chunk_size, target_filter_ids_m1, job_idx, false, first_run) == 1) {
            std::cout << "Error on decompressing Connection bit" << std::endl;
            return;
        }
        if (iaa_decompression_metadata(job, compressed_connector_bits_m2, parsed_connector_bits_m2, chunk_size, target_filter_ids_m2, job_idx, false, first_run) == 1) {
            std::cout << "Error on decompressing Connection bit" << std::endl;
            return;
        }
        fn(id, 0, job_idx);
    }
}

