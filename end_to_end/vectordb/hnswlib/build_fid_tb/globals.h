#ifndef GLOBALS_H
#define GLOBALS_H

#include <vector>
#include <string>
#include <bitset>
#include <chrono>
#include <unordered_set>

// Declare global variables
extern std::vector<std::bitset<256>> connector_bits;
extern std::vector<std::uint8_t> filter_ids;

extern std::vector<std::uint8_t> reordered_connector_bits;
extern std::vector<std::vector<std::uint8_t>> parsed_connector_bits;
extern std::vector<std::vector<std::uint8_t>> parsed_filter_ids;
extern std::vector<std::vector<std::uint8_t>> parsed_connector_bits_m1;
extern std::vector<std::vector<std::uint8_t>> parsed_connector_bits_m2;
extern std::vector<std::vector<std::uint8_t>> parsed_filter_ids_m1;
extern std::vector<std::vector<std::uint8_t>> parsed_filter_ids_m2;
extern bool and_flag;

extern bool ena_algorithm;
extern bool ena_cnt_distcal;

extern int nElements;
extern int nFilters;
extern int nThreads;

extern int cnt_distcal_lv0; // Number of query calculations at level 0
extern int cnt_distcal_upper; // Number of query calculations at upper level(level 1-)
// extern int target_filter_id;
extern std::vector<int> target_filter_ids;
extern std::vector<int> target_filter_ids_m1;
extern std::vector<int> target_filter_ids_m2;

extern std::ofstream RunResultFile;
extern std::string dataset_type;

typedef unsigned int tableint;

extern int isolated_connection_factor;
extern int steiner_factor;
extern int ep_factor;
extern float break_factor;

extern unsigned long long iaa_cpu_cycles;
extern std::chrono::high_resolution_clock::time_point top_layer_end;
extern std::vector<std::vector<uint8_t>> compressed_filter_ids;
extern std::vector<std::vector<uint8_t>> compressed_filter_ids_m1;
extern std::vector<std::vector<uint8_t>> compressed_filter_ids_m2;
extern std::vector<int> ids_list;
extern std::vector<float> ids_dist_list;
extern std::size_t group_chunk_size;
extern std::unordered_set<int> unique_ids_list;
extern std::vector<int> group_bit_visited_list;
extern std::vector<int> group_bit_decompression_plan;
extern std::vector<uint8_t> new_connector_bits;

#endif // GLOBALS_H
