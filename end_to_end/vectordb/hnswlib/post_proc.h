#pragma once

#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
#include "hnswlib.h"
#include <bitset>
#include "globals.h"
#include "parallel_utils.h"
#include "io_utils.h"
#include "json.hpp"

std::vector<size_t> get_neighbor_ids(const hnswlib::HierarchicalNSW<float>* index, size_t point_id) {
    std::vector<size_t> neighbor_ids;
    unsigned int* linklist = index->get_linklist0(point_id);
    int list_size = linklist[0];

    neighbor_ids.reserve(list_size);
    for (int i = 1; i <= list_size; ++i) {
        neighbor_ids.push_back(linklist[i]);
    }

    return neighbor_ids;
}

std::vector<std::pair<size_t, std::vector<size_t>>> leaf_search_mix(hnswlib::HierarchicalNSW<float>* index, const std::vector<float>& group, size_t group_id) {
    size_t group_size = group.size() / 128; 
    std::vector<std::pair<size_t, std::vector<size_t>>> isolated_points_with_neighbors;

    for (size_t i = group_id; i < nElements; i += nFilters) {
        size_t current_label = i;

        unsigned int *linklist = index->get_linklist0(current_label);
        int list_size = linklist[0];

        bool has_same_group_neighbor = false;
        std::vector<size_t> neighbors;
        for (int j = 1; j <= list_size; ++j) {
            size_t neighbor_label = linklist[j];
            neighbors.push_back(neighbor_label);
            if (neighbor_label % nFilters == group_id && neighbor_label != current_label) {
                has_same_group_neighbor = true;
                break;
            }
        }

        if (!has_same_group_neighbor) {
            isolated_points_with_neighbors.push_back({current_label, neighbors});
        }
    }
    std::cout << std::endl << "Total number of isolated points: " << isolated_points_with_neighbors.size() << std::endl;

    return isolated_points_with_neighbors;
}

std::vector<std::pair<size_t, std::vector<size_t>>> leaf_search(hnswlib::HierarchicalNSW<float>* index, size_t target_filter_id, std::vector<std::uint8_t> group_ids) {

    std::vector<std::pair<size_t, std::vector<size_t>>> isolated_points_with_neighbors;
    std::mutex result_mutex; //for parallel processing
    
    // size_t elements_per_group = std::ceil(static_cast<double>(nElements) / nFilters); //group division method should be aligned with the way used for "generateGroupIds_for_nonmix_groups"

    ParallelFor(0, nElements, nThreads, [&](size_t i, size_t /*thread_id*/) {
        if (group_ids[i] != target_filter_id) return;

        unsigned int* linklist = index->get_linklist0(i);
        int list_size = linklist[0];

        bool has_same_group_neighbor = false;
        std::vector<size_t> neighbors;

        for (int j = 1; j <= list_size; ++j) {
            size_t neighbor_label = linklist[j];
            size_t neighbor_gid = group_ids[neighbor_label];
            neighbors.push_back(neighbor_label);

            if (neighbor_gid == target_filter_id && neighbor_label != i) {
                has_same_group_neighbor = true;
                break;
            }
        }

        if (!has_same_group_neighbor) {
            std::lock_guard<std::mutex> lock(result_mutex);
            isolated_points_with_neighbors.emplace_back(i, std::move(neighbors));
        }
    });

    // std::cout << isolated_points_with_neighbors.size() << std::endl;
    return isolated_points_with_neighbors;
}

void validate_leaf_search(
    hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::pair<size_t, std::vector<size_t>>>& result,
    size_t nElements,
    size_t nFilters) {

    size_t elements_per_group = std::ceil(static_cast<double>(nElements) / nFilters);

    int checked = 0;
    for (const auto& [node, neighbors] : result) {
        if (checked++ > 10) break;  // 최대 10개만 확인

        size_t node_gid = node / elements_per_group;
        bool violation_found = false;

        for (size_t n : neighbors) {
            size_t n_gid = n / elements_per_group;
            if (n != node && n_gid == node_gid) {
                std::cerr << "[❌] Node " << node << " claims to be isolated in group "
                          << node_gid << ", but has neighbor " << n
                          << " in same group!\n";
                violation_found = true;
            }
        }

        if (!violation_found) {
            std::cout << "[✅] Node " << node << " correctly has no same-group neighbors.\n";
        }
    }
}

float calculate_distance(hnswlib::HierarchicalNSW<float>* index, size_t id1, size_t id2) {
    // id1과 id2에 해당하는 데이터 포인트를 가져옵니다.
    char* data1 = index->getDataByInternalId(id1);
    char* data2 = index->getDataByInternalId(id2);

    // fstdistfunc_를 사용하여 거리 계산
    return index->fstdistfunc_(data1, data2, index->dist_func_param_);
}

void process_isolated_points(hnswlib::HierarchicalNSW<float>* index, const std::vector<std::pair<size_t, std::vector<size_t>>>& isolated_points_with_neighbors) {
    for (const auto& point_with_neighbors : isolated_points_with_neighbors) {
        size_t current_label = point_with_neighbors.first;
        const auto& neighbors = point_with_neighbors.second;

        std::cout << "Processing isolated point " << current_label << " with neighbors: ";
        for (const auto& neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;

        // 각 이웃에 대해 그 이웃의 이웃을 탐색
        for (const auto& neighbor : neighbors) {
            unsigned int *linklist = index->get_linklist0(neighbor);
            int neighbor_list_size = linklist[0];

            // Neighbor의 이웃을 탐색
            for (int j = 1; j <= neighbor_list_size; ++j) {
                size_t neighbor_of_neighbor = linklist[j];
                
                // Group1에 속하는지 확인
                if (neighbor_of_neighbor < 250000 && neighbor_of_neighbor != current_label) { // Group1의 ID 범위
                    float distance = calculate_distance(index, current_label, neighbor_of_neighbor);
                    std::cout << "  Neighbor of isolated point " << current_label 
                              << " (from neighbor " << neighbor << ") is " 
                              << neighbor_of_neighbor 
                              << " (in Group 1) with distance: " << distance << std::endl;
                }
            }
        }
    }
}

std::vector<std::string> analyze_graph_connections_mix(hnswlib::HierarchicalNSW<float>* index) {
    std::vector<std::string> connections(nElements, std::string(nFilters, '0'));

    for (size_t i = 0; i < nElements; ++i) {
        unsigned int* linklist = index->get_linklist0(i);
        int list_size = linklist[0];

        for (int j = 1; j <= list_size; ++j) {
            size_t neighbor_label = linklist[j];
            size_t group_id = neighbor_label % nFilters; 
            connections[i][group_id] = '1';
        }
    }

    return connections;
}

std::vector<std::bitset<256>> analyze_graph_connections(
    hnswlib::HierarchicalNSW<float>* index, std::vector<std::uint8_t> group_ids) {
    std::vector<std::bitset<256>> connections(nElements); // 모두 0으로 초기화됨

    for (size_t i = 0; i < nElements; ++i) {
        unsigned int* linklist = index->get_linklist0(i);
        int list_size = linklist[0];

        for (int j = 1; j <= list_size; ++j) {
            size_t neighbor_label = linklist[j];
            size_t neighbor_group_id = group_ids[neighbor_label];
            connections[i][neighbor_group_id]=1;  // set 1 at the position of [group_id]
            // size_t group_id = neighbor_label / (nElements / nFilters);
            // if (group_id < nFilters) {
            //     connections[i][group_id]=1;  // set 1 at the position of [group_id]
            // }
        }
    }

    return connections;
}

// float pointer 기반에서 특정 벡터 추출
inline std::vector<float> get_vector_by_id(float* base_data, size_t base_dim, size_t internal_id) {
    std::vector<float> vector(base_dim);
    float* start = base_data + internal_id * base_dim;
    std::copy(start, start + base_dim, vector.begin());
    return vector;
}

// mix_groups인 경우: i % nFilters 방식으로 그룹 부여
inline std::vector<uint8_t> generateGroupIds_for_mix_groups() {
    // if (nFilters > 256) {
    //     throw std::runtime_error("Number of groups cannot exceed 256 when using uint8_t.");
    // }

    std::vector<uint8_t> filter_ids(nElements, 0);

    for (int i = 0; i < nElements; ++i) {
        int group_index = i % nFilters;
        filter_ids[i] = static_cast<uint8_t>(group_index);
    }

    return filter_ids;
}

// nonmix_groups인 경우: 앞에서부터 잘라서 균등하게 그룹 부여
inline std::vector<uint8_t> generateGroupIds_for_nonmix_groups() {
    // if (nFilters > 256) {
    //     throw std::runtime_error("Number of groups cannot exceed 256 when using uint8_t.");
    // }

    int max_elements_per_group = std::ceil(static_cast<double>(nElements) / nFilters);
    std::vector<uint8_t> filter_ids(nElements, 0);

    for (int i = 0; i < nElements; ++i) {
        int group_index = i / max_elements_per_group;

        if (group_index >= nFilters) {
            group_index = nFilters - 1;
        }

        filter_ids[i] = static_cast<uint8_t>(group_index);
    }

    return filter_ids;
}

// laion인 경우: payloads.jsonl을 읽고 그룹 부여
inline std::vector<uint8_t> generateGroupIds_for_laion(std::string payload_file_path, std::string laion_field) {
    std::vector<uint8_t> filter_ids(nElements, 0);

    std::ifstream payload_file(payload_file_path);
    if (!payload_file.is_open()) {
        std::cerr << "Failed to open payload_file." << std::endl;
        return filter_ids;
    }

    std::vector<float> metadata;
    std::string line;
    int line_idx = 0;
    while (std::getline(payload_file, line)) {
        try {
            // Parse the JSON lin
            nlohmann::json jsonLine = nlohmann::json::parse(line);
            float value = jsonLine[laion_field].get<float>();
            metadata.push_back(value);
            // uint8_t group_index = value / 0.002;
            // filter_ids[line_idx] = group_index;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << e.what() << std::endl;
        }
        line_idx ++;
    }
    payload_file.close();

    std::unordered_set<float> unique_set(metadata.begin(), metadata.end());
    if (unique_set.size() > 256) {
        float min_val = 0;
        float max_val = 0;
        if (!metadata.empty()) {
            min_val = *std::min_element(metadata.begin(), metadata.end());
            max_val = *std::max_element(metadata.begin(), metadata.end());
        }
        float divider = ceil(max_val / 256.0);
        for (int i = 0; i < line_idx; i++) {
            filter_ids[i] = static_cast<int>(metadata[i] / divider); 
        }
    } else {
        std::vector<float> unique_values(unique_set.begin(), unique_set.end());
        for (int i = 0; i < line_idx; i++) {
            auto it = std::find(unique_values.begin(), unique_values.end(), metadata[i]);
            int index = std::distance(unique_values.begin(), it);
            filter_ids[i] = index;
        }
    }

    return filter_ids;
}


void update_connections_for_path(std::vector<std::string>& connections, 
                                 const std::vector<size_t>& path, 
                                 size_t leaf_node_group_num) {
    // size_t group_size = nElements / nFilters;
    size_t group_size = std::ceil(static_cast<double>(nElements) / nFilters);
    
    for (size_t node_id : path) {
        // 노드가 유효한 범위 내에 있는지 확인
        if (node_id < nElements) {
            // 해당 노드의 connections 문자열에서 leaf_node_group_num 위치를 '1'로 설정
            connections[node_id][leaf_node_group_num] = '1';
            
            // 노드가 속한 그룹도 '1'로 설정
            size_t node_group = node_id / group_size;
            if (node_group < nFilters) {
                connections[node_id][node_group] = '1';
            }
        }
    }
}


std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> search_leaf_path_1(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::pair<size_t, std::vector<size_t>>>& isolated_points,
    const std::vector<std::string>& connections,
    size_t group_id)
{
    std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> results;
    // size_t group_size = nElements / nFilters;
    size_t group_size = std::ceil(static_cast<double>(nElements) / nFilters);
    size_t start_index = group_id * group_size;
    size_t end_index = start_index + group_size;

    for (const auto& point : isolated_points) {
        size_t isolated_id = point.first;
        std::queue<std::pair<size_t, std::vector<size_t>>> to_visit;
        std::unordered_set<size_t> visited;
        to_visit.push({isolated_id, {isolated_id}});
        visited.insert(isolated_id);

        while (!to_visit.empty()) {
            auto [current, path] = to_visit.front();
            to_visit.pop();

            unsigned int* linklist = index->get_linklist0(current);
            int list_size = linklist[0];

            for (int i = 1; i <= list_size; ++i) {
                size_t neighbor = linklist[i];
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);

                    std::vector<size_t> new_path = path;
                    new_path.push_back(neighbor);

                    if (connections[neighbor][group_id] == '1') {
                        results.push_back({isolated_id, neighbor, new_path.size() - 1, new_path});
                        goto next_isolated_point;
                    }  
                    to_visit.push({neighbor, new_path});
                    
                }
            }
        }
        next_isolated_point:;
    }
    return results;
}




///////////exp2////////
#include <vector>
#include <tuple>
#include <queue>
#include <unordered_set>
#include <set>
#include <algorithm>

std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> search_leaf_path_2(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::pair<size_t, std::vector<size_t>>>& isolated_points,
    const std::vector<std::string>& connections,
    const std::vector<std::string>& group_bits,
    size_t group_id)
{
    std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> results;
    // size_t group_size = nElements / nFilters;
    size_t group_size = std::ceil(static_cast<double>(nElements) / nFilters);

    for (const auto& point : isolated_points) {
        size_t isolated_id = point.first;
        std::queue<std::pair<size_t, std::vector<size_t>>> to_visit;
        std::unordered_set<size_t> visited;
        to_visit.push({isolated_id, {isolated_id}});
        visited.insert(isolated_id);

        int found_paths = 0;
        std::vector<std::set<size_t>> matched_sets; // saving neighbor's neighbor of each path

        while (!to_visit.empty()) {
            auto [current, path] = to_visit.front();
            to_visit.pop();

            unsigned int* linklist = index->get_linklist0(current);
            int list_size = linklist[0];

            for (int i = 1; i <= list_size; ++i) {
                size_t neighbor = linklist[i];
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);

                    std::vector<size_t> new_path = path;
                    new_path.push_back(neighbor);

                    if (connections[neighbor][group_id] == '1') {
                        unsigned int* neighbor_linklist = index->get_linklist0(neighbor);
                        int neighbor_list_size = neighbor_linklist[0];

                        std::set<size_t> matched_neighbor_neighbors_set;
                        std::vector<size_t> matched_neighbor_neighbors_vec;

                        for (int j = 1; j <= neighbor_list_size; ++j) {
                            size_t neighbor_neighbor = neighbor_linklist[j];
                            if (group_bits[neighbor_neighbor][group_id] == '1') {
                                matched_neighbor_neighbors_set.insert(neighbor_neighbor);
                                matched_neighbor_neighbors_vec.push_back(neighbor_neighbor);
                            }
                        }

                        // First path must be included
                        if (found_paths == 0) {
                            results.push_back({isolated_id, neighbor, new_path.size() - 1, new_path});
                            matched_sets.push_back(matched_neighbor_neighbors_set);
                            found_paths++;
                        }
                        // Second path gonna be included if neighbors set is different from first one
                        else if (found_paths == 1) {
                            if (matched_sets[0] != matched_neighbor_neighbors_set) {
                                results.push_back({isolated_id, neighbor, new_path.size() - 1, new_path});
                                matched_sets.push_back(matched_neighbor_neighbors_set);
                                found_paths++;
                            }
                        }
                        if (found_paths == 2) goto next_isolated_point;
                        continue;
                    }
                    to_visit.push({neighbor, new_path});
                }
            }
        }
        next_isolated_point:;
    }
    return results;
}


//////////exp3/////////
std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> search_leaf_path_3(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::pair<size_t, std::vector<size_t>>>& isolated_points,
    const std::vector<std::bitset<256>>& connections,
    const std::vector<uint8_t>& filter_ids,
    size_t group_id)
{
    std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> results;
    std::mutex result_mutex;

    ParallelFor(0, isolated_points.size(), nThreads, [&](size_t idx, size_t /*thread_id*/) {
        const auto& point = isolated_points[idx];
        size_t isolated_id = point.first;

        std::queue<std::pair<size_t, std::vector<size_t>>> to_visit;
        std::unordered_map<size_t, size_t> visited_depth;

        to_visit.push({isolated_id, {isolated_id}}); //BFS search Init
        visited_depth[isolated_id] = 0;

        int found_paths = 0;
        size_t first_path_depth = 0;
        std::vector<std::set<size_t>> matched_sets;

        while (!to_visit.empty()) {
            auto [current, path] = to_visit.front();
            to_visit.pop();

            unsigned int* linklist = index->get_linklist0(current);
            int list_size = linklist[0];

            for (int i = 1; i <= list_size; ++i) {
                size_t neighbor = linklist[i];
                size_t new_depth = path.size();

                if (visited_depth.find(neighbor) == visited_depth.end() || new_depth < visited_depth[neighbor]) {
                    visited_depth[neighbor] = new_depth;
                    std::vector<size_t> new_path = path;
                    new_path.push_back(neighbor);

                    // bitset check
                    if (connections[neighbor][group_id]==1) {
                        unsigned int* neighbor_linklist = index->get_linklist0(neighbor);
                        int neighbor_list_size = neighbor_linklist[0];

                        std::set<size_t> current_matched_set;
                        std::vector<size_t> current_matched_vec;

                        for (int j = 1; j <= neighbor_list_size; ++j) {
                            size_t neighbor_neighbor = neighbor_linklist[j];
                            if (filter_ids[neighbor_neighbor] == group_id) {
                                current_matched_set.insert(neighbor_neighbor);
                                current_matched_vec.push_back(neighbor_neighbor);
                            }
                        }

                        bool push_second_path = false;

                        {
                            std::lock_guard<std::mutex> lock(result_mutex);
                            if (found_paths == 0) {
                                first_path_depth = new_path.size() - 1;
                                results.emplace_back(isolated_id, neighbor, first_path_depth, new_path);
                                matched_sets.push_back(current_matched_set);
                                found_paths++;
                            } else if (found_paths == 1) {
                                if (matched_sets[0] != current_matched_set &&
                                    (new_path.size() - 1) >= first_path_depth + 2) {
                                    results.emplace_back(isolated_id, neighbor, new_path.size() - 1, new_path); //returned information
                                    found_paths++;
                                    return;  // like goto next_isolated_point
                                }
                            }
                        }
                    }

                    to_visit.push({neighbor, new_path});
                }
            }
        }
    }
);

    return results; //Paths information
}

//////////exp4/////////
std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> search_leaf_path_4(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::pair<size_t, std::vector<size_t>>>& isolated_points,
    const std::vector<std::bitset<256>>& connections,
    const std::vector<uint8_t>& filter_ids,
    size_t group_id)
{
    std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> results;
    std::mutex result_mutex;

    int threshold = isolated_connection_factor;
    ParallelFor(0, isolated_points.size(), nThreads, [&](size_t idx, size_t /*thread_id*/) {
        const auto& point = isolated_points[idx];
        size_t isolated_id = point.first;

        std::queue<std::pair<size_t, std::vector<size_t>>> to_visit;
        std::unordered_map<size_t, size_t> visited_depth;

        to_visit.push({isolated_id, {isolated_id}}); //BFS search Init
        visited_depth[isolated_id] = 0;

        int found_paths = 0;
        size_t first_path_depth = 0;
        std::vector<std::set<size_t>> matched_sets;

        while (!to_visit.empty()) {
            auto [current, path] = to_visit.front();
            to_visit.pop();

            unsigned int* linklist = index->get_linklist0(current);
            int list_size = linklist[0];

            for (int i = 1; i <= list_size; ++i) {
                size_t neighbor = linklist[i];
                size_t new_depth = path.size();

                if (visited_depth.find(neighbor) == visited_depth.end()) {
                    visited_depth[neighbor] = new_depth;
                    std::vector<size_t> new_path = path;
                    new_path.push_back(neighbor);

                    // bitset check
                    if (filter_ids[neighbor] == group_id) {
                        unsigned int* neighbor_linklist = index->get_linklist0(neighbor);
                        int neighbor_list_size = neighbor_linklist[0];
                        
                        std::set<size_t> current_matched_set;
                        std::vector<size_t> current_matched_vec;

                        current_matched_set.insert(neighbor);
                        current_matched_vec.push_back(neighbor);

                        bool push_second_path = false;

                        {
                            std::lock_guard<std::mutex> lock(result_mutex);
                            if (found_paths < threshold) {
                                first_path_depth = new_path.size() - 1;
                                results.emplace_back(isolated_id, neighbor, first_path_depth, new_path);
                                matched_sets.push_back(current_matched_set);
                                found_paths++;
                            } else {
                                return;
                            }
                        }
                    }

                    to_visit.push({neighbor, new_path});
                }
            }
        }
    }
);

    return results; //Paths information
}

//////////exp5/////////
std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> search_leaf_path_5(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::pair<size_t, std::vector<size_t>>>& isolated_points,
    const std::vector<std::bitset<256>>& connections,
    const std::vector<uint8_t>& filter_ids,
    size_t group_id)
{
    std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>> results;
    std::mutex result_mutex;
    std::vector<std::atomic<int>> threshold_vector(nElements);
    int threshold = 3;
    int n_connection = 3;

    // initialize threshold vector
    for (size_t i = 0; i < nElements; i++) {
        threshold_vector[i] = threshold;
    }

    ParallelFor(0, isolated_points.size(), nThreads, [&](size_t idx, size_t /*thread_id*/) {
        const auto& point = isolated_points[idx];
        size_t isolated_id = point.first;

        std::queue<std::pair<size_t, std::vector<size_t>>> to_visit;
        std::unordered_map<size_t, size_t> visited_depth;

        to_visit.push({isolated_id, {isolated_id}});
        visited_depth[isolated_id] = 0;

        int found_paths = 0;

        while (!to_visit.empty()) {
            auto [current, path] = to_visit.front();
            to_visit.pop();

            unsigned int* linklist = index->get_linklist0(current);
            int list_size = linklist[0];

            for (int i = 1; i <= list_size; ++i) {
                size_t neighbor = linklist[i];
                size_t new_depth = path.size();

                if (visited_depth.find(neighbor) == visited_depth.end()) {
                    visited_depth[neighbor] = new_depth;
                    std::vector<size_t> new_path = path;
                    new_path.push_back(neighbor);

                    if (filter_ids[neighbor] == group_id) {
                        // check threshold vector atomically
                        int old_value = threshold_vector[neighbor].load();
                        while (old_value > 0) {
                            if (threshold_vector[neighbor].compare_exchange_weak(old_value, old_value - 1)) {
                                // success: threshold was > 0, now decremented
                                {
                                    std::lock_guard<std::mutex> lock(result_mutex);
                                    results.emplace_back(isolated_id, neighbor, new_path.size() - 1, new_path);
                                }
                                found_paths++;
                                break;
                            }
                            // else: threshold_vector changed by another thread, reload old_value and retry
                        }

                        if (found_paths >= n_connection) {
                            return; // found enough connections for this isolated point
                        }
                    }

                    to_visit.push({neighbor, new_path});
                }
            }
        }
    });

    return results;
}

std::vector<std::string> group_bit_generator() {
    std::vector<std::string> group_bits(nElements, std::string(nFilters, '0'));
    // size_t group_size = nElements / nFilters;
    size_t group_size = std::ceil(static_cast<double>(nElements) / nFilters);

    for (size_t i = 0; i < nElements; ++i) {
        size_t group_id = i / group_size;
        if (group_id < nFilters) {
            group_bits[i][group_id] = '1';
        }
    }

    return group_bits;
}


#include <random>
std::vector<std::bitset<256>> connection_bit_generator(
    const std::vector<std::bitset<256>>& connector_bits,
    const std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>>& results,
    size_t group_id,
    float random_rate
) {
    std::vector<std::bitset<256>> updated_bits = connector_bits;

    // 1. path에 포함된 모든 node에 대해 group_id bit를 1로 설정
    for (const auto& [isolated_id, connected_id, hops, path] : results) {
        for (size_t node_id : path) {
            if (node_id < nElements) {
                updated_bits[node_id].set(group_id);
            }
        }
    }

    // // 2. 랜덤하게 일부 노드도 group_id bit를 1로 설정
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0.0, 1.0);

    // for (size_t i = 0; i < updated_bits.size(); ++i) {
    //     if (dis(gen) < random_rate) {
    //         updated_bits[i].set(group_id);
    //     }
    // }

    return updated_bits;
}

std::vector<std::vector<int>> find_num_cluster(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<int>& node_indices,                           // only those with filter_id == group_id
    size_t group_id
) {

    // 2) Visitation flags for *all* nodes in the graph.
    //    We use max_elements_ so we can index directly by node ID.
    std::vector<char> visited_all(nElements, 0);
    std::vector<char> visited_group(nElements, 0);

    std::vector<std::vector<int>> cluster_node_info;

    // 3) For each group-node not yet assigned to a cluster, BFS to collect its cc.
    for (int start : node_indices) {
        if (visited_group[start]) continue;
        visited_all.clear();
        visited_all.resize(nElements, 0);

        std::vector<int> cluster;
        std::queue<int> q;

        // begin BFS from `start`
        visited_all[start] = 1;
        q.push(start);

        while (!q.empty()) {
            int u = q.front(); q.pop();

            // if u belongs to our group, record it
            if (filter_ids[u] == group_id) {
                // std::cout << "u: " << u << std::endl;
                cluster.push_back(u);
                visited_group[u] = 1;
            }

            // if u itself doesn’t have this connector bit, we can’t traverse out of it
            if (!connector_bits[u][group_id] && filter_ids[u] != group_id) {
                continue;
            }
            if (filter_ids[u] == group_id) {
                // std::cout << "After condition, u: " << u << std::endl;
            }
            // otherwise, walk all level-0 neighbors
            int* nbrs = (int*) index->get_linklist0(u);
            int num_nbrs = nbrs[0];
            for (int i = 1; i <= num_nbrs; ++i) {
                int v = nbrs[i];
                // only cross edges where both endpoints have the bit set
                if (!connector_bits[v][group_id]) {
                    continue;
                }
                if (!visited_all[v]) {
                    visited_all[v] = 1;
                    q.push(v);
                }
            }
        }

        // one connected component (cluster) found
        cluster_node_info.push_back(std::move(cluster));
    }

    // std::cout << "Num. of clusters for group " 
    //           << group_id << ": " 
    //           << cluster_node_info.size() 
    //           << std::endl;
    return cluster_node_info;
}

void validate_connector_bits(
    const std::vector<std::bitset<256>>& connector_bits,
    const std::vector<std::tuple<size_t, size_t, size_t, std::vector<size_t>>>& paths,
    size_t group_id
) {
    int error_count = 0;

    for (const auto& [isolated_id, connected_id, depth, path] : paths) {
        for (size_t node_id : path) {
            if (!connector_bits[node_id][group_id]) {
                std::cerr << "[❌] Node " << node_id << " is in path for group "
                          << group_id << " but its connector_bits is not set!\n";
                error_count++;
            }
        }
    }

    if (error_count == 0) {
        std::cout << "[✅] All path nodes have connector_bits correctly set for group "
                  << group_id << ".\n";
    } else {
        std::cout << "[⚠️] Validation failed for " << error_count << " nodes.\n";
    }
}


inline float calculateRecallForGroup(
    const std::vector<int>& filtered_gt,
    const std::vector<hnswlib::labeltype>& knn_result,
    int k
) {
    // If both vectors are empty, return perfect recall
    if (filtered_gt.empty() && knn_result.empty()) {
        return 1.0f;
    }

    size_t compare_count = std::min(static_cast<size_t>(k), filtered_gt.size());

    if (compare_count == 0) {
        return 0.0f;
    }

    int correct = 0;
    for (size_t i = 0; i < compare_count; ++i) {
        for (size_t j = 0; j < compare_count; ++j) {
            if (filtered_gt[i] == knn_result[j]) {
                correct++;
                break;
            }
        }
    }

    return static_cast<float>(correct) / compare_count;
}


// Enter Point Processing
inline std::vector<tableint> getNeighbors(const hnswlib::HierarchicalNSW<float>* index, tableint ep_id) {
    std::vector<tableint> neighbors;
    int *data = (int *) index->get_linklist0(ep_id);
    size_t size = data[0]; // first entry is count

    for (size_t j = 1; j <= size; j++) {
        neighbors.push_back(data[j]);
    }
    return neighbors;
}

inline std::vector<int> findMissingGroups( //only connect to single extra node for each group
    const hnswlib::HierarchicalNSW<float>* index,
    tableint ep_id,
    const std::vector<tableint>& neighbors,
    const std::vector<uint8_t>& filter_ids)
{
    std::vector<bool> group_present(nFilters, false);
    for (auto neighbor : neighbors) {
        int group_id = filter_ids[neighbor];
        if (group_id < nFilters) {
            group_present[group_id] = true;
        }
    }

    std::vector<int> missing_nodes;
    std::mt19937 gen(std::random_device{}());
    int max_elements_per_group = std::ceil(static_cast<double>(nElements) / nFilters);

    for (int g = nFilters-1; g < nFilters; ++g) {
        if (!group_present[g]) {
            std::vector<tableint> candidates;
            for (tableint i = max_elements_per_group * g; i < std::min(max_elements_per_group * (g+1), nElements); ++i) {
                if (filter_ids[i] == g) {
                    candidates.push_back(i);
                }
            }

            if (!candidates.empty()) {

                int num_extra_ep_neighbors = 3;
                std::vector<std::pair<size_t, tableint>> candidate_distances;
                for (auto candidate_id : candidates) {
                    char* ep_data = index->getDataByInternalId(ep_id);
                    char* candidate_data = index->getDataByInternalId(candidate_id);
                    size_t dist = index->fstdistfunc_(ep_data, candidate_data, index->dist_func_param_);
                    candidate_distances.emplace_back(dist, candidate_id);
                }
                std::sort(candidate_distances.begin(), candidate_distances.end());

                // pick top 3 closest
                int num_to_pick = std::min(num_extra_ep_neighbors, (int)candidate_distances.size());
                for (int i = 0; i < num_to_pick; i++) {
                    missing_nodes.push_back(candidate_distances[i].second);
                }

                // std::uniform_int_distribution<> dis(0, candidates.size() - 1);
                // tableint selected = candidates[dis(gen)];

                // // avoiding duplication
                // if (std::find(missing_nodes.begin(), missing_nodes.end(), selected) == missing_nodes.end()) {
                //     missing_nodes.push_back(selected);
                // }
            }
        }
    }

    return missing_nodes;
}

inline std::vector<int> findMissingGroups_laion( //only connect to single extra node for each group
    const hnswlib::HierarchicalNSW<float>* index,
    tableint ep_id,
    const std::vector<tableint>& neighbors,
    const std::vector<uint8_t>& filter_ids,
    std::vector<std::vector<int>> pre_candidates)
{
    std::vector<bool> group_present(nFilters, false);
    for (auto neighbor : neighbors) {
        int group_id = filter_ids[neighbor];
        if (group_id < nFilters) {
            group_present[group_id] = true;
        }
    }

    std::vector<int> missing_nodes;

    for (int g = 0; g < nFilters; ++g) {
        if (!group_present[g]) {
            std::vector<tableint> candidates;
            for (tableint i = 0; i < pre_candidates[g].size(); ++i) {
                candidates.push_back(pre_candidates[g][i]);
            }

            if (!candidates.empty()) {

                int num_extra_ep_neighbors = 3;
                std::vector<std::pair<size_t, tableint>> candidate_distances;
                for (auto candidate_id : candidates) {
                    char* ep_data = index->getDataByInternalId(ep_id);
                    char* candidate_data = index->getDataByInternalId(candidate_id);
                    size_t dist = index->fstdistfunc_(ep_data, candidate_data, index->dist_func_param_);
                    candidate_distances.emplace_back(dist, candidate_id);
                }
                std::sort(candidate_distances.begin(), candidate_distances.end());

                // pick top 3 closest
                int num_to_pick = std::min(num_extra_ep_neighbors, (int)candidate_distances.size());
                for (int i = 0; i < num_to_pick; i++) {
                    missing_nodes.push_back(candidate_distances[i].second);
                }

            }
        }
    }

    return missing_nodes;
}

inline void generate_ep_missing_group_info(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<uint8_t>& filter_ids)
{
    int ep_num, ep_dim;
    int* ep_ids_data = read_ivecs("sift_level1_ids.ivecs", ep_num, ep_dim);
    if (ep_dim != 1) {
        std::cerr << "Error: ep_id vectors should be 1-dimensional" << std::endl;
        delete[] ep_ids_data;
        return;
    }

    std::ofstream outFile(dataset_type + "_ep_id_missing_nodes.txt", std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Error: cannot open output file." << std::endl;
        delete[] ep_ids_data;
        return;
    }

    std::mutex file_mutex;

    ParallelFor(0, ep_num, nThreads, [&](size_t i, size_t /*thread_id*/) {
        tableint ep_id = static_cast<tableint>(ep_ids_data[i]);
        std::vector<tableint> neighbors = getNeighbors(index, ep_id);
        std::vector<int> missing_nodes = findMissingGroups(index, ep_id, neighbors, filter_ids);

        if (!missing_nodes.empty()) {
            std::ostringstream oss;
            oss << ep_id;
            for (const auto& node : missing_nodes) {
                oss << " " << node;
            }
            oss << "\n";

            std::lock_guard<std::mutex> lock(file_mutex);
            outFile << oss.str();
        }
    });

    delete[] ep_ids_data;
    outFile.close();
    std::cout << "Created " + dataset_type + "_ep_id_missing_nodes.txt with " << ep_num << " entries." << std::endl;
}

inline void generate_ep_missing_group_info_laion(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<uint8_t>& filter_ids)
{
    int ep_num, ep_dim;
    int* ep_ids_data = read_ivecs(dataset_type + "_level1_ids.ivecs", ep_num, ep_dim);
    if (ep_dim != 1) {
        std::cerr << "Error: ep_id vectors should be 1-dimensional" << std::endl;
        delete[] ep_ids_data;
        return;
    }

    std::ofstream outFile(dataset_type + "_ep_id_missing_nodes.txt", std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Error: cannot open output file." << std::endl;
        delete[] ep_ids_data;
        return;
    }

    std::mutex file_mutex;
    std::vector<std::vector<int>> pre_candidates;
    pre_candidates.resize(nFilters);
    for (int i = 0; i < nElements; i++) {
        pre_candidates[filter_ids[i]].push_back(i);
    }


    ParallelFor(0, ep_num, nThreads, [&](size_t i, size_t /*thread_id*/) {
        tableint ep_id = static_cast<tableint>(ep_ids_data[i]);

        std::vector<tableint> neighbors = getNeighbors(index, ep_id);
        std::vector<int> missing_nodes = findMissingGroups_laion(index, ep_id, neighbors, filter_ids, pre_candidates);

        if (!missing_nodes.empty()) {
            std::ostringstream oss;
            oss << ep_id;
            for (const auto& node : missing_nodes) {
                oss << " " << node;
            }
            oss << "\n";

            std::lock_guard<std::mutex> lock(file_mutex);
            outFile << oss.str();
        }
    });

    delete[] ep_ids_data;
    outFile.close();
    std::cout << "Created " + dataset_type + "_ep_id_missing_nodes.txt with " << ep_num << " entries." << std::endl;
}

inline void generate_ep_missing_group_info_hnm(
    const hnswlib::HierarchicalNSW<float>* index,
    const std::vector<std::vector<uint8_t>>& filter_ids,
    int filtering_cond_idx)
{
    int ep_num, ep_dim;
    int* ep_ids_data = read_ivecs(dataset_type + "_level1_ids.ivecs", ep_num, ep_dim);
    if (ep_dim != 1) {
        std::cerr << "Error: ep_id vectors should be 1-dimensional" << std::endl;
        delete[] ep_ids_data;
        return;
    }

    std::ofstream outFile(std::to_string(filtering_cond_idx) + "_" + dataset_type + "_ep_id_missing_nodes.txt", std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Error: cannot open output file." << std::endl;
        delete[] ep_ids_data;
        return;
    }

    std::mutex file_mutex;
    std::vector<std::vector<int>> pre_candidates;
    pre_candidates.resize(nFilters);
    for (int i = 0; i < nElements; i++) {
        pre_candidates[filter_ids[filtering_cond_idx][i]].push_back(i);
    }


    ParallelFor(0, ep_num, nThreads, [&](size_t i, size_t /*thread_id*/) {
        tableint ep_id = static_cast<tableint>(ep_ids_data[i]);

        std::vector<tableint> neighbors = getNeighbors(index, ep_id);
        std::vector<int> missing_nodes = findMissingGroups_laion(index, ep_id, neighbors, filter_ids[filtering_cond_idx], pre_candidates);

        if (!missing_nodes.empty()) {
            std::ostringstream oss;
            oss << ep_id;
            for (const auto& node : missing_nodes) {
                oss << " " << node;
            }
            oss << "\n";

            std::lock_guard<std::mutex> lock(file_mutex);
            outFile << oss.str();
        }
    });

    delete[] ep_ids_data;
    outFile.close();
    std::cout << "Created " + dataset_type + "_ep_id_missing_nodes.txt with " << ep_num << " entries." << std::endl;
}

#include <unordered_set>
#include <unordered_map>
#include <queue>

// Function to extract cluster sizes within given subset of nodes
std::vector<int> compute_clusters_in_subset(
    hnswlib::HierarchicalNSW<float>* alg_hnsw,
    const std::vector<int>& node_indices,
    int level = 0 // analyze graph at level 0
) {
    std::unordered_set<int> node_set(node_indices.begin(), node_indices.end());
    std::unordered_set<int> visited;
    std::vector<int> cluster_sizes;

    for (int node : node_indices) {
        if (visited.find(node) != visited.end())
            continue;

        int cluster_size = 0;
        std::queue<int> q;
        q.push(node);
        visited.insert(node);

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cluster_size++;

            // Get neighbors of current node at given level
            unsigned int*  neighbors = alg_hnsw->get_linklist0(current);
            int size_of_neighbors = neighbors[0];
            for (int i = 1; i <= size_of_neighbors; i++) {
                int neighbor = neighbors[i];
                if (node_set.find(neighbor) != node_set.end() && visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        cluster_sizes.push_back(cluster_size);
    }
    return cluster_sizes;
}

struct Edge {
    int from, to;
    float distance;
};

struct CompareEdge {
    bool operator()(const Edge& a, const Edge& b) {
        return a.distance > b.distance;
    }
};

float L2Distance(float* a, float* b, int dim) {
    float dist = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

std::vector<int> get_steiner_tree(hnswlib::HierarchicalNSW<float>* alg_hnsw,
                                  int dim, std::vector<int> terminal_nodes) {
    std::vector<int> terminals;

    // Identify terminal nodes
    for (int i = 0; i < terminal_nodes.size(); ++i) {
        terminals.push_back(terminal_nodes[i]);
    }

    // Early exit
    if (terminals.size() < 1) return terminals;

    // Build complete graph among terminals (using L2 distance from HNSW vectors)
    std::vector<Edge> complete_graph;
    std::mutex graph_mutex;
    ParallelFor(0, terminals.size(), nThreads, [&](size_t i, size_t /*thread_id*/) { 
        for (size_t j = i + 1; j < terminals.size(); ++j) {
            float* vec_i = reinterpret_cast<float*>(alg_hnsw->getDataByInternalId(terminals[i]));
            float* vec_j = reinterpret_cast<float*>(alg_hnsw->getDataByInternalId(terminals[j]));
            float dist = L2Distance(vec_i, vec_j, dim);

            std::lock_guard<std::mutex> lock(graph_mutex);
            complete_graph.push_back({terminals[i], terminals[j], dist});
        }
    });

    // Run Prim's algorithm to compute MST over terminal complete graph
    std::priority_queue<Edge, std::vector<Edge>, CompareEdge> pq;
    std::unordered_set<int> in_tree;
    std::unordered_map<int, int> parent;
    std::vector<int> steiner_nodes;

    in_tree.insert(terminals[0]);
    for (const auto& edge : complete_graph) {
        if (edge.from == terminals[0] || edge.to == terminals[0])
            pq.push(edge);
    }

    while (!pq.empty() && in_tree.size() < terminals.size()) {
        Edge e = pq.top(); pq.pop();
        int u = e.from, v = e.to;
        if (in_tree.count(u) && in_tree.count(v)) continue;

        int new_node = in_tree.count(u) ? v : u;
        in_tree.insert(new_node);
        parent[new_node] = in_tree.count(u) ? u : v;

        for (const auto& edge : complete_graph) {
            if ((edge.from == new_node && !in_tree.count(edge.to)) ||
                (edge.to == new_node && !in_tree.count(edge.from))) {
                pq.push(edge);
            }
        }
    }

    std::mutex steiner_mutex;

    ParallelFor(0, parent.size(), nThreads, [&](size_t idx, size_t) {
        auto it = std::next(parent.begin(), idx);
        int child = it->first;
        int par = it->second;

        std::unordered_set<int> visited;
        std::queue<int> q;
        std::unordered_map<int, int> came_from;

        q.push(par);
        visited.insert(par);
        bool found = false;

        while (!q.empty() && !found) {
            int current = q.front(); q.pop();
            unsigned int* neighbors = alg_hnsw->get_linklist0(current);  // level 0
            int num_neighbors = neighbors[0];
            for (int neighbor_idx = 1; neighbor_idx <= num_neighbors; neighbor_idx++) {
                int neighbor = neighbors[neighbor_idx];
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
                    came_from[neighbor] = current;
                    q.push(neighbor);
                    if (neighbor == child) {
                        found = true;
                        break;
                    }
                }
            }
        }
        if (found) {
            std::vector<int> local_path;
            int node = child;
            while (node != par) {
                local_path.push_back(node);
                node = came_from[node];
            }
            local_path.push_back(par);

            std::lock_guard<std::mutex> lock(steiner_mutex);
            steiner_nodes.insert(steiner_nodes.end(), local_path.begin(), local_path.end());
        }
    });

    // Remove duplicates
    std::sort(steiner_nodes.begin(), steiner_nodes.end());
    steiner_nodes.erase(std::unique(steiner_nodes.begin(), steiner_nodes.end()), steiner_nodes.end());

    return steiner_nodes;
}

/*Based on the hop*/
std::vector<std::vector<int>> getClusterNodeInfo(
    hnswlib::HierarchicalNSW<float>* alg_hnsw,
    int group_id,
    int cluster_factor = 1)
{
    std::vector<std::vector<int>> cluster_node_info;

    // build list of nodes in this group
    std::vector<int> filtered_nodes;
    for (int i = 0; i < nElements; ++i) {
        if (filter_ids[i] == group_id) {
            filtered_nodes.push_back(i);
        }
    }

    // for fast lookup: node -> which cluster it's in (or -1)
    std::vector<int> node_cluster(nElements, -1);

    // process each node in turn
    for (int start : filtered_nodes) {
        if (node_cluster[start] != -1) 
            continue;  // already clustered

        // ---- BFS setup ----
        std::queue<int>       q;
        std::vector<int>      depth(nElements, std::numeric_limits<int>::max());
        std::vector<int>      parent(nElements, -1);

        depth[start] = 0;
        q.push(start);

        bool added_to_existing = false;
        int  target_cluster = -1;
        int  meet_node      = -1;

        // ---- BFS up to cluster_factor ----
        while (!q.empty() && !added_to_existing) {
            int cur = q.front(); 
            q.pop();

            if (depth[cur] >= cluster_factor) 
                continue;

            // get neighbors from level 0
            unsigned int* nbrs = alg_hnsw->get_linklist0(cur);
            int            sz  = static_cast<int>(nbrs[0]);

            for (int i = 1; i <= sz; ++i) {
                int nb = nbrs[i];
                if (filter_ids[nb] != group_id) 
                    continue;        // only same‐group

                if (depth[nb] <= depth[cur] + 1) 
                    continue;        // already visited at ≤ this depth

                depth[nb]  = depth[cur] + 1;
                parent[nb] = cur;

                // if neighbor already sits in an existing cluster, we can stop
                if (node_cluster[nb] != -1) {
                    added_to_existing = true;
                    target_cluster    = node_cluster[nb];
                    meet_node         = nb;
                    break;
                }

                q.push(nb);
            }
        }

        if (added_to_existing) {
            // walk back the path from meet_node to start
            int cursor = meet_node;
            while (cursor != -1 && node_cluster[cursor] == -1) {
                // mark connector bit on this path node
                connector_bits[cursor].set(group_id);
                // add it into that existing cluster
                cluster_node_info[target_cluster].push_back(cursor);
                node_cluster[cursor] = target_cluster;
                cursor = parent[cursor];
            }
        }
        else {
            // no existing cluster hit: make a brand-new one
            int new_cluster_idx = static_cast<int>(cluster_node_info.size());
            cluster_node_info.emplace_back();
            
            // walk the entire reachable subgraph (up to depth) to form the cluster
            std::queue<int> q2;
            q2.push(start);
            depth.assign(nElements, std::numeric_limits<int>::max());
            depth[start] = 0;

            while (!q2.empty()) {
                int cur = q2.front(); 
                q2.pop();

                // add cur to the new cluster
                connector_bits[cur].set(group_id);
                cluster_node_info[new_cluster_idx].push_back(cur);
                node_cluster[cur] = new_cluster_idx;

                if (depth[cur] >= cluster_factor) 
                    continue;

                unsigned int* nbrs = alg_hnsw->get_linklist0(cur);
                int            sz  = static_cast<int>(nbrs[0]);
                for (int i = 1; i <= sz; ++i) {
                    int nb = nbrs[i];
                    if (filter_ids[nb] != group_id) 
                        continue;
                    if (depth[nb] <= depth[cur] + 1) 
                        continue;
                    depth[nb] = depth[cur] + 1;
                    parent[nb] = cur;
                    q2.push(nb);
                }
            }
        }
    }

    return cluster_node_info;
}

std::vector<int> getTerminalNodes(const std::vector<std::vector<int>>& cluster_node_info, int steiner_factor = 1) {
    // std::cout << "steiner_factor: " << steiner_factor << std::endl;
    std::vector<int> terminal_nodes;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto& cluster : cluster_node_info) {
        int pick_count = std::min(steiner_factor, static_cast<int>(cluster.size()));
        std::vector<int> indices(cluster.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int i = 0; i < pick_count; ++i) {
            terminal_nodes.push_back(cluster[indices[i]]);
        }
    }

    return terminal_nodes;
}

std::vector<std::vector<int>> findConnectorPathsFromEPs(
    hnswlib::HierarchicalNSW<float>* alg_hnsw,
    const std::vector<int>& ep_ids,
    int group_id,
    int e_factor
) {
    // std::cout << "ep_factor: " << e_factor << std::endl;
    std::vector<std::vector<int>> connector_paths_from_eps;

    for (int ep_id : ep_ids) {
        std::vector<std::vector<int>> paths_found;
        std::unordered_set<int> visited;
        std::queue<std::vector<int>> q;
        
        q.push({ep_id});
        visited.insert(ep_id);

        while (!q.empty() && static_cast<int>(paths_found.size()) < e_factor) {
            std::vector<int> current_path = q.front();
            q.pop();

            int current_node = current_path.back();
            unsigned int* neighbors = alg_hnsw->get_linklist0(current_node);
            int neighbor_count = neighbors[0];

            for (int i = 1; i <= neighbor_count; ++i) {
                int neighbor = neighbors[i];

                if (visited.count(neighbor)) continue;

                visited.insert(neighbor);
                std::vector<int> new_path = current_path;
                new_path.push_back(neighbor);

                if (filter_ids[neighbor] == group_id) {
                    paths_found.push_back(new_path);
                    if (static_cast<int>(paths_found.size()) >= e_factor) break;
                } else {
                    q.push(new_path); // Continue exploring
                }
            }
        }

        // If fewer than ep_factor paths found, just use what we got
        for (const auto& path : paths_found) {
            connector_paths_from_eps.push_back(path);
        }
    }

    return connector_paths_from_eps;
}

std::vector<uint8_t> flattenBitsets(const std::vector<std::bitset<256>>& input) {
    constexpr size_t BitsPerSet   = 256;
    constexpr size_t BitsPerChunk = 8;
    constexpr size_t Chunks       = BitsPerSet / BitsPerChunk;  // 32

    std::vector<uint8_t> output;
    output.reserve(input.size() * Chunks);

    for (size_t chunk = 0; chunk < Chunks; ++chunk) {
        size_t base = chunk * BitsPerChunk;
        for (const auto& bs : input) {
            uint8_t byte = 0;
            // pack bits [base .. base+7] into one byte
            for (size_t bit = 0; bit < BitsPerChunk; ++bit) {
                if (bs[base + bit]) {
                    byte |= (uint8_t(1) << bit);
                }
            }
            output.push_back(byte);
        }
    }

    return output;
}

// Helper: trim whitespace from both ends of a string
static inline std::string trim(const std::string &s) {
    auto ws_front = std::find_if_not(s.begin(), s.end(), [](unsigned char c){ return std::isspace(c); });
    auto ws_back  = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c){ return std::isspace(c); }).base();
    return (ws_front < ws_back ? std::string(ws_front, ws_back) : std::string());
}

// Parses "hnm_metadata.txt" and returns a vector of unique counts
// in the same order as the fields list.
std::vector<std::vector<std::string>>  parseUniqueCounts(std::vector<std::string>& condition_name_Vecs, std::vector<std::vector<std::string>>& condition_value_Vecs) {
    static const std::vector<std::string> fields = {
        "product_code",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_code",
        "index_name",
        "index_group_no",
        "section_no",
        "garment_group_no"
    };
    for (auto field: fields) {
        condition_name_Vecs.push_back(field);
    }
    std::unordered_set<std::string> valid(fields.begin(), fields.end());
    std::unordered_map<std::string, size_t> unique_counts;
    std::unordered_map<std::string, size_t> my_map;

    std::ifstream in("./profiled/hnm_metadata.txt");
    if (!in) throw std::runtime_error("Unable to open hnm_metadata.txt");

    std::vector<std::vector<std::string>> unique_values;

    std::string line, current, my_current;
    while (std::getline(in, line)) {
        // lower-case copy for case-insensitive checks
        std::string lc = line;
        std::transform(lc.begin(), lc.end(), lc.begin(), [](unsigned char c){ return std::tolower(c); });

        // Detect a new field: "Field: name"
        if (lc.rfind("field:", 0) == 0) {
            auto pos = line.find(':');
            try{
                if (pos != std::string::npos && pos + 1 < line.size()) {
                    std::string name = trim(line.substr(pos + 1));
                    current = valid.count(name) ? name : std::string();
                    my_current = name;
                }
            }
            catch (const std::out_of_range &e) {
                std::cerr << "Error in Field parsing at line: \"" 
                        << line << "\" pos=" << pos 
                        << " size=" << line.size() << "\n";
                throw;  // rethrow if you want to crash here
            }
        }
        // Detect the unique count line: "Num. Unique values: N"
        else if (!current.empty() && lc.find("num. unique values:") != std::string::npos) {
            auto pos = line.find(':');
            try{
                if (pos != std::string::npos && pos + 1 < line.size()) {
                    size_t val = std::stoul(trim(line.substr(pos + 1)));
                    unique_counts[current] = val;
                    my_map[my_current] = val;
                }
            }
            catch (const std::out_of_range &e) {
                std::cerr << "Error in Field parsing at line: \"" 
                        << line << "\" pos=" << pos 
                        << " size=" << line.size() << "\n";
                throw;  // rethrow if you want to crash here
            }
            // current.clear();
        }
        else if (!current.empty() && lc.find("list of unique values:") != std::string::npos) {
            auto b = line.find('[');
            auto e = line.find(']', b);
            if (b != std::string::npos && e != std::string::npos) {
                std::string blob;
                try {
                    blob = line.substr(b + 1, e - b - 1);
                }
                catch (const std::out_of_range &e) {
                    std::cerr << "Error in Field parsing at line: \"" 
                            << line << "\" b + 1=" << b + 1 
                            << " size=" << line.size() << "\n";
                    throw;  // rethrow if you want to crash here
                }
                std::vector<std::string> entries;

                if (current == "product_code") {
                    // parse as ints, then take min/max
                    std::istringstream ss(blob);
                    std::vector<int> nums;
                    int x;
                    while (ss >> x) {
                        nums.push_back(x);
                        if (ss.peek() == ',') ss.ignore();
                    }
                    if (!nums.empty()) {
                        int mn = *std::min_element(nums.begin(), nums.end());
                        int mx = *std::max_element(nums.begin(), nums.end());
                        entries.push_back(std::to_string(mn));
                        entries.push_back(std::to_string(mx));
                    }
                } 
                else {
                    // for other fields, keep all as strings
                    // std::istringstream ss2(blob);
                    // std::string token;
                    // while (std::getline(ss2, token, ',')) {
                    //     entries.push_back(trim(token));
                    // }
                    std::string cur;
                    bool inQuote = false;
                    for (char c : blob) {
                        if (c == '\'') {
                            // toggle “inside quote” state, but don’t include the quote in the token
                            inQuote = !inQuote;
                        }
                        else if (c == ',' && !inQuote) {
                            // found a delimiter *outside* quotes
                            auto t = trim(cur);
                            if (!t.empty()) entries.push_back(t);
                            cur.clear();
                        }
                        else {
                            cur += c;
                        }
                    }
                    // last token
                    auto t = trim(cur);
                    if (!t.empty()) entries.push_back(t);
                }
                // for (auto entry: entries) {
                //     std::cout << entry << " ";
                // }
                // std::cout << std::endl;
                condition_value_Vecs.push_back(std::move(entries));
                // unique_values.push_back(std::move(entries));
            }
            current.clear();
        }
    }
    return unique_values;
}
