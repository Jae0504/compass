CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pthread

INCLUDES = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib -I../../hnswlib/build_fid_tb
INCLUDES_LZ4 = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib/filter_search_hnswlib_with_lz4 -I../../hnswlib/build_fid_tb
LIBS_LZ4 = -llz4

all: hnswlib_filter_search.run compass_search_w_lz4.run

hnswlib_filter_search.run: hnswlib_filter_search.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

compass_search_w_lz4.run: compass_search_w_lz4.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_LZ4) $< -o $@ $(LIBS_LZ4)

clean:
	rm -f hnswlib_filter_search.run compass_search_w_lz4.run

.PHONY: all clean
