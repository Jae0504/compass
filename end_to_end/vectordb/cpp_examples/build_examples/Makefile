CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pthread

INCLUDES_FID_TB = -I../../hnswlib/build_fid_tb
INCLUDES_HNSW = -I../../hnswlib/build_hnsw

all: build_FID_TB.run build_hnsw.run

build_FID_TB.run: build_FID_TB.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_FID_TB) $< -o $@

build_hnsw.run: build_hnsw.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_HNSW) $< -o $@

clean:
	rm -f build_FID_TB.run build_hnsw.run

.PHONY: all clean
