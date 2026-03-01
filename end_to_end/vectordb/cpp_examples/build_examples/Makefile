CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pthread

INCLUDES_FID_TB = -I../../hnswlib/build_fid_tb
INCLUDES_HNSW = -I../../hnswlib/build_hnsw
REPO_ROOT := $(abspath ../../../../)
ACORN_SUBMODULE := $(REPO_ROOT)/ACORN
ACORN_SRC_ROOT ?= $(if $(wildcard $(ACORN_SUBMODULE)/faiss/IndexACORN.h),$(ACORN_SUBMODULE),/home/jykang5/ACORN)
ACORN_BUILD_ROOT ?= $(if $(wildcard $(ACORN_SUBMODULE)/build/faiss/libfaiss.a),$(ACORN_SUBMODULE)/build,/home/jykang5/ACORN/build)
INCLUDES_ACORN = -I$(ACORN_SRC_ROOT)
LDLIBS_ACORN = $(ACORN_BUILD_ROOT)/faiss/libfaiss.a -fopenmp -lopenblas -lz -ldl -lm -pthread

all: build_FID_TB.run build_hnsw.run build_acorn.run

build_FID_TB.run: build_FID_TB.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_FID_TB) $< -o $@

build_hnsw.run: build_hnsw.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_HNSW) $< -o $@

build_acorn.run: build_acorn.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_ACORN) $< -o $@ $(LDLIBS_ACORN)

clean:
	rm -f build_FID_TB.run build_hnsw.run build_acorn.run

.PHONY: all clean
