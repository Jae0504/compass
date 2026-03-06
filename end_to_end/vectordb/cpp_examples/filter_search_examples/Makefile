CXX ?= g++
CXXFLAGS ?= -Ofast -std=c++17 -march=native -fopenmp -ftree-vectorize -DNDEBUG -Wall -Wextra -pthread
CXXFLAGS_LZ4 ?= -O3 -std=c++17 -Wall -Wextra -pthread

ROOT_DIR ?= $(abspath ../../../..)

INCLUDES = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib -I../../hnswlib/build_fid_tb
INCLUDES_LZ4 = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib/filter_search_hnswlib_with_lz4 -I../../hnswlib/build_fid_tb
INCLUDES_IAA = -I$(ROOT_DIR)/qpl/include -I../../hnswlib/filter_search_hnswlib -I../../hnswlib/filter_search_iaa -I../../hnswlib/build_fid_tb
ACORN_SUBMODULE := $(ROOT_DIR)/ACORN
ACORN_SRC_ROOT ?= $(if $(wildcard $(ACORN_SUBMODULE)/faiss/IndexACORN.h),$(ACORN_SUBMODULE),/home/jykang5/ACORN)
ACORN_BUILD_ROOT ?= $(if $(wildcard $(ACORN_SUBMODULE)/build/faiss/libfaiss.a),$(ACORN_SUBMODULE)/build,/home/jykang5/ACORN/build)
INCLUDES_ACORN = $(INCLUDES) -I$(ACORN_SRC_ROOT)
LIBS_ACORN = $(ACORN_BUILD_ROOT)/faiss/libfaiss.a -fopenmp -lopenblas -lz -ldl -lm -pthread

LIBS_LZ4 = -llz4

QPL_STATIC_LIB :=
ifneq ($(wildcard $(ROOT_DIR)/qpl/build/sources/libqpl.a),)
QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/sources/libqpl.a
else ifneq ($(wildcard $(ROOT_DIR)/qpl/build/lib/libqpl.a),)
QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/lib/libqpl.a
else ifneq ($(wildcard $(ROOT_DIR)/qpl/build/libqpl.a),)
QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/libqpl.a
endif

LIBS_IAA = $(QPL_STATIC_LIB) -ldl

ifeq ($(strip $(QPL_STATIC_LIB)),)
QPL_CHECK = @echo "Error: QPL static library not found. Build qpl first with ./qpl_build.sh" && false
else
QPL_CHECK = @true
endif

ACORN_CHECK = @if [ ! -f "$(ACORN_SRC_ROOT)/faiss/IndexACORN.h" ] || [ ! -f "$(ACORN_BUILD_ROOT)/faiss/libfaiss.a" ]; then \
	echo "Error: ACORN header/lib not found. Expected $(ACORN_SRC_ROOT)/faiss/IndexACORN.h and $(ACORN_BUILD_ROOT)/faiss/libfaiss.a"; \
	false; \
else \
	true; \
fi

all: hnswlib_filter_search.run compass_search_w_lz4.run compass_search_w_iaa.run compass_search_w_iaa_async.run compass_search_w_iaa_async_grouping.run compass_search_w_iaa_async_range_filter.run compass_search_w_iaa_async_multiple_em.run acorn_search.run

hnswlib_filter_search.run: hnswlib_filter_search.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

compass_search_w_lz4.run: compass_search_w_lz4.cpp
	$(CXX) $(CXXFLAGS_LZ4) $(INCLUDES_LZ4) $< -o $@ $(LIBS_LZ4)

compass_search_w_iaa.run: compass_search_w_iaa.cpp
	$(QPL_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_IAA) $< -o $@ $(LIBS_IAA)

compass_search_w_iaa_async.run: compass_search_w_iaa_async.cpp
	$(QPL_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_IAA) $< -o $@ $(LIBS_IAA)

compass_search_w_iaa_async_grouping.run: compass_search_w_iaa_async_grouping.cpp
	$(QPL_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_IAA) $< -o $@ $(LIBS_IAA)

compass_search_w_iaa_async_range_filter.run: compass_search_w_iaa_async_range_filter.cpp
	$(QPL_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_IAA) $< -o $@ $(LIBS_IAA)

compass_search_w_iaa_async_multiple_em.run: compass_search_w_iaa_async_multiple_em.cpp
	$(QPL_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_IAA) $< -o $@ $(LIBS_IAA)

acorn_search.run: acorn_search.cpp
	$(ACORN_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_ACORN) $< -o $@ $(LIBS_ACORN)

clean:
	rm -f hnswlib_filter_search.run compass_search_w_lz4.run compass_search_w_iaa.run compass_search_w_iaa_async.run compass_search_w_iaa_async_grouping.run compass_search_w_iaa_async_range_filter.run compass_search_w_iaa_async_multiple_em.run acorn_search.run

.PHONY: all clean
