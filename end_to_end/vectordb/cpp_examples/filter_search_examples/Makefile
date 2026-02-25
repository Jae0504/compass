CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pthread

ROOT_DIR ?= $(abspath ../../../..)

INCLUDES = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib -I../../hnswlib/build_fid_tb
INCLUDES_LZ4 = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib/filter_search_hnswlib_with_lz4 -I../../hnswlib/build_fid_tb
INCLUDES_IAA = -I$(ROOT_DIR)/qpl/include -I../../hnswlib/filter_search_hnswlib -I../../hnswlib/filter_search_iaa -I../../hnswlib/build_fid_tb

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

all: hnswlib_filter_search.run compass_search_w_lz4.run compass_search_w_iaa.run

hnswlib_filter_search.run: hnswlib_filter_search.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

compass_search_w_lz4.run: compass_search_w_lz4.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_LZ4) $< -o $@ $(LIBS_LZ4)

compass_search_w_iaa.run: compass_search_w_iaa.cpp
	$(QPL_CHECK)
	$(CXX) $(CXXFLAGS) $(INCLUDES_IAA) $< -o $@ $(LIBS_IAA)

clean:
	rm -f hnswlib_filter_search.run compass_search_w_lz4.run compass_search_w_iaa.run

.PHONY: all clean
