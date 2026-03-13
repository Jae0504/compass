CXX ?= g++

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(SCRIPT_DIR)/../..)

CXXFLAGS_COMMON := -Ofast -std=c++17 -fopenmp -ftree-vectorize -DNDEBUG -Wall -Wextra -pthread -march=native
INCLUDES_COMMON := -I$(ROOT_DIR)/qpl/include -I$(ROOT_DIR)/end_to_end/vectordb/nlohmann

QPL_STATIC_LIB :=
ifneq ($(wildcard $(ROOT_DIR)/qpl/build/sources/libqpl.a),)
  QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/sources/libqpl.a
else ifneq ($(wildcard $(ROOT_DIR)/qpl/build/lib/libqpl.a),)
  QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/lib/libqpl.a
else ifneq ($(wildcard $(ROOT_DIR)/qpl/build/libqpl.a),)
  QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/libqpl.a
endif

ifeq ($(strip $(QPL_STATIC_LIB)),)
  $(error QPL static library not found. Build QPL first)
endif

LIBS_TRAVERSAL := $(QPL_STATIC_LIB) -ldl
LIBS_BENCH := $(QPL_STATIC_LIB) -ldl -lpthread -llz4

SRC_DIR := $(SCRIPT_DIR)/src

TARGET_EXACT := roofline_exact_iaa.run
TARGET_RANGE := roofline_range_iaa.run
TARGET_MULTI := roofline_multi_iaa.run
TARGET_BENCH := roofline_dynamic_wait_bench

SRC_EXACT := $(SRC_DIR)/compass_search_w_iaa_async_grouping_roofline.cpp
SRC_RANGE := $(SRC_DIR)/compass_search_w_iaa_only_tb_roofline.cpp
SRC_MULTI := $(SRC_DIR)/compass_search_w_iaa_multi_roofline.cpp
SRC_BENCH := $(SCRIPT_DIR)/roofline_dynamic_wait_bench.cpp

.PHONY: all clean

all: $(TARGET_EXACT) $(TARGET_RANGE) $(TARGET_MULTI) $(TARGET_BENCH)

$(TARGET_EXACT): $(SRC_EXACT)
	$(CXX) $(CXXFLAGS_COMMON) $(INCLUDES_COMMON) -o $@ $< $(LIBS_TRAVERSAL)

$(TARGET_RANGE): $(SRC_RANGE)
	$(CXX) $(CXXFLAGS_COMMON) $(INCLUDES_COMMON) -o $@ $< $(LIBS_TRAVERSAL)

$(TARGET_MULTI): $(SRC_MULTI)
	$(CXX) $(CXXFLAGS_COMMON) $(INCLUDES_COMMON) -o $@ $< $(LIBS_TRAVERSAL)

$(TARGET_BENCH): $(SRC_BENCH)
	$(CXX) $(CXXFLAGS_COMMON) $(INCLUDES_COMMON) -o $@ $< $(LIBS_BENCH)

clean:
	rm -f $(TARGET_EXACT) $(TARGET_RANGE) $(TARGET_MULTI) $(TARGET_BENCH)
