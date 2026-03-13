CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -march=native -Wall -Wextra

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR   := $(abspath $(SCRIPT_DIR)/../..)

# Auto-detect QPL static library
QPL_STATIC_LIB :=
ifneq ($(wildcard $(ROOT_DIR)/qpl/build/sources/libqpl.a),)
  QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/sources/libqpl.a
else ifneq ($(wildcard $(ROOT_DIR)/qpl/build/lib/libqpl.a),)
  QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/lib/libqpl.a
else ifneq ($(wildcard $(ROOT_DIR)/qpl/build/libqpl.a),)
  QPL_STATIC_LIB := $(ROOT_DIR)/qpl/build/libqpl.a
endif

ifeq ($(strip $(QPL_STATIC_LIB)),)
  $(error QPL static library not found. Build QPL first with: ./qpl_build.sh)
endif

INCLUDES := -I$(ROOT_DIR)/qpl/include
LIBS     := $(QPL_STATIC_LIB) -ldl -lpthread -llz4

TARGET := profile_roofline
SRC    := profile.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
