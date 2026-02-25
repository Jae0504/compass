CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(SCRIPT_DIR)/../..)

DEFAULT_ACORN_DIR := $(ROOT_DIR)/ACORN
ifeq ($(wildcard $(DEFAULT_ACORN_DIR)/CMakeLists.txt),)
ifneq ($(wildcard /home/jykang5/ACORN/CMakeLists.txt),)
DEFAULT_ACORN_DIR := /home/jykang5/ACORN
endif
endif

ACORN_DIR ?= $(DEFAULT_ACORN_DIR)
ACORN_BUILD_DIR ?= $(ROOT_DIR)/ACORN/build
FAISS_STATIC_LIB := $(ACORN_BUILD_DIR)/faiss/libfaiss.a

TARGET := profile_section3
SRC := profile.cpp

INCLUDES := -I$(ACORN_DIR)
LIBS := $(FAISS_STATIC_LIB) -lopenblas -llz4 -lz -fopenmp -lpthread

.PHONY: all faiss clean

all: $(TARGET)

$(TARGET): $(SRC) $(FAISS_STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC) $(LIBS)

$(FAISS_STATIC_LIB):
	cmake -S $(ACORN_DIR) -B $(ACORN_BUILD_DIR) \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
		-DFAISS_ENABLE_GPU=OFF \
		-DFAISS_ENABLE_PYTHON=OFF \
		-DBUILD_TESTING=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build $(ACORN_BUILD_DIR) --target faiss -j

faiss: $(FAISS_STATIC_LIB)

clean:
	rm -f $(TARGET)
