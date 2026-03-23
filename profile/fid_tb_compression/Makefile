CXX      ?= g++
REPO_ROOT := $(abspath $(CURDIR)/../..)
QPL_INC  := $(REPO_ROOT)/qpl/include
QPL_LIB  := $(REPO_ROOT)/qpl/build/sources

JSON_INC  := $(REPO_ROOT)/end_to_end/vectordb/hnswlib/build_fid_tb
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -I$(QPL_INC) -I$(JSON_INC)
LDLIBS   := $(QPL_LIB)/libqpl.a -ldl -llz4

TARGET := compression.run
SRC    := compression.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
