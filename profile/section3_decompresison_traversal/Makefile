CXX ?= g++

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
QPL_INC   := $(REPO_ROOT)/qpl/include
QPL_LIB   := $(REPO_ROOT)/qpl/build/sources/libqpl.a

# profile.cpp uses the hnswlib stub qpl header — no QPL_INC here.
CXXFLAGS_PROFILE := -O2 -std=c++17 -Wall -Wextra -Wpedantic

# iaa_decomp.cpp uses the REAL QPL header.
CXXFLAGS_IAA := -O2 -std=c++17 -Wall -Wextra -Wpedantic -I$(QPL_INC)

TARGET := profile_section3_decompression
LIBS := -llz4 -lz -lpthread $(QPL_LIB) -ldl

.PHONY: all clean

all: $(TARGET)

$(TARGET): profile.cpp iaa_decomp.cpp iaa_decomp.h
	$(CXX) $(CXXFLAGS_PROFILE) -c -o profile.o  profile.cpp
	$(CXX) $(CXXFLAGS_IAA)     -c -o iaa_decomp.o iaa_decomp.cpp
	$(CXX) -o $@ profile.o iaa_decomp.o $(LIBS)

clean:
	rm -f $(TARGET) profile.o iaa_decomp.o
