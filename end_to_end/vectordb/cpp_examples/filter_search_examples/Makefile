CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pthread

INCLUDES = -I../../hnswlib/filter_search_hnswlib -I../../hnswlib

all: hnswlib_filter_search.run

hnswlib_filter_search.run: hnswlib_filter_search.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f hnswlib_filter_search.run

.PHONY: all clean

