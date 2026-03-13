CXX ?= g++

REPO_ROOT ?= /home/jykang5/compass
QPL_INC ?= $(REPO_ROOT)/qpl/include
QPL_LIB ?= $(REPO_ROOT)/qpl/build/sources/libqpl.a

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -I$(QPL_INC)
LDFLAGS ?=
LDLIBS ?= $(QPL_LIB) -ldl

TARGET := decompress_scan_alignment_bench.run
SRC := decompress_scan_alignment_bench.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	@if [ ! -f "$(QPL_LIB)" ]; then \
		echo "Error: QPL static lib not found at $(QPL_LIB)"; \
		echo "Build QPL first (e.g., /home/jykang5/compass/qpl_build.sh)."; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
