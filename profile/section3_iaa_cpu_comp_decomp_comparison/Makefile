CXX ?= g++

REPO_ROOT ?= $(abspath $(CURDIR)/../..)
QPL_INC ?= $(REPO_ROOT)/qpl/include
QPL_LIB ?= $(REPO_ROOT)/qpl/build/sources
LOCAL_NLOHMANN_INC ?= $(REPO_ROOT)/end_to_end/vectordb

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -I$(LOCAL_NLOHMANN_INC) -I/usr/include -I$(QPL_INC)
LDFLAGS ?=
LDLIBS ?= $(QPL_LIB)/libqpl.a -ldl -llz4 -lz

TARGET := profile.run
SRC := profile.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
