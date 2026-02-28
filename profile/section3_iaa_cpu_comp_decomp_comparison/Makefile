CXX ?= g++

QPL_INC ?= /home/jykang5/src/qpl/include
QPL_LIB ?= /home/jykang5/src/qpl/build/lib
LOCAL_NLOHMANN_INC ?= /home/jykang5/compass/end_to_end/vectordb

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
