# Compiler and flags for your existing project
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Raylib library and include directories from pkg-config
RAYLIB_FLAGS = $(shell pkg-config --libs --cflags raylib)

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Executable name for the original project
EXE = tetris

# Find all .cpp files in src directory
SRC = $(wildcard $(SRC_DIR)/*.cpp)

# Create a list of object files in the obj directory
OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))

# Default target
all: $(BIN_DIR)/$(EXE) TetrisRL

# Compile and link the original tetris project with Raylib
$(BIN_DIR)/$(EXE): $(OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(RAYLIB_FLAGS)

# Compile each .cpp file to an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(RAYLIB_FLAGS)

# Run the original program
run: $(BIN_DIR)/$(EXE)
	./$(BIN_DIR)/$(EXE)

# Create directories if they don't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean up build files for the original project
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets for the original project
.PHONY: all run clean

# CMake build for TetrisRL
TetrisRL:
	@mkdir -p build
	@cd build && cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -G "Unix Makefiles" ..
	@cd build && cmake --build .

# Clean CMake build files
clean-cmake:
	rm -rf build
