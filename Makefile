# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Raylib library and include directories from pkg-config
RAYLIB_FLAGS = $(shell pkg-config --libs --cflags raylib)

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Executable name
EXE = tetris

# Find all .cpp files in src directory
SRC = $(wildcard $(SRC_DIR)/*.cpp)

# Create a list of object files in the obj directory
OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))

# Default target
all: $(BIN_DIR)/$(EXE)

# Compile and link
$(BIN_DIR)/$(EXE): $(OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(RAYLIB_FLAGS)

# Compile each .cpp file to an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(RAYLIB_FLAGS)

# Run the program
run: $(BIN_DIR)/$(EXE)
	./$(BIN_DIR)/$(EXE)

# Create directories if they don't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all run clean
