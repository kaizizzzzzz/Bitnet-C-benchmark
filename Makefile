# Define the compiler and flags
CXX = g++ -g
CXXFLAGS = -std=c++17 -I./include

# Define directories
SRC_DIR = src_c
INFERENCE_DIR = inference

# Define source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp) $(INFERENCE_DIR)/load_model.cpp $(INFERENCE_DIR)/inference.cpp

# Define the output executable
OUTPUT = inference/inference

# Build target
$(OUTPUT): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(SRC_FILES)

# Clean up build files
clean:
	rm -f $(OUTPUT)
