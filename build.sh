#!/bin/bash

BUILD_DIR="build"
DATA_ZIP="data.zip"
DATA_DIR="data"

echo "--- Data Handling ---"

if [ -f "$DATA_ZIP" ]; then
    if [ -d "$DATA_DIR" ]; then
        echo "Directory '$DATA_DIR/' already exists. Skipping '$DATA_ZIP' extraction."
    else
        echo "Found '$DATA_ZIP'. Extracting data..."
        unzip -q "$DATA_ZIP" 
        if [ $? -ne 0 ]; then
            echo "Error: Failed to extract '$DATA_ZIP'. Check if 'unzip' is installed."
            exit 1
        fi
    fi
else
    echo "Warning: '$DATA_ZIP' not found in the root directory. Skipping data extraction."
fi

echo "--- Build Process ---"

echo "Configuring project..."
cmake -S . -B $BUILD_DIR

if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

echo "Building project..."
cmake --build $BUILD_DIR

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

echo "--- Complete ---"
echo "Build successful! Executable 'neural_network_cpp' is in the root directory."