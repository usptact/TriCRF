#!/bin/bash

# TriCRF Build Script
# This script builds the TriCRF project using CMake

set -e  # Exit on any error

echo "==============================================="
echo "  TriCRF - Triangular-chain CRF Build Script"
echo "==============================================="

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed or not in PATH"
    echo "Please install CMake 3.10 or later"
    exit 1
fi

# Check CMake version
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo "Using CMake version: $CMAKE_VERSION"

# Create build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Change to build directory
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring project with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ -f "bin/tricrf" ]; then
    echo ""
    echo "==============================================="
    echo "  Build completed successfully!"
    echo "==============================================="
    echo "Executable location: $(pwd)/bin/tricrf"
    echo ""
    echo "Usage:"
    echo "  ./bin/tricrf                    # Show usage"
    echo "  ./bin/tricrf <config_file>      # Run with config"
    echo ""
    echo "Example:"
    echo "  ./bin/tricrf ../example/example.cfg"
    echo ""
else
    echo "Error: Build failed - executable not found"
    exit 1
fi
