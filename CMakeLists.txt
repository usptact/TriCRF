cmake_minimum_required(VERSION 3.10)

# Project definition
project(TriCRF 
    VERSION 1.0.0
    DESCRIPTION "Triangular-chain Conditional Random Fields (Tri-CRF)"
    LANGUAGES CXX C
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
endif()

# Include directories
include_directories(src)

# Source files
set(TRICRF_SOURCES
    src/Main.cpp
    src/CRF.cpp
    src/MaxEnt.cpp
    src/TriCRF1.cpp
    src/TriCRF2.cpp
    src/TriCRF3.cpp
    src/Evaluator.cpp
    src/Param.cpp
    src/Data.cpp
    src/LBFGS.cpp
    src/Utility.cpp
)

# Header files
set(TRICRF_HEADERS
    src/CRF.h
    src/MaxEnt.h
    src/TriCRF1.h
    src/TriCRF2.h
    src/TriCRF3.h
    src/Evaluator.h
    src/Param.h
    src/Data.h
    src/LBFGS.h
    src/Utility.h
)

# Create executable
add_executable(TriCRF ${TRICRF_SOURCES} ${TRICRF_HEADERS})

# Set target properties
set_target_properties(TriCRF PROPERTIES
    OUTPUT_NAME "tricrf"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Link libraries if needed
# find_package(Threads REQUIRED)
# target_link_libraries(TriCRF Threads::Threads)

# Install targets
install(TARGETS TriCRF
    RUNTIME DESTINATION bin
)

# Install example files
install(DIRECTORY example/
    DESTINATION share/TriCRF/examples
    FILES_MATCHING PATTERN "*.cfg" PATTERN "*.data"
)

# Install documentation
install(FILES README LICENSE
    DESTINATION share/doc/TriCRF
)

# Create a package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TriCRFConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Enable testing
enable_testing()

# Add a simple test
add_test(NAME TriCRF_version
    COMMAND TriCRF
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Set test properties
set_tests_properties(TriCRF_version PROPERTIES
    WILL_FAIL TRUE
    PASS_REGULAR_EXPRESSION "Usage"
)

# Print configuration summary
message(STATUS "")
message(STATUS "TriCRF Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  C standard: ${CMAKE_C_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
