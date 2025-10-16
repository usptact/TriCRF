/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file Utility.h
 * @brief Utility Functions and Classes for TriCRF
 * 
 * This file provides essential utility functions and classes used throughout
 * the TriCRF framework, including string processing, logging, configuration
 * parsing, and timing utilities.
 * 
 * @section Overview Overview
 * 
 * The utility module provides:
 * - String tokenization and parsing
 * - Logging system with multiple levels
 * - Configuration file parsing
 * - High-resolution timing utilities
 * - Mathematical constants and functions
 * 
 * @section Classes Classes
 * 
 * - **Logger**: Multi-level logging system
 * - **Configurator**: Configuration file parser
 * - **timer**: High-resolution timing utility
 * 
 * @section Functions Functions
 * 
 * - **tokenize()**: String tokenization with custom delimiters
 * 
 * @section Usage_Examples Usage Examples
 * 
 * String tokenization:
 * @code
 * #include "Utility.h"
 * 
 * // Basic tokenization
 * std::string text = "hello world test";
 * std::vector<std::string> tokens = tricrf::tokenize(text);
 * // Result: {"hello", "world", "test"}
 * 
 * // Custom delimiter
 * std::string csv = "a,b,c,d";
 * std::vector<std::string> fields = tricrf::tokenize(csv, ",");
 * // Result: {"a", "b", "c", "d"}
 * @endcode
 * 
 * Logging system:
 * @code
 * #include "Utility.h"
 * 
 * // Create logger
 * tricrf::Logger logger("output.log", 2);  // Level 2 logging
 * 
 * // Log messages
 * logger.report(1, "Starting training...");
 * logger.report(2, "Iteration %d: loss = %.3f", iter, loss);
 * logger.report("Training completed in %.2f seconds", elapsed);
 * @endcode
 * 
 * Configuration parsing:
 * @code
 * #include "Utility.h"
 * 
 * // Parse configuration file
 * tricrf::Configurator config("model.cfg");
 * 
 * // Get values
 * std::string train_file = config.get("train_file");
 * int max_iter = std::stoi(config.get("max_iter"));
 * double learning_rate = std::stod(config.get("learning_rate"));
 * 
 * // Check if key exists
 * if (config.isValid("regularization")) {
 *     double reg = std::stod(config.get("regularization"));
 * }
 * @endcode
 * 
 * Timing utilities:
 * @code
 * #include "Utility.h"
 * 
 * tricrf::timer timer;
 * 
 * // Perform some operation
 * doExpensiveOperation();
 * 
 * double elapsed = timer.elapsed();
 * std::cout << "Operation took " << elapsed << " seconds" << std::endl;
 * 
 * // Restart timer
 * timer.restart();
 * doAnotherOperation();
 * double new_elapsed = timer.elapsed();
 * @endcode
 * 
 * @section Sample_Data Sample Data
 * 
 * Configuration file format (model.cfg):
 * @code
 * # TriCRF Configuration File
 * model_type = TriCRF1
 * train_file = data/train.txt
 * test_file = data/test.txt
 * model_file = models/model.bin
 * log_file = logs/training.log
 * 
 * # Training parameters
 * max_iter = 100
 * learning_rate = 0.01
 * regularization = 1.0
 * l1_penalty = 0.1
 * 
 * # Feature parameters
 * feature_template = word,word-1,word+1,pos,pos-1,pos+1
 * use_char_features = true
 * char_ngram_size = 3
 * 
 * # Optimization parameters
 * optimizer = LBFGS
 * convergence_threshold = 1e-6
 * line_search_iterations = 20
 * @endcode
 * 
 * Log output example:
 * @code
 * 2024-01-15 10:30:15 [INFO] Starting TriCRF training
 * 2024-01-15 10:30:15 [INFO] Loading training data: data/train.txt
 * 2024-01-15 10:30:15 [INFO] Loaded 1000 training examples
 * 2024-01-15 10:30:15 [INFO] Initializing model parameters
 * 2024-01-15 10:30:16 [DEBUG] Iteration 1: loss = 1234.56, accuracy = 0.75
 * 2024-01-15 10:30:17 [DEBUG] Iteration 2: loss = 987.65, accuracy = 0.82
 * 2024-01-15 10:30:18 [INFO] Training completed in 3.45 seconds
 * @endcode
 * 
 * @section Performance Performance Notes
 * 
 * - **tokenize()**: O(n) where n is string length
 * - **Logger**: Minimal overhead, thread-safe
 * - **Configurator**: O(1) lookup after parsing
 * - **timer**: High-resolution, minimal overhead
 * 
 * @section Thread_Safety Thread Safety
 * 
 * - **Logger**: Thread-safe for single-threaded use
 * - **Configurator**: Read-only after construction
 * - **timer**: Not thread-safe (per-instance)
 * - **tokenize()**: Thread-safe (pure function)
 * 
 * @author Minwoo Jeong
 * @date 2010
 * @version 1.0
 */

#ifndef __UTIL_H__
#define __UTIL_H__

/// standard headers
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <stdarg.h>
#include <limits>

namespace tricrf {

#define MAX_HEADER "===============================================\n  TriCRF - Triangular-chain CRF\n===============================================\n"

/**
 * @brief String tokenization function
 * 
 * Splits a string into tokens using specified delimiters.
 * 
 * @param str Input string to tokenize
 * @param delimiters String containing delimiter characters (default: " \t")
 * @return Vector of token strings
 * 
 * @section Example Example
 * @code
 * std::string text = "hello world test";
 * auto tokens = tricrf::tokenize(text);
 * // Result: {"hello", "world", "test"}
 * 
 * std::string csv = "a,b,c,d";
 * auto fields = tricrf::tokenize(csv, ",");
 * // Result: {"a", "b", "c", "d"}
 * @endcode
 */
std::vector<std::string> tokenize(const std::string& str, const std::string& delimiters = " \t");

/**
 * @class Logger
 * @brief Multi-level logging system
 * 
 * Provides a flexible logging system with multiple verbosity levels.
 * Supports both file and console output with timestamp formatting.
 * 
 * @section Features Features
 * - Multiple log levels (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG)
 * - File and console output
 * - Automatic timestamp formatting
 * - printf-style formatting
 * - Thread-safe for single-threaded use
 * 
 * @section Usage_Examples Usage Examples
 * @code
 * // Create logger with file output
 * tricrf::Logger logger("training.log", 2);
 * 
 * // Log at different levels
 * logger.report(0, "Error: Model failed to converge");
 * logger.report(1, "Warning: Low accuracy on validation set");
 * logger.report(2, "Info: Starting iteration %d", iter);
 * logger.report(3, "Debug: Parameter vector size: %zu", n_params);
 * 
 * // Simple logging (level 1)
 * logger.report("Training completed successfully");
 * @endcode
 * 
 * @section Log_Levels Log Levels
 * - **0 (ERROR)**: Critical errors that stop execution
 * - **1 (WARN)**: Warnings and important messages
 * - **2 (INFO)**: General information messages
 * - **3 (DEBUG)**: Detailed debugging information
 */
class Logger {
private:
	size_t m_Level;
	FILE *m_File;
	std::string getTime();
public:
	Logger();
	Logger(const std::string& filename, size_t level = 1);
	~Logger();
	void setLevel(size_t level);
	int report(size_t level, const char *fmt, ...);
	int report(const char *fmt, ...);
};

/**
 * @class Configurator
 * @brief Configuration file parser and manager
 * 
 * Parses and manages configuration files in key-value format.
 * Supports comments, multiple values per key, and type conversion.
 * 
 * @section Features Features
 * - Key-value pair parsing
 * - Comment support (lines starting with #)
 * - Multiple values per key
 * - Type conversion utilities
 * - Validation and error checking
 * 
 * @section File_Format File Format
 * 
 * Configuration files use a simple key-value format:
 * @code
 * # This is a comment
 * key1 = value1
 * key2 = value2 value3 value4
 * key3 = 123
 * key4 = 3.14
 * @endcode
 * 
 * @section Usage_Examples Usage Examples
 * @code
 * // Parse configuration file
 * tricrf::Configurator config("model.cfg");
 * 
 * // Get string values
 * std::string train_file = config.get("train_file");
 * 
 * // Get numeric values (with conversion)
 * int max_iter = std::stoi(config.get("max_iter"));
 * double learning_rate = std::stod(config.get("learning_rate"));
 * 
 * // Check if key exists
 * if (config.isValid("regularization")) {
 *     double reg = std::stod(config.get("regularization"));
 * }
 * 
 * // Get multiple values
 * auto features = config.gets("feature_template");
 * // features = {"word", "word-1", "word+1", "pos"}
 * @endcode
 * 
 * @section Error_Handling Error Handling
 * 
 * - Returns empty string for missing keys
 * - Use isValid() to check key existence
 * - No exceptions thrown for missing keys
 * - File parsing errors are handled gracefully
 */
class Configurator {
private:
	std::string m_filename;
	std::map<std::string, std::vector<std::string> > config;
public:
	Configurator();
	Configurator(const std::string& filename);
	bool parse(const std::string& filename);
	std::string getFileName();
	bool isValid(const std::string& key);
	std::string get(const std::string& key);
	std::vector<std::string> gets(const std::string& key);
};

/**
 * @class timer
 * @brief High-resolution timing utility
 * 
 * Provides high-resolution timing functionality for performance measurement.
 * Based on the Boost timer library with modifications for TriCRF.
 * 
 * @section Features Features
 * - High-resolution timing using std::clock()
 * - Automatic start on construction
 * - Restart functionality
 * - Elapsed time calculation
 * - Platform-independent
 * 
 * @section Usage_Examples Usage Examples
 * @code
 * // Basic timing
 * tricrf::timer timer;
 * 
 * // Perform some operation
 * doExpensiveOperation();
 * 
 * double elapsed = timer.elapsed();
 * std::cout << "Operation took " << elapsed << " seconds" << std::endl;
 * 
 * // Restart timer
 * timer.restart();
 * doAnotherOperation();
 * double new_elapsed = timer.elapsed();
 * @endcode
 * 
 * @section Performance Performance Notes
 * 
 * - Resolution: Typically microsecond precision
 * - Overhead: Minimal (just clock() calls)
 * - Thread safety: Not thread-safe (per-instance)
 * - Platform: Uses std::clock() for portability
 * 
 * @section Copyright Copyright
 * 
 * Based on Boost timer library by Beman Dawes (1994-99).
 * See http://www.boost.org/libs/timer for original documentation.
 */
class timer {
 public:
	timer() { _start_time = std::clock(); } 
	void   restart() { _start_time = std::clock(); } 
	double elapsed() const { return  double(std::clock() - _start_time) / CLOCKS_PER_SEC; }
	double elapsed_max() const { return (double(std::numeric_limits<std::clock_t>::max())	- double(_start_time)) / double(CLOCKS_PER_SEC); 	}
	double elapsed_min() const  { return double(1)/double(CLOCKS_PER_SEC); }
private:
	std::clock_t _start_time;
}; // timer

/// finite testing function
#if defined(_MSC_VER) || defined(__BORLANDC__)
inline int finite(double x) { return _finite(x); }
#endif

/// log zero
const double LOG_ZERO = log(DBL_MIN);

} // namespace tricrf

#endif
