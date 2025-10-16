/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file Data.h
 * @brief Core data structures for TriCRF framework
 * 
 * This header defines the fundamental data structures used throughout the TriCRF framework:
 * - Event: Individual observation with label and feature vector
 * - StringEvent: String-based features for human-readable processing
 * - Sequence: Linear sequence of events (for CRF models)
 * - TriSequence: Hierarchical sequence with topic and subtopic levels
 * - Data: Template container for managing collections of sequences
 * 
 * @author Minwoo Jeong
 * @date 2007-2008 (original implementation)
 * @version 1.0 (CMake conversion 2024)
 * 
 * @section Memory_Management Memory Management
 * All data structures use STL containers (std::vector, std::map) which provide
 * automatic memory management through RAII. No manual memory allocation/deallocation
 * is required for these structures.
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Creating a simple event:
 * @code
 * Event ev;
 * ev.label = 0;  // Label ID
 * ev.fval = 1.0; // Feature value
 * ev.obs.push_back(std::make_pair(1, 0.5)); // Feature ID, value pair
 * @endcode
 * 
 * Creating a sequence:
 * @code
 * Sequence seq;
 * seq.push_back(ev1);
 * seq.push_back(ev2);
 * @endcode
 * 
 * Creating triangular sequence (hierarchical):
 * @code
 * TriStringSequence triseq;
 * triseq.topic.label = 0;     // Topic label (e.g., "FLIGHT")
 * triseq.topic.fval = 1.0;
 * triseq.seq.push_back(sevent1); // String events for sequence
 * triseq.seq.push_back(sevent2);
 * @endcode
 * 
 * Using Data container:
 * @code
 * Data<Sequence> trainData;
 * trainData.append(seq1);
 * trainData.append(seq2);
 * size_t totalElements = trainData.size_element();
 * @endcode
 * 
 * @section Data_Format Data Format
 * The framework expects data in the following format:
 * - Each example separated by blank lines
 * - First column: class label
 * - For triangular models: First row contains topic information
 * - Subsequent rows: Sequential labeling data with features
 * 
 * Example:
 * @code
 * FLIGHT i wanna go from denver to indianapolis
 * NONE word=i word-1=<s> word+1=wanna
 * FROMLOC.CITY_NAME-B word=denver word-1=from word-2=go
 * TOLOC.CITY_NAME-B word=indianapolis word-1=to word-2=denver
 * 
 * HOTEL book a room in new york
 * NONE word=book word-1=<s> word+1=a
 * CITY_NAME-B word=new word-1=in word-2=room
 * CITY_NAME-I word=york word-1=new word-2=in
 * @endcode
 */

#ifndef __DATA_H__
#define __DATA_H__

/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** 
 * @brief Individual observation with label and feature vector
 * 
 * Represents a single observation in a sequence with:
 * - label: The ground truth or predicted label ID
 * - fval: Feature value (typically 1.0 for binary features)
 * - obs: Vector of (feature_id, feature_value) pairs
 * 
 * @note Uses integer feature IDs for efficiency in large-scale training
 */
struct Event {
	size_t label;                                          ///< Label ID (0-based index)
	double fval;                                           ///< Feature value (usually 1.0)
	std::vector<std::pair<size_t, double> > obs;          ///< Feature vector: (feature_id, value) pairs
};

/** 
 * @brief String-based event with human-readable features
 * 
 * Similar to Event but uses string feature names instead of integer IDs.
 * Useful for debugging and human-readable model inspection.
 * 
 * @note Less memory efficient than Event due to string storage
 */
struct StringEvent {
	size_t label;                                          ///< Label ID (0-based index)
	double fval;                                           ///< Feature value (usually 1.0)
	std::vector<std::pair<std::string, double > > obs;    ///< Feature vector: (feature_name, value) pairs
};

/** 
 * @brief Linear sequence of events (for CRF models)
 * 
 * Standard sequence representation for linear-chain CRF models.
 * Each element is an Event with integer feature IDs.
 */
typedef std::vector<Event> Sequence;

/** 
 * @brief String-based sequence (for debugging/inspection)
 * 
 * Sequence with string feature names for human readability.
 * Used in triangular models for better interpretability.
 */
typedef std::vector<StringEvent> StringSequence;

/** 
 * @brief Hierarchical sequence with topic and subtopic levels
 * 
 * Represents a two-level hierarchical structure:
 * - topic: High-level semantic meaning (e.g., dialogue act)
 * - seq: Fine-grained sequence within the topic
 * 
 * Used by TriCRF2 model with integer features.
 */
class TriSequence {
public:
	Event topic;                                           ///< Topic-level event
	Sequence seq;                                          ///< Sequence of events within topic
	size_t size() { return seq.size(); };                 ///< Get sequence length
};

/** 
 * @brief String-based hierarchical sequence
 * 
 * Hierarchical structure with string features for better interpretability.
 * Used by TriCRF1 and TriCRF3 models.
 */
class TriStringSequence {
public:
	Event topic;                                           ///< Topic-level event
	StringSequence seq;                                    ///< String-based sequence within topic
	size_t size() { return seq.size(); };                 ///< Get sequence length
};

/** 
 * @brief Template container for managing collections of sequences
 * 
 * Extends std::vector to provide additional functionality for sequence data:
 * - Automatic element counting across all sequences
 * - Efficient appending with size tracking
 * - Memory management through STL containers
 * 
 * @tparam T Sequence type (Sequence, StringSequence, TriSequence, etc.)
 * 
 * @section Memory_Management Memory Management
 * Uses STL containers with automatic memory management. No manual cleanup required.
 * 
 * @section Usage_Example Usage Example
 * @code
 * Data<Sequence> trainData;
 * trainData.append(seq1);  // Adds sequence and updates element count
 * trainData.append(seq2);
 * 
 * size_t totalSequences = trainData.size();      // Number of sequences
 * size_t totalElements = trainData.size_element(); // Total events across all sequences
 * @endcode
 */
template <typename T = Sequence>
class Data : public std::vector<T> {
private:
	size_t n_element;                                      ///< Total number of elements across all sequences
	
public:
	/** 
	 * @brief Append a sequence and update element count
	 * @param element The sequence to append
	 */
	void append(T element) { 
		this->push_back(element); 
		n_element += element.size(); 
	}
	
	/** 
	 * @brief Get total number of elements across all sequences
	 * @return Total count of individual events
	 */
	size_t size_element() { 
		return n_element; 
	}
};

} // namespace tricrf

#endif

