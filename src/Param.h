/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file Param.h
 * @brief Parameter Management for Conditional Random Fields
 * 
 * This file defines the Parameter class and related structures for managing
 * model parameters, feature mappings, and state dictionaries in CRF models.
 * It provides efficient storage and access to the large parameter vectors
 * used in machine learning models.
 * 
 * @section Overview Overview
 * 
 * The Parameter class is the core component for managing:
 * - Model weights (theta parameters)
 * - Feature dictionaries and mappings
 * - State dictionaries and mappings
 * - Parameter indexing for efficient access
 * - Gradient computation and updates
 * - Model serialization (save/load)
 * 
 * @section Data_Structures Data Structures
 * 
 * - **ObsParam**: Observation parameter (feature-label pair)
 * - **StateParam**: State transition parameter (label-label-feature triple)
 * - **Parameter**: Main class for parameter management
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Basic parameter setup:
 * @code
 * #include "Param.h"
 * 
 * tricrf::Parameter param;
 * 
 * // Add states (labels)
 * size_t state_id = param.addNewState("B-PER");
 * size_t state_id2 = param.addNewState("I-PER");
 * 
 * // Add features
 * size_t feat_id = param.addNewObs("word=John");
 * size_t feat_id2 = param.addNewObs("word=Smith");
 * 
 * // Update parameters
 * param.updateParam(state_id, feat_id, 1.0);
 * param.updateParam(state_id2, feat_id2, 1.0);
 * param.endUpdate();
 * 
 * // Get parameter vector
 * double* weights = param.getWeight();
 * size_t n_params = param.size();
 * @endcode
 * 
 * Feature extraction example:
 * @code
 * // Extract features from a sentence
 * std::vector<std::pair<std::string, double>> features;
 * features.push_back({"word=hello", 1.0});
 * features.push_back({"word-1=<s>", 1.0});
 * features.push_back({"word+1=world", 1.0});
 * 
 * // Convert to observation parameters
 * std::vector<tricrf::ObsParam> obs_params = 
 *     param.makeObsIndex(features);
 * @endcode
 * 
 * @section Sample_Data Sample Data
 * 
 * Typical feature set for named entity recognition:
 * @code
 * // Word features
 * "word=John"           // Current word
 * "word-1=<s>"          // Previous word
 * "word+1=Smith"        // Next word
 * "word-2=<s>"          // Word 2 positions back
 * "word+2=is"           // Word 2 positions forward
 * 
 * // Character features
 * "char-1=J"            // First character
 * "char-2=o"            // Second character
 * "char-3=h"            // Third character
 * "char-4=n"            // Fourth character
 * "char+1=S"            // Last character
 * 
 * // Pattern features
 * "pattern=XXXX"        // Character pattern (all caps)
 * "pattern=Xxxx"        // Mixed case pattern
 * "suffix=hn"           // Word suffix
 * "prefix=Jo"           // Word prefix
 * 
 * // Context features
 * "pos=NNP"             // Part-of-speech tag
 * "pos-1=DT"            // Previous POS tag
 * "pos+1=NNP"           // Next POS tag
 * @endcode
 * 
 * State labels for NER:
 * @code
 * "O"                   // Outside entity
 * "B-PER"               // Beginning of person
 * "I-PER"               // Inside person
 * "B-LOC"               // Beginning of location
 * "I-LOC"               // Inside location
 * "B-ORG"               // Beginning of organization
 * "I-ORG"               // Inside organization
 * "B-MISC"              // Beginning of miscellaneous
 * "I-MISC"              // Inside miscellaneous
 * @endcode
 * 
 * @section Memory_Management Memory Management
 * 
 * The Parameter class uses efficient memory management:
 * - Feature and state dictionaries use hash maps for O(1) lookup
 * - Parameter vectors are stored as contiguous arrays
 * - Memory is allocated dynamically based on usage
 * - Automatic cleanup in destructor
 * 
 * @section Performance Performance Notes
 * 
 * - Feature lookup: O(1) average case
 * - Parameter access: O(1) with indexing
 * - Memory usage: O(n_features + n_states + n_params)
 * - Supports millions of features efficiently
 * 
 * @author Minwoo Jeong
 * @date 2010
 * @version 1.0
 */

#ifndef __PARAM_H__
#define __PARAM_H__

/// max headers
#include "Utility.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/**
 * @struct ObsParam
 * @brief Observation parameter structure for feature-label pairs
 * 
 * Represents a single observation parameter in the CRF model.
 * Each parameter corresponds to a feature-label pair and its weight.
 * 
 * @section Members Members
 * - **y**: Label/state ID (index into state dictionary)
 * - **fid**: Feature ID (index into feature dictionary)  
 * - **fval**: Feature value (typically 1.0 for binary features)
 * 
 * @section Example Example
 * @code
 * tricrf::ObsParam param;
 * param.y = 5;        // Label "B-PER"
 * param.fid = 123;    // Feature "word=John"
 * param.fval = 1.0;   // Binary feature
 * @endcode
 */
struct ObsParam {
	size_t y, fid;     ///< Label ID and feature ID
	double fval;       ///< Feature value
};

/**
 * @struct StateParam
 * @brief State transition parameter structure for label-label-feature triples
 * 
 * Represents a state transition parameter in the CRF model.
 * Each parameter corresponds to a transition between two labels
 * with a specific feature.
 * 
 * @section Members Members
 * - **y1**: Previous label ID
 * - **y2**: Current label ID
 * - **fid**: Feature ID
 * - **fval**: Feature value
 * 
 * @section Example Example
 * @code
 * tricrf::StateParam param;
 * param.y1 = 5;       // Previous label "B-PER"
 * param.y2 = 6;       // Current label "I-PER"
 * param.fid = 123;    // Feature "word=Smith"
 * param.fval = 1.0;   // Binary feature
 * @endcode
 */
struct StateParam {
	size_t y1, y2, fid;  ///< Previous label, current label, and feature IDs
	double fval;          ///< Feature value
};

/**
 * @typedef Map
 * @brief String-to-index mapping for dictionaries
 * 
 * Maps string keys (features, states) to integer indices.
 * Used for efficient lookup in feature and state dictionaries.
 */
typedef std::map<std::string, size_t> Map;

/**
 * @typedef Vec
 * @brief Index-to-string mapping for dictionaries
 * 
 * Maps integer indices back to string values.
 * Used for reverse lookup and human-readable output.
 */
typedef std::vector<std::string> Vec;

/**
 * @class Parameter
 * @brief Main parameter management class for CRF models
 * 
 * This class manages all aspects of parameter storage and access in CRF models:
 * - Feature and state dictionaries
 * - Parameter weight vectors
 * - Gradient computation
 * - Model serialization
 * 
 * @section Features Features
 * - Efficient dictionary lookups (O(1) average case)
 * - Dynamic memory management
 * - Support for millions of features
 * - Thread-safe for read operations
 * - Automatic parameter indexing
 * 
 * @section Memory_Layout Memory Layout
 * 
 * The class maintains several key data structures:
 * - **m_Weight**: Parameter weight vector
 * - **m_Gradient**: Gradient vector for optimization
 * - **m_FeatureMap**: String->ID mapping for features
 * - **m_FeatureVec**: ID->String mapping for features
 * - **m_StateMap**: String->ID mapping for states
 * - **m_StateVec**: ID->String mapping for states
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Basic setup and feature extraction:
 * @code
 * tricrf::Parameter param;
 * 
 * // Add states
 * size_t state_o = param.addNewState("O");
 * size_t state_b_per = param.addNewState("B-PER");
 * size_t state_i_per = param.addNewState("I-PER");
 * 
 * // Add features
 * size_t feat_word = param.addNewObs("word=John");
 * size_t feat_cap = param.addNewObs("is_capitalized");
 * 
 * // Create observation parameters
 * std::vector<std::pair<std::string, double>> features;
 * features.push_back({"word=John", 1.0});
 * features.push_back({"is_capitalized", 1.0});
 * 
 * std::vector<tricrf::ObsParam> obs_params = 
 *     param.makeObsIndex(features);
 * 
 * // Update parameters
 * for (const auto& obs : obs_params) {
 *     param.updateParam(state_b_per, obs.fid, obs.fval);
 * }
 * param.endUpdate();
 * @endcode
 * 
 * Model serialization:
 * @code
 * // Save model
 * std::ofstream fout("model.bin", std::ios::binary);
 * param.save(fout);
 * fout.close();
 * 
 * // Load model
 * std::ifstream fin("model.bin", std::ios::binary);
 * param.load(fin);
 * fin.close();
 * @endcode
 */
class Parameter {
protected:
	/// Weight
	size_t n_weight;
	std::vector<double> m_Weight;
	std::vector<double> m_Gradient;
	std::vector<double> m_Count;
	
	/// Dictionary
	Map m_FeatureMap;
	Vec m_FeatureVec;
	Map m_StateMap;
	Vec m_StateVec;
	

	/// Options
	std::string mEDGE;
	size_t m_default_oid;

public:
	/// 
	//std::vector<size_t> m_StateID;

	Parameter();
	~Parameter();

	/// Parameter index
	std::vector<std::vector<std::pair<size_t, size_t> > > m_ParamIndex;

	/// weight vector
	void initialize();
	void initializeGradient();
	void initializeGradient2();
	size_t size(); 
	void clear(bool state = false);

	/// Parameters 
	double* getWeight();
	double* getGradient();
	void setWeight(double* theta);

	std::vector<StateParam> m_StateIndex;
	std::vector<ObsParam> makeObsIndex(std::vector<std::pair<size_t, double> >& obs);
	std::vector<ObsParam> makeObsIndex(std::vector<std::pair<size_t, double> >& obs, std::map<size_t, size_t>& beam);
	std::vector<ObsParam> makeObsIndex(std::vector<std::pair<std::string, double> >& obs);
	int findObs(const std::string& key);
	int findState(const std::string& key);
	size_t getDefaultState();

	/// Dictionary access functions
	size_t sizeFeatureVec();
	size_t sizeStateVec();
	std::pair<Map, Vec> getState();
	//int findState(size_t key); 

	/// Update and test the parameters
	size_t addNewState(const std::string& key);
	size_t addNewObs(const std::string& key);
	size_t updateParam(size_t oid, size_t pid,  double fval = 1.0);
	void endUpdate();
	void makeStateIndex(bool makeIndex = true);
	std::vector<StateParam> makeStateIndex(size_t y1);
	void makeActiveIndex(double eta = 1E-02);

	// for tied potential
	std::vector<StateParam> m_SelectedStateIndex;
	std::vector<StateParam> m_RemainStateIndex;
	void makeTiedPotential(double K);
	std::vector<size_t> remain_fid;
	std::vector<double> remain_count;
	std::vector<std::vector<size_t> > m_SelectedStateList1;
	std::vector<std::vector<size_t> > m_SelectedStateList2;

	/// save and load
	bool save(std::ofstream& f);
	bool load(std::ifstream& f);

	/// Reporting
	void print(Logger *log);
};

} // namespace tricrf

#endif

