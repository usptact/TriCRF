/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file TriCRF1.h
 * @brief Triangular-chain Conditional Random Fields (Model 1) - String Feature Model
 * 
 * TriCRF1 implements the first variant of triangular-chain CRF models with full
 * string feature support and hierarchical structure. This model provides a good
 * balance between interpretability and performance for spoken language understanding
 * tasks with moderate complexity.
 * 
 * @author Minwoo Jeong
 * @date 2007-2008 (original implementation)
 * @version 1.0 (CMake conversion 2024)
 * 
 * @section Model_Description Model Description
 * TriCRF1 models hierarchical sequences with two levels:
 * 1. **Topic Level**: High-level semantic meaning (e.g., "FLIGHT", "HOTEL")
 * 2. **Sequence Level**: Fine-grained labels within topics (e.g., "CITY_NAME-B", "DATE-B")
 * 
 * The model uses separate parameter sets for each topic, allowing topic-specific
 * learning while maintaining string-based features for better interpretability.
 * 
 * Mathematical formulation:
 * P(y,z|x) = (1/Z(x)) * exp(Σ λᵗₒₚᵢc * fᵗₒₚᵢc(x,z) + Σ λᶻₛₑᵩ * fᶻₛₑᵩ(x,y,z))
 * 
 * Where:
 * - z is the topic variable
 * - y is the sequence label variable
 * - λᵗₒₚᵢc are topic-specific parameters
 * - λᶻₛₑᵩ are sequence parameters for topic z
 * - All features are string-based for human readability
 * 
 * @section Memory_Management Memory Management
 * - Uses STL containers for automatic memory management
 * - Dynamic parameter allocation per topic (managed via std::vector<Parameter>)
 * - Large matrices (M, R, Alpha, Beta) use std::vector for RAII
 * - Memory usage: O(T*V + T*S²) where T=topics, V=vocabulary, S=sequence states
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Basic training and testing:
 * @code
 * TriCRF1 model;
 * model.readTrainData("train.txt");
 * model.initializeModel();
 * model.train(100, 2.0, false);  // 100 iterations, L2 regularization
 * model.saveModel("model.bin");
 * 
 * // Testing
 * model.loadModel("model.bin");
 * model.test("test.txt", "output.txt");
 * @endcode
 * 
 * Configuration file usage:
 * @code
 * // config.cfg
 * model_type = TriCRF1
 * mode = both
 * train_file = train.txt
 * test_file = test.txt
 * model_file = model.bin
 * estimation = LBFGS-L2
 * l2_prior = 2.0
 * iter = 100
 * initialize = PL
 * initialize_iter = 30
 * @endcode
 * 
 * @section Data_Format Data Format
 * Expected hierarchical format:
 * @code
 * FLIGHT i wanna go from denver to indianapolis on november eighteenth
 * NONE word=i word-1=<s> word+1=wanna word+2=go
 * FROMLOC.CITY_NAME-B word=denver word-1=from word-2=go word+1=to word+2=indianapolis
 * TOLOC.CITY_NAME-B word=indianapolis word-1=to word-2=denver word+1=on word+2=november
 * MONTH_NAME-B word=november word-1=on word-2=indianapolis word+1=eighteenth word+2=</s>
 * DAY_NUMBER-B word=eighteenth word-1=november word-2=on word+1=</s>
 * 
 * HOTEL book a room in new york for tomorrow
 * NONE word=book word-1=<s> word+1=a word+2=room
 * CITY_NAME-B word=new word-1=in word-2=room word+1=york word+2=for
 * CITY_NAME-I word=york word-1=new word-2=in word+1=for word+2=tomorrow
 * DATE-B word=tomorrow word-1=for word-2=york word+1=</s>
 * @endcode
 * 
 * @section Performance_Notes Performance Notes
 * - Good balance of accuracy and interpretability
 * - String features provide better debugging capabilities
 * - Training time: O(T*N*S²) where N=training examples
 * - Inference time: O(T*S²) per sequence
 * - Memory usage scales with number of topics and vocabulary size
 * 
 * @section Comparison_with_Other_Models Comparison with Other Models
 * - **vs MaxEnt**: Adds sequential dependencies and hierarchical structure
 * - **vs CRF**: Adds topic-level modeling for better semantic understanding
 * - **vs TriCRF2**: Uses string features vs integer features (better interpretability)
 * - **vs TriCRF3**: Simpler parameter management, slightly less efficient
 * 
 * @section Key_Features Key Features
 * - **String-based features**: Human-readable feature names for debugging
 * - **Hierarchical modeling**: Two-level structure (topic + sequence)
 * - **Topic-specific parameters**: Separate learning for each topic
 * - **Forward-backward inference**: Efficient probability computation
 * - **Viterbi decoding**: Optimal sequence prediction
 */

#ifndef __TRICRF1_H__
#define __TRICRF1_H__

/// max headers
#include "CRF.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** 
 * @brief Triangular-chain Conditional Random Fields (Model 1) - String Feature Model
 * 
 * The first variant of triangular CRF models, combining hierarchical structure
 * with string-based features for better interpretability and debugging.
 * 
 * @note Good choice for development and debugging due to string features
 */
class TriCRF1 : public CRF {
protected:
	/// Data sets
	Data<TriStringSequence> m_TrainSet;	 ///< Training data with string features
	Data<TriStringSequence> m_DevSet;	///< Development data (held-out data)
	std::vector<std::vector<TriSequence> > m_TrainLabelSet; ///< Label sequences for training
	
	/// Forward-backward algorithm matrices
	std::vector<std::vector<long double> > m_M;			///< Edge transition matrix (topic-specific)
	std::vector<std::vector<long double> > m_R;			///< Node observation matrix (topic-specific)
	std::vector<std::vector<long double> > m_Alpha;	///< Forward probabilities (alpha values)
	std::vector<std::vector<long double> > m_Beta;		///< Backward probabilities (beta values)
	std::vector<long double> m_Gamma;			///< Topic prior probabilities
	std::vector<long double> m_Z;			///< Normalization constants (partition functions)	

	/// Model parameters
	std::vector<Parameter> m_ParamSeq;	///< Sequence parameters (one per topic)
	Parameter m_ParamTopic;				///< Topic-level parameters
	std::map<std::pair<size_t, size_t>, size_t> m_Mapping;	///< Topic-sequence label mapping
	std::map<std::pair<size_t, size_t>, size_t> m_RMapping;	///< Reverse mapping for efficiency

	/// Computational variables
	size_t m_topic_size;				///< Number of topics in the model
	std::vector<size_t> m_state_size;	///< Number of states per topic
	size_t m_state_size2;				///< Total number of sequence states

	/// Inference algorithms
	void calculateFactors(TriStringSequence &seq);	///< Calculate feature factors for given sequence
	void calculateEdge() override;					///< Compute edge transition probabilities
	void forward() override;						///< Forward recursion for probability computation
	void backward() override;						///< Backward recursion for probability computation
	long double getPartitionZ() override;			///< Calculate partition function (normalization constant)
	long double calculateProb(TriStringSequence& seq);	///< Calculate sequence probability P(y|x)
	std::vector<size_t> viterbiSearch(size_t& max_z, long double& prob);	///< Viterbi algorithm for best path

	/// Parameter estimation methods
	bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);	///< LBFGS optimization
	bool estimateWithPL(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);		///< Pseudo-likelihood estimation

public:
	/** @brief Default constructor */
	TriCRF1();
	
	/** @brief Constructor with logger
	 * @param logger Pointer to logger instance for output control
	 */
	TriCRF1(Logger *logger);
	
	/// Data manipulation
	/** @brief Load training data from file
	 * @param filename Path to training data file
	 * @note Data format: hierarchical with topic and sequence labels
	 */
	void readTrainData(const std::string& filename);
	
	/** @brief Load development data from file
	 * @param filename Path to development data file
	 * @note Used for validation during training
	 */
	void readDevData(const std::string& filename);

	/// Model persistence
	/** @brief Load trained model from file
	 * @param filename Path to model file
	 * @return true if successful, false otherwise
	 */
	bool loadModel(const std::string& filename);
	
	/** @brief Save trained model to file
	 * @param filename Path to output model file
	 * @return true if successful, false otherwise
	 */
	bool saveModel(const std::string& filename);

	/// Training methods
	/** @brief Clear all model data and parameters */
	void clear();
	
	/** @brief Initialize model parameters and data structures */
	void initializeModel();
	
	/** @brief Pre-train model using pseudo-likelihood
	 * @param max_iter Maximum number of iterations
	 * @param sigma Regularization parameter
	 * @param L1 Use L1 regularization if true, L2 otherwise
	 * @return true if successful, false otherwise
	 */
	bool pretrain(size_t max_iter = 100, double sigma = 20, bool L1 = false); 
	
	/** @brief Train model using LBFGS optimization
	 * @param max_iter Maximum number of iterations
	 * @param sigma Regularization parameter
	 * @param L1 Use L1 regularization if true, L2 otherwise
	 * @return true if successful, false otherwise
	 */
	bool train(size_t max_iter = 100, double sigma = 20, bool L1 = false); 

	/// Testing and evaluation
	/** @brief Test model on new data
	 * @param filename Path to test data file
	 * @param outputfile Path to output file (optional)
	 * @param confidence Include confidence scores if true
	 * @return true if successful, false otherwise
	 */
	bool test(const std::string& filename, const std::string& outputfile = "", bool confidence = false);	
	
	/// Parameter access
	/** @brief Get topic-level parameters
	 * @return Reference to topic parameter object
	 */
	Parameter& getTopicParam() { return m_ParamTopic; };
	
	/** @brief Get sequence-level parameters
	 * @return Reference to vector of sequence parameter objects (one per topic)
	 */
	std::vector<Parameter>& getSeqParam() { return m_ParamSeq; };

};	///< TriCRF1


} // namespace tricrf

#endif
