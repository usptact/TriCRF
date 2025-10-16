/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file TriCRF2.h
 * @brief Triangular-chain Conditional Random Fields (Model 2) - Integer Feature Model
 * 
 * TriCRF2 implements the second variant of triangular-chain CRF models with integer
 * feature support and optimized computational efficiency. This model provides the
 * best performance for large-scale applications where memory and speed are critical.
 * 
 * @author Minwoo Jeong
 * @date 2007-2008 (original implementation)
 * @version 1.0 (CMake conversion 2024)
 * 
 * @section Model_Description Model Description
 * TriCRF2 models hierarchical sequences with two levels:
 * 1. **Topic Level**: High-level semantic meaning (e.g., "FLIGHT", "HOTEL")
 * 2. **Sequence Level**: Fine-grained labels within topics (e.g., "CITY_NAME-B", "DATE-B")
 * 
 * The model uses integer feature IDs for maximum efficiency and includes optimized
 * indexing structures for fast computation during training and inference.
 * 
 * Mathematical formulation:
 * P(y,z|x) = (1/Z(x)) * exp(Σ λᵗₒₚᵢc * fᵗₒₚᵢc(x,z) + Σ λᶻₛₑᵩ * fᶻₛₑᵩ(x,y,z))
 * 
 * Where:
 * - z is the topic variable
 * - y is the sequence label variable
 * - λᵗₒₚᵢc are topic-specific parameters
 * - λᶻₛₑᵩ are sequence parameters for topic z
 * - All features use integer IDs for efficiency
 * 
 * @section Memory_Management Memory Management
 * - Uses STL containers for automatic memory management
 * - Optimized indexing structures for fast lookups
 * - Single parameter set for all topics (shared parameters)
 * - Memory usage: O(V + S²) where V=vocabulary, S=sequence states
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Basic training and testing:
 * @code
 * TriCRF2 model;
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
 * model_type = TriCRF2
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
 * Expected hierarchical format (same as TriCRF1 but with integer features):
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
 * - Fastest training and inference among TriCRF models
 * - Integer features provide maximum computational efficiency
 * - Optimized indexing for fast parameter lookups
 * - Training time: O(N*S²) where N=training examples
 * - Inference time: O(S²) per sequence
 * - Memory usage scales with vocabulary and state space size
 * 
 * @section Comparison_with_Other_Models Comparison with Other Models
 * - **vs MaxEnt**: Adds sequential dependencies and hierarchical structure
 * - **vs CRF**: Adds topic-level modeling for better semantic understanding
 * - **vs TriCRF1**: Uses integer features vs string features (faster but less interpretable)
 * - **vs TriCRF3**: Shared parameters vs topic-specific parameters (faster but less flexible)
 * 
 * @section Key_Features Key Features
 * - **Integer-based features**: Maximum computational efficiency
 * - **Optimized indexing**: Fast parameter lookups during training
 * - **Shared parameters**: Single parameter set for all topics
 * - **Forward-backward inference**: Efficient probability computation
 * - **Viterbi decoding**: Optimal sequence prediction
 * - **Memory efficient**: Minimal memory footprint for large datasets
 */

#ifndef __TRICRF2_H__
#define __TRICRF2_H__

/// max headers
#include "CRF.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** 
 * @brief Triangular-chain Conditional Random Fields (Model 2) - Integer Feature Model
 * 
 * The second variant of triangular CRF models, optimized for maximum computational
 * efficiency using integer features and shared parameters.
 * 
 * @note Best choice for production systems requiring high performance
 */
class TriCRF2 : public CRF {
protected:
	/// Data sets
	Data<TriSequence> m_TrainSet;	 ///< Training data with integer features
	Data<TriSequence> m_DevSet;	///< Development data (held-out data)
	
	/// Forward-backward algorithm matrices
	std::vector<long double> m_Z;			///< Normalization constants (partition functions)
	std::vector<std::vector<long double> > m_Alpha;	///< Forward probabilities (alpha values)
	std::vector<std::vector<long double> > m_Beta;		///< Backward probabilities (beta values)
	std::vector<long double> m_Gamma;			///< Topic prior probabilities

	/// Optimized indexing structures for fast computation
	std::vector<std::vector<size_t> > m_zy_index;	///< Topic-to-sequence state mapping
	std::vector<std::vector<size_t> > m_yz_index;	///< Sequence-to-topic state mapping
	std::vector<size_t> m_zy_size;				///< Size of each topic's state space
	std::vector<std::vector<StateParam> > m_y_state;	///< State parameter mapping
	void createIndex();						///< Build optimized indexing structures

	/// Model parameters (shared across topics for efficiency)
	Parameter m_ParamSeq;		///< Sequence-level parameters (shared)
	Parameter m_ParamTopic;		///< Topic-level parameters

	/// Computational variables
	size_t m_topic_size;		///< Number of topics in the model

	/// Inference algorithms
	void calculateFactors(TriSequence &seq);	///< Calculate feature factors for integer sequence
	void calculateFactors(TriStringSequence &seq);	///< Calculate feature factors for string sequence
	void calculateEdge() override;				///< Compute edge transition probabilities
	void forward() override;					///< Forward recursion for probability computation
	void backward() override;					///< Backward recursion for probability computation
	long double getPartitionZ() override;		///< Calculate partition function (normalization constant)
	long double calculateProb(TriSequence& seq);	///< Calculate sequence probability P(y|x)
	std::vector<size_t> viterbiSearch(size_t& max_z, long double& prob);	///< Viterbi algorithm for best path

	/// Parameter estimation methods
	bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);	///< LBFGS optimization
	bool estimateWithPL(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);		///< Pseudo-likelihood estimation
		
public:
	/** @brief Default constructor */
	TriCRF2();
	
	/** @brief Constructor with logger
	 * @param logger Pointer to logger instance for output control
	 */
	TriCRF2(Logger *logger);
	
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

};	///< TriCRF2

} // namespace tricrf

#endif
