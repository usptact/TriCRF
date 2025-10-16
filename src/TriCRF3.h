/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file TriCRF3.h
 * @brief Triangular-chain Conditional Random Fields (Model 3) - Most Advanced Model
 * 
 * TriCRF3 implements the most sophisticated triangular-chain CRF model with full
 * string feature support and hierarchical structure. This is the flagship model
 * used in the original research papers and provides the best performance for
 * complex spoken language understanding tasks.
 * 
 * @author Minwoo Jeong
 * @date 2007-2008 (original implementation)
 * @version 1.0 (CMake conversion 2024)
 * 
 * @section Model_Description Model Description
 * TriCRF3 models hierarchical sequences with two levels:
 * 1. **Topic Level**: High-level semantic meaning (e.g., "FLIGHT", "HOTEL")
 * 2. **Sequence Level**: Fine-grained labels within topics (e.g., "CITY_NAME-B", "DATE-B")
 * 
 * The model uses separate parameter sets for each topic, allowing topic-specific
 * learning while sharing common features across domains.
 * 
 * Mathematical formulation:
 * P(y,z|x) = (1/Z(x)) * exp(Σ λᵗₒₚᵢc * fᵗₒₚᵢc(x,z) + Σ λᶻₛₑᵩ * fᶻₛₑᵩ(x,y,z))
 * 
 * Where:
 * - z is the topic variable
 * - y is the sequence label variable
 * - λᵗₒₚᵢc are topic-specific parameters
 * - λᶻₛₑᵩ are sequence parameters for topic z
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
 * TriCRF3 model;
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
 * model_type = TriCRF3
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
 * - Most accurate model for complex SLU tasks
 * - Computationally intensive due to hierarchical structure
 * - Training time: O(T*N*S²) where N=training examples
 * - Inference time: O(T*S²) per sequence
 * - Memory usage scales with number of topics and sequence complexity
 * 
 * @section Comparison_with_Other_Models Comparison with Other Models
 * - **vs MaxEnt**: Adds sequential dependencies and hierarchical structure
 * - **vs CRF**: Adds topic-level modeling for better semantic understanding
 * - **vs TriCRF1**: More efficient parameter management
 * - **vs TriCRF2**: Full string feature support vs integer-only features
 */

#ifndef __TRICRF3_H__
#define __TRICRF3_H__

/// max headers
#include "CRF.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** 
 * @brief Triangular-chain Conditional Random Fields (Model 3) - Most Advanced
 * 
 * The most sophisticated model in the TriCRF framework, combining hierarchical
 * structure with full string feature support. Provides best performance for
 * complex spoken language understanding tasks.
 * 
 * @note This is the recommended model for production use when accuracy is critical
 */
class TriCRF3 : public CRF {
protected:
	/// Data sets
	Data<TriStringSequence> m_TrainSet;	 ///< Train data
	Data<TriStringSequence> m_DevSet;	///< Development data (held-out data)
	std::vector<std::vector<TriSequence> > m_TrainLabelSet;
	
	std::vector<std::vector<long double> > m_M;			///< M matrix ; edge transition 
	std::vector<std::vector<long double> > m_R;			///< R matrix ; node observation
	std::vector<std::vector<long double> > m_Alpha;	///< Alpha matrix
	std::vector<std::vector<long double> > m_Beta;		///< Beta matrix
	std::vector<long double> m_Gamma;			///< Gamma matrix ; topic prior
	std::vector<long double> m_Z;			///< Z matrix ; topic prior	

	/// Parameters
	std::vector<Parameter> m_ParamSeq;
	Parameter m_ParamTopic;
	std::map<std::pair<size_t, size_t>, size_t> m_Mapping;
	std::map<std::pair<size_t, size_t>, size_t> m_RMapping;

	/// Variables for computation
	size_t m_topic_size;
	std::vector<size_t> m_state_size;	 ///< re-definition
	size_t m_state_size2;

	/// Inference
	void calculateFactors(TriStringSequence &seq);	///< Calculating the factors
	void calculateEdge();
	void forward();	 ///< Forward recursion
	void backward();	///< Backward recursion
	long double getPartitionZ();	///< Z
	long double calculateProb(TriStringSequence& seq);	///< Prob(y|x)
	std::vector<size_t> viterbiSearch(size_t& max_z, long double& prob);	///< Find the best path

	/// Parameter Estimation
	bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);
	bool estimateWithPL(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);
	virtual bool averageParam() { return true; };
	
public:
	TriCRF3();
	TriCRF3(Logger *logger);
	
	/// Data manipulation
	void readTrainData(const std::string& filename);
	void readDevData(const std::string& filename);

	/// Model 
	bool loadModel(const std::string& filename);
	bool saveModel(const std::string& filename);

	/// Training 
	void clear();
	void initializeModel();
	bool pretrain(size_t max_iter = 100, double sigma = 20, bool L1 = false);
	bool train(size_t max_iter = 100, double sigma = 20, bool L1 = false); 

	/// Testing
	bool test(const std::string& filename, const std::string& outputfile = "", bool confidence = false);	
	
	Parameter& getTopicParam() { return m_ParamTopic; };
	std::vector<Parameter>& getSeqParam() { return m_ParamSeq; };

};	///< TriCRF3


} // namespace tricrf

#endif
