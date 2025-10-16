/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file MaxEnt.h
 * @brief Maximum Entropy (Log-Linear) Model for Classification
 * 
 * The MaxEnt class implements a maximum entropy (log-linear) model for individual
 * event classification without sequential dependencies. This serves as the base
 * class for all other models in the TriCRF framework.
 * 
 * @author Minwoo Jeong
 * @date 2007-2008 (original implementation)
 * @version 1.0 (CMake conversion 2024)
 * 
 * @section Model_Description Model Description
 * Maximum Entropy models the probability of a label given features using:
 * P(y|x) = (1/Z(x)) * exp(Σ λᵢ * fᵢ(x,y))
 * 
 * Where:
 * - Z(x) is the normalization constant
 * - λᵢ are model parameters
 * - fᵢ(x,y) are feature functions
 * 
 * @section Memory_Management Memory Management
 * - Uses STL containers for automatic memory management
 * - Logger pointer is managed externally (no automatic cleanup)
 * - Parameter vectors use std::vector for RAII
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Basic usage:
 * @code
 * MaxEnt model;
 * model.readTrainData("train.txt");
 * model.initializeModel();
 * model.train(100, 2.0, false);  // 100 iterations, L2 regularization
 * 
 * // Testing
 * model.loadModel("model.bin");
 * model.test("test.txt", "output.txt");
 * @endcode
 * 
 * With custom logger:
 * @code
 * Logger logger("training.log", 2);
 * MaxEnt model(&logger);
 * model.train(200, 1.0, true);  // L1 regularization
 * @endcode
 * 
 * Configuration file usage:
 * @code
 * // config.cfg
 * model_type = MaxEnt
 * train_file = data.txt
 * model_file = model.bin
 * estimation = LBFGS-L2
 * l2_prior = 2.0
 * iter = 100
 * @endcode
 * 
 * @section Data_Format Data Format
 * Expected format (each example separated by blank line):
 * @code
 * LABEL feature1 feature2 feature3
 * LABEL feature1 feature2 feature3
 * 
 * LABEL feature1 feature2 feature3
 * @endcode
 * 
 * @section Performance_Notes Performance Notes
 * - Fastest training and inference among all models
 * - No sequential dependencies (each event independent)
 * - Suitable for tasks where sequence structure is not important
 * - Memory usage: O(V) where V is vocabulary size
 */

#ifndef __MAXENT_H__
#define __MAXENT_H__

/// max headers
#include "Param.h"
#include "Data.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** 
 * @brief Maximum Entropy Model for Individual Event Classification
 * 
 * Base class implementing maximum entropy (log-linear) models.
 * Models the probability of labels given features without sequential dependencies.
 * 
 * @note All other models (CRF, TriCRF1, etc.) inherit from this class
 */
class MaxEnt {
protected:
	/// Data sets
	Data<Sequence> m_TrainSet;	 ///< Train data
	Data<Sequence> m_DevSet;	///< Development data (held-out data)
	std::vector<double> m_TrainSetCount;	///< Counts for data
	std::vector<double> m_DevSetCount;

	/// Parameter vector
	Parameter m_Param;

	/// Logger 
	Logger *logger;
	
	/// Inference
	virtual std::vector<double> evaluate(Event ev, size_t& max_outcome);

	/// Parameter Estimation
	virtual bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1, double eta = 1E-05);

	/// Prune
	/// for pruning
	std::vector<std::pair<long double, size_t> > m_prune;
	long double m_prune_threshold;


public:
	MaxEnt();	 
	MaxEnt(Logger *logger);
	virtual ~MaxEnt();	

	/// Data manipulation
	Event packEvent(std::vector<std::string>& tokens, Parameter* p_Param = NULL, bool test = false);
	Event packEvent2(std::vector<std::string>& tokens, Parameter* p_Param = NULL, bool test = false);
	StringEvent packStringEvent(std::vector<std::string>& tokens, Parameter* p_Param = NULL, bool test = false);
	virtual void readTrainData(const std::string& filename);
	virtual void readDevData(const std::string& filename);
	
	/// Model 
	virtual bool loadModel(const std::string& filename);
	virtual bool saveModel(const std::string& filename);
	virtual bool averageParam() { return true; };

	/// Testing
	virtual bool test(const std::string& filename, const std::string& outputfile = "", bool confidence = false);

	/// Training 
	virtual void clear();
	virtual void initializeModel();
	virtual bool pretrain(size_t max_iter = 100, double sigma = 20, bool L1 = false);
	virtual bool train(size_t max_iter = 100, double sigma = 20, bool L1 = false);

	/// Logger 
	void setLogger(Logger *logger);
	void setPrune(double prune);
	
	Parameter& getParam() { return m_Param; };
};

} // namespace tricrf

#endif
