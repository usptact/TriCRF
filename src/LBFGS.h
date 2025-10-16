/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/**
 * @file LBFGS.h
 * @brief Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) Optimization Algorithm
 * 
 * This file implements the L-BFGS optimization algorithm for parameter estimation in
 * Conditional Random Fields and related models. L-BFGS is a quasi-Newton method that
 * approximates the inverse Hessian matrix using limited memory, making it suitable for
 * large-scale optimization problems.
 * 
 * @section Overview Overview
 * 
 * The L-BFGS algorithm is particularly well-suited for training CRF models because:
 * - It handles large parameter spaces efficiently (hundreds of thousands of parameters)
 * - It converges faster than gradient descent methods
 * - It uses limited memory compared to full Newton methods
 * - It supports L1 regularization for feature selection
 * 
 * @section Algorithm Algorithm Details
 * 
 * L-BFGS maintains a limited history of gradient and parameter updates to approximate
 * the inverse Hessian matrix. This allows it to take more intelligent steps toward
 * the optimum while using only O(m*n) memory where m is the history size and n is
 * the number of parameters.
 * 
 * @section Usage_Examples Usage Examples
 * 
 * Basic optimization loop:
 * @code
 * #include "LBFGS.h"
 * 
 * tricrf::LBFGS optimizer;
 * 
 * // Initialize parameters
 * std::vector<double> theta(n_params, 0.0);
 * std::vector<double> gradient(n_params);
 * 
 * // Optimization loop
 * for (int iter = 0; iter < max_iterations; ++iter) {
 *     // Compute objective function value and gradient
 *     double objective = computeObjective(theta);
 *     computeGradient(theta, gradient);
 *     
 *     // Call L-BFGS optimizer
 *     int result = optimizer.optimize(n_params, theta.data(), objective, 
 *                                   gradient.data(), use_l1, l1_penalty);
 *     
 *     if (result == 0) {
 *         std::cout << "Converged after " << iter << " iterations" << std::endl;
 *         break;
 *     } else if (result < 0) {
 *         std::cerr << "Optimization failed" << std::endl;
 *         break;
 *     }
 *     // Continue with next iteration
 * }
 * @endcode
 * 
 * With L1 regularization for feature selection:
 * @code
 * // Enable L1 regularization with penalty C = 1.0
 * int result = optimizer.optimize(n_params, theta.data(), objective,
 *                               gradient.data(), true, 1.0);
 * @endcode
 * 
 * @section Parameters Parameters
 * 
 * - **size**: Number of parameters to optimize
 * - **x**: Parameter vector (input/output)
 * - **f**: Current objective function value
 * - **g**: Gradient vector (input)
 * - **orthant**: Whether to use L1 regularization
 * - **C**: L1 penalty parameter (only used if orthant=true)
 * 
 * @section Memory_Management Memory Management
 * 
 * The LBFGS class automatically manages internal memory:
 * - Memory is allocated on first call to optimize()
 * - Memory is cleared when optimization terminates (result=0)
 * - Memory is reused across multiple optimization runs
 * - Memory is freed in destructor
 * 
 * @section Performance Performance Notes
 * 
 * - Memory usage: O(m*n) where m=100 (history size) and n=number of parameters
 * - Typical convergence: 10-100 iterations depending on problem size
 * - Best for problems with 1000+ parameters
 * - Supports both L1 and L2 regularization
 * 
 * @section Error_Handling Error Handling
 * 
 * The optimize() method returns:
 * - **1**: Continue optimization (evaluate f and g again)
 * - **0**: Optimization converged successfully
 * - **-1**: Optimization failed (check error messages)
 * 
 * @author Minwoo Jeong
 * @date 2010
 * @version 1.0
 */

#ifndef __LBFGS_H_
#define __LBFGS_H_

#include <vector>
#include <iostream>

namespace tricrf {
  // helper functions defined in the paper
  inline double sigma(double x) {
    if (x > 0) return 1.0;
    else if (x < 0) return -1.0;
    return 0.0;
  }

  /**
   * @class LBFGS
   * @brief Limited-memory BFGS optimization algorithm implementation
   * 
   * This class provides an efficient implementation of the L-BFGS algorithm for
   * large-scale unconstrained optimization problems. It's particularly well-suited
   * for training machine learning models with many parameters.
   * 
   * @section Features Features
   * - Memory-efficient: Uses O(m*n) memory instead of O(n²)
   * - Fast convergence: Typically 10-100 iterations
   * - L1 regularization support for feature selection
   * - Automatic memory management
   * - Thread-safe for single-threaded use
   * 
   * @section Sample_Data Sample Data
   * 
   * Typical parameter vector for CRF training:
   * @code
   * // Example: 1000 features, 10 labels
   * size_t n_features = 1000;
   * size_t n_labels = 10;
   * size_t n_params = n_features * n_labels;  // 10,000 parameters
   * 
   * std::vector<double> theta(n_params, 0.0);  // Initialize to zero
   * std::vector<double> gradient(n_params);    // Will be computed
   * 
   * // Example objective function value (negative log-likelihood)
   * double objective = 1234.56;  // Computed from training data
   * @endcode
   * 
   * @section Memory_Usage Memory Usage
   * 
   * For a problem with n parameters:
   * - Internal storage: ~201*n + 200 doubles
   * - History size: 100 (fixed)
   * - Total memory: ~1.6KB per 1000 parameters
   * 
   * Example memory usage:
   * - 10,000 parameters: ~16KB
   * - 100,000 parameters: ~160KB
   * - 1,000,000 parameters: ~1.6MB
   */
  class LBFGS {
  private:
    class Mcsrch;  ///< Line search algorithm implementation
    int iflag_, iscn, nfev, iycn, point, npt, iter, info, ispt, isyt, iypt, maxfev;
    double stp, stp1;  ///< Step size parameters
    std::vector <double> diag_;  ///< Diagonal approximation of inverse Hessian
    std::vector <double> w_;     ///< Working array for L-BFGS updates
    Mcsrch *mcsrch_;  ///< Line search algorithm instance

    void lbfgs_optimize(int size,
                        int msize,
                        double *x,
                        double f,
                        const double *g,
                        double *diag,
                        double *w, bool orthant, double C, int *iflag);

  public:
    /**
     * @brief Default constructor
     * 
     * Initializes all internal variables to zero. Memory allocation
     * is deferred until the first call to optimize().
     */
    explicit LBFGS(): iflag_(0), iscn(0), nfev(0), iycn(0),
                      point(0), npt(0), iter(0), info(0),
                      ispt(0), isyt(0), iypt(0), maxfev(0),
                      stp(0.0), stp1(0.0), mcsrch_(0) {}
    
    /**
     * @brief Destructor
     * 
     * Automatically cleans up allocated memory.
     */
    virtual ~LBFGS() { clear(); }

    /**
     * @brief Clear internal state and free memory
     * 
     * Resets the optimizer to its initial state and frees
     * all allocated memory. Safe to call multiple times.
     */
    void clear();

    /**
     * @brief Perform one L-BFGS optimization step
     * 
     * This is the main optimization method. It should be called repeatedly
     * in a loop until convergence (return value 0) or failure (return value -1).
     * 
     * @param size Number of parameters to optimize
     * @param x Parameter vector (input/output) - will be updated
     * @param f Current objective function value (input)
     * @param g Gradient vector (input) - must be computed
     * @param orthant Whether to use L1 regularization (orthant projection)
     * @param C L1 penalty parameter (only used if orthant=true)
     * 
     * @return Optimization status:
     *         - 1: Continue optimization (evaluate f and g again)
     *         - 0: Optimization converged successfully
     *         - -1: Optimization failed (check error messages)
     * 
     * @section Example_Usage Example Usage
     * @code
     * tricrf::LBFGS optimizer;
     * 
     * // Initialize
     * std::vector<double> theta(1000, 0.0);
     * std::vector<double> gradient(1000);
     * 
     * // Optimization loop
     * for (int iter = 0; iter < 100; ++iter) {
     *     double objective = computeObjective(theta);
     *     computeGradient(theta, gradient);
     *     
     *     int result = optimizer.optimize(1000, theta.data(), objective,
     *                                   gradient.data(), false, 0.0);
     *     
     *     if (result == 0) {
     *         std::cout << "Converged!" << std::endl;
     *         break;
     *     } else if (result < 0) {
     *         std::cerr << "Failed!" << std::endl;
     *         break;
     *     }
     * }
     * @endcode
     */
    int optimize(size_t size, double *x, double f, double *g, bool orthant, double C) {
      // increase this parameter to use more memory but also cut down the number of training iterations
      static const int msize = 100;
      if (w_.empty()) {
        iflag_ = 0;
        w_.resize(size * (2 * msize + 1) + 2 * msize);
        diag_.resize(size);
      } else if (diag_.size() != size) {
        std::cerr << "size of array is different" << std::endl;
        return -1;
      }

      lbfgs_optimize(static_cast<int>(size),
                      msize, x, f, g, &diag_[0], &w_[0], orthant, C, &iflag_);

      if (iflag_ < 0) {
        std::cerr << "routine stops with unexpected error" << std::endl;
        return -1;
      }

      if (iflag_ == 0) {
        clear();
        return 0;   // terminate
      }

      return 1;   // evaluate next f and g
    }
  };
}

#endif
