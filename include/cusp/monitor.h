/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file monitor.h
 *  \brief Monitor iterative solver convergence
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/blas.h>

#include <limits>
#include <iostream>
#include <iomanip>

// Classes to monitor iterative solver progress, check for convergence, etc.
// Follows the implementation of Iteration in the ITL:
//   http://www.osl.iu.edu/research/itl/doc/Iteration.html

namespace cusp
{
/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup monitors Monitors
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p default_monitor : Implements standard convergence criteria
 * and reporting for iterative solvers.
 *
 * \tparam Real real-valued type (e.g. \c float or \c double).
 *
 *  The following code snippet demonstrates how to configure
 *  the \p default_monitor and use it with an iterative solver.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/krylov/cg.h>
 *  #include <cusp/gallery/poisson.h>
 *  
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *  
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *  
 *      // allocate storage for solution (x) and right hand side (b)
 *      cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *      cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *  
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      cusp::default_monitor<float> monitor(b, 100, 1e-6);
 *  
 *      // solve the linear system A x = b
 *      cusp::krylov::cg(A, x, b, monitor);
 *  
 *      // report solver results
 *      if (monitor.converged())
 *      {
 *          std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
 *          std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl;
 *      }
 *      else
 *      {
 *          std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
 *          std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl;
 *      }
 *  
 *      return 0;
 *  }
 *  \endcode
 *
 *  \see \p verbose_monitor
 *
 */
template <typename Real>
class default_monitor
{
    public:
    /*! Construct a \p default_monitor for a given right-hand-side \p b
     *
     *  The \p default_monitor terminates iteration when the residual norm
     *  satisfies the condition
     *       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
     *  or when the iteration limit is reached.
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *
     *  \tparam VectorType vector
     */
    template <typename Vector>
    default_monitor(const Vector& b, size_t iteration_limit = 500, Real relative_tolerance = 1e-5, Real absolute_tolerance = 0)
        : b_norm(cusp::blas::nrm2(b)),
          r_norm(std::numeric_limits<Real>::max()),
          iteration_limit_(iteration_limit),
          iteration_count_(0),
          relative_tolerance_(relative_tolerance),
          absolute_tolerance_(absolute_tolerance)
    {}

    /*! increment the iteration count
     */
    void operator++(void) {  ++iteration_count_; } // prefix increment

    /*! applies convergence criteria to determine whether iteration is finished
     *
     *  \param r residual vector of the linear system (r = b - A x)
     *  \tparam Vector vector
     */
    template <typename Vector>
    bool finished(const Vector& r)
    {
        r_norm = cusp::blas::nrm2(r);
        
        return converged() || iteration_count() >= iteration_limit();
    }
   
    /*! whether the last tested residual satifies the convergence tolerance
     */
    bool converged() const
    {
        return residual_norm() <= tolerance();
    }

    /*! Euclidean norm of last residual
     */
    Real residual_norm() const { return r_norm; }

    /*! number of iterations
     */
    size_t iteration_count() const { return iteration_count_; }

    /*! maximum number of iterations
     */
    size_t iteration_limit() const { return iteration_limit_; }

    /*! relative tolerance
     */
    Real relative_tolerance() const { return relative_tolerance_; }
    
    /*! absolute tolerance
     */
    Real absolute_tolerance() const { return absolute_tolerance_; }
   
    /*! tolerance
     *
     *  Equal to absolute_tolerance() + relative_tolerance() * ||b||
     *
     */ 
    Real tolerance() const { return absolute_tolerance() + relative_tolerance() * b_norm; }

    protected:
    
    Real r_norm;
    Real b_norm;
    Real relative_tolerance_;
    Real absolute_tolerance_;

    size_t iteration_limit_;
    size_t iteration_count_;
};

/*! \p verbose_monitor is similar to \p default monitor except that
 * it displays the solver status during iteration and reports a 
 * summary after iteration has stopped.
 *
 * \tparam Real real-valued type (e.g. \c float or \c double).
 *
 * \see \p default_monitor
 */
template <typename Real>
class verbose_monitor : public default_monitor<Real>
{
    typedef cusp::default_monitor<Real> super;

    public:
    /*! Construct a \p verbose_monitor for a given right-hand-side \p b
     *
     *  The \p verbose_monitor terminates iteration when the residual norm
     *  satisfies the condition
     *       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
     *  or when the iteration limit is reached.
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *
     *  \tparam VectorType vector
     */
    template <typename Vector>
    verbose_monitor(const Vector& b, size_t iteration_limit = 500, Real relative_tolerance = 1e-5, Real absolute_tolerance = 0)
        : super(b, iteration_limit, relative_tolerance, absolute_tolerance)
    {
        std::cout << "Solver will continue until ";
        std::cout << "residual norm " << super::tolerance() << " or reaching ";
        std::cout << super::iteration_limit() << " iterations " << std::endl;
        std::cout << "  Iteration Number  | Residual Norm" << std::endl;
    }
    
    template <typename Vector>
    bool finished(const Vector& r)
    {
        super::r_norm = cusp::blas::nrm2(r);

        std::cout << "       "  << std::setw(10) << super::iteration_count();
        std::cout << "       "  << std::setw(10) << std::scientific << super::residual_norm() << std::endl;

        if (super::converged())
        {
            std::cout << "Successfully converged after " << super::iteration_count() << " iterations." << std::endl;
            return true;
        }
        else if (super::iteration_count() >= super::iteration_limit())
        {
            std::cout << "Failed to converge after " << super::iteration_count() << " iterations." << std::endl;
            return true;
        }
        else
        {
            return false;
        }
    }
};
/*! \}
 */


/*! \p convergence_monitor is similar to \p default monitor except that
 * it displays the solver status during iteration and reports a 
 * summary after iteration has stopped.
 *
 * \tparam Real real-valued type (e.g. \c float or \c double).
 *
 * \see \p default_monitor
 */
template <typename Real>
class convergence_monitor : public default_monitor<Real>
{
    typedef cusp::default_monitor<Real> super;

    public:
    /*! Construct a \p convergence_monitor for a given right-hand-side \p b
     *
     *  The \p convergence_monitor terminates iteration when the residual norm
     *  satisfies the condition
     *       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
     *  or when the iteration limit is reached.
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *
     *  \tparam VectorType vector
     */

    cusp::array1d<Real,cusp::host_memory> residuals;

    template <typename Vector>
    convergence_monitor(const Vector& b, size_t iteration_limit = 500, Real relative_tolerance = 1e-5, Real absolute_tolerance = 0)
        : super(b, iteration_limit, relative_tolerance, absolute_tolerance)
    {
	residuals.reserve(iteration_limit);
    }
    
    template <typename Vector>
    bool finished(const Vector& r)
    {
        super::r_norm = cusp::blas::nrm2(r);
	residuals.push_back(super::r_norm);

        return super::converged() || super::iteration_count() >= super::iteration_limit();
    }

    void print(void)
    {
        std::cout << "Solver will continue until ";
        std::cout << "residual norm " << super::tolerance() << " or reaching ";
        std::cout << super::iteration_limit() << " iterations " << std::endl;

        std::cout << "Ran " << super::iteration_count();
	std::cout << " iterations with a final residual of ";
	std::cout << super::r_norm << std::endl;

	std::cout << "geometric convergence factor : " << geometric_rate() << std::endl;
	std::cout << "immediate convergence factor : " << immediate_rate() << std::endl;
	std::cout << "average convergence factor   : " << average_rate() << std::endl;
    }

    Real immediate_rate(void)
    {
	size_t num = residuals.size();	
	return residuals[num-1] / residuals[num-2]; 
    }

    Real geometric_rate(void)
    {
	size_t num = residuals.size();	
	return std::pow(residuals[num-1] / residuals[0], Real(1.0)/num); 
    }

    Real average_rate(void)
    {
	size_t num = residuals.size();	
    	cusp::array1d<Real,cusp::host_memory> avg_vec(num-1);
	thrust::transform(residuals.begin() + 1, residuals.end(), residuals.begin(), avg_vec.begin(), thrust::divides<Real>());
  	Real sum = thrust::reduce(avg_vec.begin(), avg_vec.end(), Real(0), thrust::plus<Real>());
	return sum / Real(avg_vec.size());
    }
};
/*! \}
 */

} // end namespace cusp

