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

/*! \file bicgstab_m.h
 *  \brief Multi-mass Biconjugate Gradient stabilized (BiCGstab-M) method
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace krylov
{

// TODO move to bicgstab_m.inl
namespace trans_m
{
  template <typename InputIterator1, typename InputIterator2,
            typename InputIterator3,
	    typename OutputIterator1,
	    typename ScalarType>
  void compute_z_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
		InputIterator2 z_m1_s_b, InputIterator3 sig_b,
		OutputIterator1 z_1_s_b,
		ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0);

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator1,
	    typename ScalarType>
  void compute_b_m(InputIterator1 z_1_s_b, InputIterator1 z_1_s_e,
		InputIterator2 z_0_s_b, OutputIterator1 beta_0_s_b,
		ScalarType beta_0);

  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename ScalarType>
  void compute_z_m(const Array1& z_0_s, const Array2& z_m1_s,
		const Array3& sig, Array4& z_1_s,
		ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0);

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_b_m(const Array1& z_1_s, const Array2& z_0_s,
		Array3& beta_0_s, ScalarType beta_0);

  template <typename InputIterator1, typename InputIterator2,
            typename InputIterator3, typename OutputIterator,
            typename ScalarType>
  void compute_a_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
		InputIterator2 z_1_s_b, InputIterator3 beta_0_s_b,
                OutputIterator alpha_0_s_b,
		ScalarType beta_0, ScalarType alpha_0);

  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename ScalarType>
  void compute_a_m(const Array1& z_0_s, const Array2& z_1_s,
                const Array3& beta_0_s, Array4& alpha_0_s,
		ScalarType beta_0, ScalarType alpha_0);

  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename Array5, typename Array6, typename Array7>
  void compute_x_m(const Array1& beta_0_s, const Array2& chi_0_s,
                const Array3& rho_0_s, const Array4& zeta_1_s,
                const Array5& w_1, const Array6& s_0_s, Array7& x_0_s);
  
  template <typename Array1, typename Array2, typename Array3, typename Array4,
	   typename Array5, typename Array6, typename Array7, typename Array8,
	   typename Array9, typename Array10, typename Array11>
  void compute_s_m(const Array1& beta_0_s, const Array2& chi_0_s,
                const Array3& rho_0_s, const Array4& zeta_0_s,
                const Array5& alpha_1_s, const Array6& rho_1_s,
		const Array7& zeta_1_s,
                const Array8& r_0, Array9& r_1,
                const Array10& w_1, Array11& s_0_s);

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_w_1_m(InputIterator1 r_0_b, InputIterator1 r_0_e,
		    InputIterator2 As_b, OutputIterator w_1_b,
		    const ScalarType beta_0);

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_w_1_m(const Array1& r_0, const Array2& As, Array3& w_1,
		  const ScalarType beta_0);

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_r_1_m(InputIterator1 w_1_b, InputIterator1 w_1_e,
		    InputIterator2 Aw_b, OutputIterator r_1_b,
		    ScalarType chi_0);

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_r_1_m(const Array1& w_1, const Array2& Aw, Array3& r_1,
		  ScalarType chi_0);

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_s_0_m(InputIterator1 r_1_b, InputIterator1 r_1_e,
		    InputIterator2 As_b, OutputIterator s_0_b,
		    ScalarType alpha_1, ScalarType chi_0);

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_s_0_m(const Array1& r_1, const Array2& As, Array3& s_0,
		  ScalarType alpha_1, ScalarType chi_0);

  template <typename InputIterator, typename OutputIterator,
	    typename ScalarType>
  void compute_chi_m(InputIterator sigma_b, InputIterator sigma_e,
		     OutputIterator chi_0_s_b, ScalarType chi_0);

  template <typename Array1, typename Array2, typename ScalarType>
  void compute_chi_m(const Array1& sigma, Array2& chi_0_s, ScalarType chi_0);

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_rho_m(InputIterator1 rho_0_s_b, InputIterator1 rho_0_s_e,
		     InputIterator2 sigma_b, OutputIterator rho_1_s_b,
		     ScalarType chi_0);

  template <typename Array1, typename Array2, typename Array3,
	    typename ScalarType>
  void compute_rho_m(const Array1& rho_0_s, const Array2& sigma,
		  Array3& rho_1_s, ScalarType chi_0);

  template <typename Array1, typename Array2>
    void vectorize_copy(const Array1& source, Array2& dest);

} // end namespace trans_m

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p bicgstab_m : Multi-mass Biconjugate Gradient stabilized method
 */
template <class LinearOperator,
          class VectorType1, class VectorType2, class VectorType3>
void bicgstab_m(LinearOperator& A,
        VectorType1& x, VectorType2& b, VectorType3& sigma);

/*! \p bicgstab_m : Multi-mass Biconjugate Gradient stabilized method
 */
template <class LinearOperator,
          class VectorType1, class VectorType2, class VectorType3,
          class Monitor>
void bicgstab_m(LinearOperator& A,
        VectorType1& x, VectorType2& b, VectorType3& sigma,
        Monitor& monitor);
/*! \}
 */

} // end namespace krylov
} // end namespace cusp

#include<cusp/krylov/detail/bicgstab_m.inl>

