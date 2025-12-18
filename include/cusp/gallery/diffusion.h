// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file diffusion.h
 *  \brief Anisotropic diffusion matrix generators
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/gallery/stencil.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES 1  // make sure M_PI is defined
#endif
#include <math.h>
#include <type_traits>

namespace cusp
{

namespace gallery
{

/*! \addtogroup gallery Matrix Gallery
 *  \ingroup gallery
 *  \{
 */

struct disc_type {};

struct FD : public disc_type {};
struct FE : public disc_type {};

/*! \p diffusion: Create a matrix representing an anisotropic
 * Poisson problem discretized on an \p m by \p n grid with
 * the a given direction.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns 
 * \param eps magnitude of anisotropy 
 * \param theta angle of anisotropy in radians
 *
 * \tparam MatrixType matrix container
 *
 */
template <typename Method, typename MatrixType>
void diffusion(	MatrixType& matrix, size_t m, size_t n, 
		const double eps = 1e-5, 
		const double theta = M_PI/4.0)
{
	typedef typename MatrixType::index_type IndexType;
	typedef typename MatrixType::value_type ValueType;
	typedef amgx::thrust::tuple<IndexType,IndexType>    StencilIndex;
	typedef amgx::thrust::tuple<StencilIndex,ValueType> StencilPoint;

	ValueType C = cos(theta);
	ValueType S = sin(theta);
	ValueType CC = C*C;
	ValueType SS = S*S;
	ValueType CS = C*S;

	ValueType a;
	ValueType b;
	ValueType c;
	ValueType d;
	ValueType e;

	if( std::is_same<Method, FE>::value )
	{
		a = (-1.0*eps - 1.0)*CC + (-1.0*eps - 1.0)*SS + ( 3.0*eps - 3.0)*CS;
        	b = ( 2.0*eps - 4.0)*CC + (-4.0*eps + 2.0)*SS;
        	c = (-1.0*eps - 1.0)*CC + (-1.0*eps - 1.0)*SS + (-3.0*eps + 3.0)*CS;
        	d = (-4.0*eps + 2.0)*CC + ( 2.0*eps - 4.0)*SS;
        	e = ( 8.0*eps + 8.0)*CC + ( 8.0*eps + 8.0)*SS;

		a /= 6.0;
		b /= 6.0;
		c /= 6.0;
		d /= 6.0;
		e /= 6.0;
	}
	else if( std::is_same<Method, FD>::value )
	{
		a = 0.5 * (eps-1.0) * CS;
		b = -(eps*SS + CC); 
		c = -a;
		d = -(eps*CC + SS);
		e = 2.0 * (eps+1.0);
	}
  	else
  	{
   		throw cusp::invalid_input_exception("unrecognized discretization method");
  	}

	cusp::array1d<StencilPoint, cusp::host_memory> stencil;

	stencil.push_back(StencilPoint(StencilIndex( -1, -1), a));
	stencil.push_back(StencilPoint(StencilIndex(  0, -1), b));
	stencil.push_back(StencilPoint(StencilIndex(  1, -1), c));
	stencil.push_back(StencilPoint(StencilIndex( -1,  0), d));
	stencil.push_back(StencilPoint(StencilIndex(  0,  0), e));
	stencil.push_back(StencilPoint(StencilIndex(  1,  0), d));
	stencil.push_back(StencilPoint(StencilIndex( -1,  1), c));
	stencil.push_back(StencilPoint(StencilIndex(  0,  1), b));
	stencil.push_back(StencilPoint(StencilIndex(  1,  1), a));

	cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n));
}
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

