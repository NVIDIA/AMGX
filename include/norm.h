/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <getvalue.h>
#include <error.h>
#include <types.h>
#include <basic_types.h>
#include <vector.h>

#include <amgx_types/util.h>

namespace amgx
{

/**********************************************************
 * Returns the norm of a vector
 *********************************************************/
template<class VectorType, class MatrixType>
typename types::PODTypes<typename VectorType::value_type>::type get_norm(const MatrixType &A, const VectorType &r, const NormType norm_type, typename types::PODTypes<typename VectorType::value_type>::type norm_factor = 1.0);

template <class VectorType, class MatrixType, class PlainVectorType>
void get_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm, typename types::PODTypes<typename VectorType::value_type>::type norm_factor = 1.0);

template <class VectorType, class MatrixType>
void compute_norm_factor(MatrixType &A, VectorType &b, VectorType &x, const NormType normType, typename types::PODTypes<typename VectorType::value_type>::type &normFactor);

} // namespace amgx

