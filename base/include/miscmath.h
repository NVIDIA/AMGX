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

namespace amgx
{
template <class Matrix>  typename Matrix::value_type estimate_largest_eigen_value(Matrix &A);
}

#include <norm.h>
#include <multiply.h>
#include <blas.h>

namespace amgx
{

template <class Matrix>
typename Matrix::value_type estimate_largest_eigen_value(Matrix &A)
{
    typedef typename Matrix::TConfig TConfig;
    typedef typename Matrix::value_type ValueTypeA;
    typedef typename TConfig::VecPrec ValueTypeB;
    typedef Vector<TConfig> VVector;
    VVector x(A.get_num_rows()), y(A.get_num_rows());
    fill(x, 1);

    for (int i = 0; i < 20; i++)
    {
        ValueTypeB Lmax = get_norm(A, x, LMAX);
        scal(x, ValueTypeB(1) / Lmax);
        multiply(A, x, y);
        x.swap(y);
    }

    ValueTypeB retval = get_norm(A, x, L2) / get_norm(A, y, L2);
    return retval;
}

} // namespace amgx
