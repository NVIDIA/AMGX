/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

namespace amgx
{

template <class T_Config> class Operator;

}

#include <operators/pagerank_operator.h>
#include <blas.h>
#include <amgx_cusparse.h>
#include <cublas_v2.h>

//#define PR_VERBOSE 1
namespace amgx
{

template <typename TConfig>
void PagerankOperator<TConfig>::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Operator<TConfig> &A = *m_A;
    Vector<TConfig> &aa = *m_a;
    Vector<TConfig> &bb = *m_b;
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    //res = H^T*pi (SpMV)
    A.apply(v, res, OWNED);
    //res =  alpha*H^T*pi
    scal(res, m_alpha, offset, size);
    //gamma  = a.pi
    ValueTypeVec gamma = dot(A, aa, v);
    //res = res+gamma*b
    axpy(bb, res, gamma, offset, size);
    // Print for debug
#ifdef PR_VERBOSE
    std::cout << "Vector a = ";

    for (int i = 0; i < aa.size(); ++i)
    {
        std::cout << aa[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "Vector b = ";

    for (int i = 0; i < bb.size(); ++i)
    {
        std::cout << bb[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "Vector v = ";

    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "Vector res = ";

    for (int i = 0; i < res.size(); ++i)
    {
        std::cout << res[i] << " ";
    }

    std::cout << std::endl;
    std::cout << std::endl;
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
#endif
}

#define AMGX_CASE_LINE(CASE) template class PagerankOperator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
