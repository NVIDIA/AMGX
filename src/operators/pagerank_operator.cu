// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
