// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sort.h>
#include <basic_types.h>
#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#include <thrust/sort.h>
#ifdef _WIN32
#pragma warning (pop)
#endif
#include <vector.h>
#include <error.h>

namespace amgx
{

template <class Vector>
void sort(Vector &v)
{
    amgx::thrust::sort(v.begin(), v.end());
    cudaCheckError();
}

typedef Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matFloat, AMGX_indInt> > Vector_uint64_d;

/****************************************
 * Explict instantiations
 ***************************************/
template void sort<Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matFloat, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matFloat, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matFloat, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matFloat, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matDouble, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matDouble, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matDouble, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matDouble, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matComplex, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matComplex, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matComplex, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matComplex, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matDoubleComplex, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_device, AMGX_vecUInt64, AMGX_matDoubleComplex, AMGX_indInt> > &v);

template void sort<Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matDoubleComplex, AMGX_indInt> > >(Vector<TemplateConfig<AMGX_host, AMGX_vecUInt64, AMGX_matDoubleComplex, AMGX_indInt> > &v);

} // namespace amgx
