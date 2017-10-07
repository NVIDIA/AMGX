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

#include <norm.h>
#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#ifdef _WIN32
#pragma warning (pop)
#endif
#include <blas.h>
#include <cusp/blas.h>
#include <basic_types.h>
#include <util.h>
#include <types.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include "strided_reduction.h"

#include "amgx_types/util.h"

namespace amgx
{

/**********************************************************
 * Returns the norm of a vector
 *********************************************************/

template<class VectorType, class MatrixType>
typename types::PODTypes<typename VectorType::value_type>::type get_norm(const MatrixType &A, const VectorType &r, const NormType norm_type)
{
    typedef typename types::PODTypes<typename VectorType::value_type>::type value_type;
    value_type nrm;
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);

    switch (norm_type)
    {
        case L1:
            nrm = nrm1(r, offset, size);

            if (A.is_matrix_distributed())
            {
                A.getManager()->global_reduce_sum(&nrm);
            }

            return nrm;

        case L2:
            nrm = nrm2(r, offset, size);

            if (A.is_matrix_distributed())
            {
                nrm = nrm * nrm;
                A.getManager()->global_reduce_sum(&nrm);
                nrm = sqrt(nrm);
            }

            return nrm;

        case LMAX:
            nrm = nrmmax(r, offset, size);

            if (A.is_matrix_distributed())
            {
                typedef TemplateConfig<AMGX_host, types::PODTypes<typename VectorType::value_type>::vec_prec, MatrixType::TConfig::matPrec, MatrixType::TConfig::indPrec> hvector_type;
                typedef Vector<hvector_type> HVector;
                //collect values from all neighbors, and do the "reduction" part
                std::vector<HVector> values(0);
                HVector my_nrm(1);
                my_nrm[0] = nrm;
                A.getManager()->getComms()->global_reduce(values, my_nrm, A, 3);

                for (int j = 0; j < values.size(); j++) { nrm = (nrm > values[j][0] ? nrm : values[j][0]); }
            }

            return nrm;
    }

    return -1;
}

template <class VectorType, class MatrixType, class PlainVectorType>
class Norm_1x1;

template <class VectorType, class MatrixType, class PlainVectorType>
void get_1x1_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm)
{
    Norm_1x1<VectorType, MatrixType, PlainVectorType>::get_1x1_norm(A, r, block_size, norm_type, block_nrm);
}

template <AMGX_MemorySpace t_memSpace, AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec, class MatrixType, AMGX_VecPrecision t_pod_vecPrec >
class Norm_1x1< Vector<TemplateConfig<t_memSpace, t_vecPrec, t_matPrec, t_indPrec> >, MatrixType, Vector<TemplateConfig<AMGX_host, t_pod_vecPrec, t_matPrec, t_indPrec>>>
{
    public:
        typedef Vector<TemplateConfig<t_memSpace, t_vecPrec, t_matPrec, t_indPrec> > Vector_h;
        typedef Vector<TemplateConfig<AMGX_host, t_pod_vecPrec, t_matPrec, t_indPrec>> PODHostVec; // return type
        typedef typename Vector_h::value_type ValueTypeB; // vector's valuetype
        typedef TemplateConfig<AMGX_host, types::PODTypes<ValueTypeB>::vec_prec, MatrixType::TConfig::matPrec, MatrixType::TConfig::indPrec> hvector_type; // TConfig host with pod-values for ValueTypeB
        typedef Vector<hvector_type> HVector; //vectors for saving norms from allgather

        static void get_1x1_norm(const MatrixType &A, const Vector_h &r, const int block_size, const NormType norm_type, PODHostVec &block_nrm)
        {
            //collect values from all neighbors, and do the "reduction" part
            std::vector<PODHostVec> values(0);
            block_nrm.resize(1);
            int offset, size;
            double sum = 0.l;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);

            switch (norm_type)
            {
                case L1:
                    block_nrm[0] = nrm1(r, offset, size);

                    if (A.is_matrix_distributed())
                    {
                        A.getManager()->getComms()->global_reduce(values, block_nrm, A, 4);
                        block_nrm[0] = 0;

                        for (int j = 0; j < values.size(); j++) { sum += values[j][0]; }

                        block_nrm[0] = sum;
                    }

                    break;

                case L2:
                    block_nrm[0] = nrm2(r, offset, size);

                    if (A.is_matrix_distributed())
                    {
                        block_nrm[0] *= block_nrm[0];
                        A.getManager()->getComms()->global_reduce(values, block_nrm, A, 5);
                        block_nrm[0] = 0;

                        for (int j = 0; j < values.size(); j++) { sum += values[j][0]; }

                        block_nrm[0] = sqrt(sum);
                    }

                    break;

                case LMAX:
                    block_nrm[0] = nrmmax(r, offset, size);

                    if (A.is_matrix_distributed())
                    {
                        A.getManager()->getComms()->global_reduce(values, block_nrm, A, 6);

                        for (int j = 0; j < values.size(); j++) { block_nrm[0] = (block_nrm[0] > values[j][0] ? block_nrm[0] : values[j][0]); }
                    }

                    break;

                default:
                    FatalError("Normtype is not supported in get_1x1_norm", AMGX_ERR_NOT_IMPLEMENTED);
            }
        };
};

template <class Vector, class MatrixType, class PlainVectorType>
class Norm_Square;

template <class VectorType, class MatrixType, class PlainVectorType>
void get_sq_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm)
{
    Norm_Square<VectorType, MatrixType, PlainVectorType>::get_sq_norm(A, r, block_size, norm_type, block_nrm);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec, class MatrixType, AMGX_VecPrecision t_pod_vecPrec >
class Norm_Square<Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >, MatrixType, Vector<TemplateConfig<AMGX_host, t_pod_vecPrec, t_matPrec, t_indPrec>>>
{
    public:
        typedef Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Vector_h; // host vector of current TConfig(r)
        typedef typename Vector_h::value_type ValueTypeB; // vector's valuetype
        typedef TemplateConfig<AMGX_host, types::PODTypes<ValueTypeB>::vec_prec, MatrixType::TConfig::matPrec, MatrixType::TConfig::indPrec> hvector_type; // TConfig host with pod-values for ValueTypeB
        typedef Vector<hvector_type> HVector; //vectors for saving norms from allgather

        static void get_sq_norm(const MatrixType &A, const Vector_h &r, const int block_size, const NormType norm_type, HVector &block_nrm)
        {
            int bsize = block_nrm.size();
            int offset, size;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);
            std::vector <double> norm(block_size, 0.l);

            if (norm_type == L1)
            {
                if ( (size * r.get_block_size()) % bsize != 0)
                {
                    FatalError("r.size should be multiple of block size", AMGX_ERR_BAD_PARAMETERS);
                }

                block_nrm.resize(bsize, 0.);
                //    for (int j=0;j<bsize;j++)
                //      block_nrm[j] = 0.;
                int num_cells = (size * r.get_block_size()) / bsize;

                for (int i = 0; i < num_cells; i++)
                {
                    for (int j = 0; j < bsize; j++)
                    {
                        norm[j] += types::util<ValueTypeB>::abs(r[(offset + i) * bsize + j]);
                    }
                }

                for (int j = 0; j < bsize; j++)
                {
                    block_nrm[j] = norm[j];
                }

                if (A.is_matrix_distributed())
                {
                    //collect values from all neighbors, and do the "reduction" part
                    std::vector<HVector> values(0);
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 7);

                    for (int i = 0; i < bsize; i++)
                    {
                        block_nrm[i] = 0;

                        for (int j = 0; j < values.size(); j++) { norm[i] += values[j][i]; }
                    }
                }

                for (int j = 0; j < bsize; j++)
                {
                    block_nrm[j] = norm[j];
                }
            }
            else if (norm_type == L2)
            {
                if ( (size * r.get_block_size()) % bsize != 0)
                {
                    FatalError("r.size should be multiple of block size", AMGX_ERR_BAD_PARAMETERS);
                }

                block_nrm.resize(bsize, 0.);
                int num_cells = (size * r.get_block_size()) / bsize;

                for (int i = 0; i < num_cells; i++)
                {
                    for (int j = 0; j < bsize; j++)
                    {
                        norm[j] = norm[j] + types::util<ValueTypeB>::abs(r[(offset + i) * bsize + j] * types::util<ValueTypeB>::conjugate(r[(offset + i) * bsize + j]));
                    }
                }

                for (int j = 0; j < bsize; j++)
                {
                    block_nrm[j] = norm[j];
                }

                if (A.is_matrix_distributed())
                {
                    //collect values from all neighbors, and do the "reduction" part
                    std::vector<HVector> values(0);
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 8);

                    for (int i = 0; i < bsize; i++)
                    {
                        norm[i] = 0;

                        for (int j = 0; j < values.size(); j++) { norm[i] += values[j][i]; }
                    }
                }

                for (int j = 0; j < bsize; j++)
                {
                    block_nrm[j] = sqrt(norm[j]);
                }
            }
            else
            {
                FatalError("Normtype not supported in get_norm", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }
};

__global__
void createReductionMapKernel(int *map, const int bsize, const int map_size)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < map_size; tid += blockDim.x * gridDim.x)
    {
        map[tid] = tid % bsize;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec, class MatrixType, AMGX_VecPrecision t_pod_vecPrec>
class Norm_Square<Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >, MatrixType, Vector<TemplateConfig<AMGX_host, t_pod_vecPrec, t_matPrec, t_indPrec>> >
{
    public:
        typedef Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Vector_d;
        typedef Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Vector_h; // host vector of current TConfig(r)
        typedef typename Vector_h::value_type ValueTypeB; // vector's valuetype
        typedef TemplateConfig<AMGX_host, types::PODTypes<ValueTypeB>::vec_prec, MatrixType::TConfig::matPrec, MatrixType::TConfig::indPrec> hvector_type; // TConfig host with pod-values for ValueTypeB
        typedef Vector<hvector_type> HVector; //vectors for saving norms from allgather
        static void get_sq_norm(const MatrixType &A, const Vector_d &r, const int block_size, const NormType norm_type, HVector &block_nrm)
        {
            int bsize = block_nrm.size();
            int offset, size;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);
            const int ncells = (size * r.get_block_size()) / bsize;

            if ( (size * r.get_block_size()) % bsize != 0)
            {
                FatalError("Size of vector r must be multiple of block size", AMGX_ERR_BAD_PARAMETERS);
            }

            if (norm_type == L1)
            {
                amgx::strided_reduction::reduction_generic_dispatch(
                    bsize,
                    &block_nrm[0], r.raw() + offset * r.get_block_size(),
                    size * r.get_block_size(), amgx::strided_reduction::fabs_transform());

                if (A.is_matrix_distributed())
                {
                    //collect values from all neighbors, and do the "reduction" part
                    std::vector<HVector> values(0);
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 9);

                    for (int i = 0; i < bsize; i++)
                    {
                        block_nrm[i] = 0;

                        for (int j = 0; j < values.size(); j++)
                        {
                            block_nrm[i] += values[j][i];
                        }
                    }
                }
            }
            else if (norm_type == L2)
            {
                amgx::strided_reduction::reduction_generic_dispatch(
                    bsize, &block_nrm[0],
                    r.raw() + offset * r.get_block_size(), size * r.get_block_size(),
                    amgx::strided_reduction::square_transform());

                for (int i = 0; i < bsize; i++)
                {
                    block_nrm[i] = sqrt(block_nrm[i]);
                }

                if (A.is_matrix_distributed())
                {
                    for (int i = 0; i < bsize; i++)
                    {
                        block_nrm[i] *= block_nrm[i];
                    }

                    //collect values from all neighbors, and do the "reduction" part
                    std::vector<HVector> values(0);
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 10);

                    for (int i = 0; i < bsize; i++)
                    {
                        block_nrm[i] = 0;

                        for (int j = 0; j < values.size(); j++)
                        {
                            block_nrm[i] += values[j][i];
                        }
                    }

                    for (int i = 0; i < bsize; i++)
                    {
                        block_nrm[i] = sqrt(block_nrm[i]);
                    }
                }
            }
            else
            {
                FatalError("Normtype not supported in get_norm", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }
};

template<class VectorType, class MatrixType, class PlainVectorType>
void get_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm)
{
    if (block_size == 1)
    {
        get_1x1_norm(A, r, block_size, norm_type, block_nrm);
    }
    else
    {
        get_sq_norm(A, r, block_size, norm_type, block_nrm);
    }
}

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes< typename Vector<TemplateMode<CASE>::Type>::value_type>::type  get_norm(const Matrix<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const NormType norm_type);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes< typename Vector<TemplateMode<CASE>::Type>::value_type>::type get_norm(const Operator<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const NormType norm_type);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  typedef typename Vector< TemplateMode<CASE>::Type >::value_type ValueTypeMB##CASE ;\
  typedef TemplateMode<CASE>::Type::template setMemSpace<AMGX_host>::Type::template setVecPrec< types::PODTypes< ValueTypeMB##CASE >::vec_prec >::Type CurTConfigMB_h##CASE ;\
  template void get_norm(const Matrix<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const int block_size, const NormType norm_type, Vector< CurTConfigMB_h##CASE >& block_nrm);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  typedef typename Vector< TemplateMode<CASE>::Type >::value_type ValueTypeOB##CASE ;\
  typedef TemplateMode<CASE>::Type::template setMemSpace<AMGX_host>::Type::template setVecPrec< types::PODTypes< ValueTypeOB##CASE >::vec_prec >::Type CurTConfigOB_h##CASE ;\
  template void get_norm(const Operator<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const int block_size, const NormType norm_type, Vector< CurTConfigOB_h##CASE >& block_nrm);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
