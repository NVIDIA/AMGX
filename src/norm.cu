// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

#include "amgx_timer.h"
#include "amgx_types/util.h"
#include "thrust_wrapper.h"

namespace amgx
{

/**********************************************************
 * Returns the norm of a vector
 *********************************************************/

template<class VectorType, class MatrixType>
typename types::PODTypes<typename VectorType::value_type>::type get_norm(const MatrixType &A, const VectorType &r, const NormType norm_type, typename types::PODTypes<typename VectorType::value_type>::type norm_factor)
{
    typedef typename types::PODTypes<typename VectorType::value_type>::type value_type;
    value_type nrm;
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);

    if (norm_type == L1 || norm_type == L1_SCALED)
    {
        nrm = nrm1(r, offset, size);

        if (A.is_matrix_distributed())
        {
            A.getManager()->global_reduce_sum(&nrm);
        }

        return (norm_type == L1_SCALED) ? nrm / norm_factor : nrm;
    }
    else if (norm_type == L2)
    {
        nrm = nrm2(r, offset, size);

        if (A.is_matrix_distributed())
        {
            nrm = nrm * nrm;
            A.getManager()->global_reduce_sum(&nrm);
            nrm = sqrt(nrm);
        }
        
        return nrm;
    }
    else if (norm_type == LMAX)
    {
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

            for (int j = 0; j < values.size(); j++)
            {
                nrm = (nrm > values[j][0] ? nrm : values[j][0]);
            }
        }

        return nrm;
    }

    return -1;
}

template <class VectorType, class MatrixType, class PlainVectorType>
class Norm_1x1;

template <class VectorType, class MatrixType, class PlainVectorType>
void get_1x1_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm, typename types::PODTypes<typename VectorType::value_type>::type norm_factor)
{
    Norm_1x1<VectorType, MatrixType, PlainVectorType>::get_1x1_norm(A, r, block_size, norm_type, block_nrm, norm_factor);
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

        static void get_1x1_norm(const MatrixType &A, const Vector_h &r, const int block_size, const NormType norm_type, PODHostVec &block_nrm, typename types::PODTypes<ValueTypeB>::type norm_factor)
        {
            //collect values from all neighbors, and do the "reduction" part
            std::vector<PODHostVec> values(0);
            block_nrm.resize(1);
            int offset, size;
            double sum = 0.l;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);

            if (norm_type == L1 || norm_type == L1_SCALED)
            {
                block_nrm[0] = nrm1(r, offset, size);

                if (A.is_matrix_distributed())
                {
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 4);
                    block_nrm[0] = 0;

                    for (int j = 0; j < values.size(); j++)
                    {
                        sum += values[j][0];
                    }

                    block_nrm[0] = sum;
                }

                if (norm_type == L1_SCALED)
                {
                    block_nrm[0] /= norm_factor;
                }
            }
            else if (norm_type == L2)
            {
                block_nrm[0] = nrm2(r, offset, size);

                if (A.is_matrix_distributed())
                {
                    block_nrm[0] *= block_nrm[0];
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 5);
                    block_nrm[0] = 0;

                    for (int j = 0; j < values.size(); j++)
                    {
                        sum += values[j][0];
                    }

                    block_nrm[0] = sqrt(sum);
                }
            }
            else if (norm_type == LMAX)
            {
                block_nrm[0] = nrmmax(r, offset, size);

                if (A.is_matrix_distributed())
                {
                    A.getManager()->getComms()->global_reduce(values, block_nrm, A, 6);

                    for (int j = 0; j < values.size(); j++)
                    {
                        block_nrm[0] = (block_nrm[0] > values[j][0] ? block_nrm[0] : values[j][0]);
                    }
                }
            }
            else
            {
                FatalError("Normtype is not supported in get_1x1_norm", AMGX_ERR_NOT_IMPLEMENTED);
            }
        };
};

template <class Vector, class MatrixType, class PlainVectorType>
class Norm_Square;

template <class VectorType, class MatrixType, class PlainVectorType>
void get_sq_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm, typename types::PODTypes<typename VectorType::value_type>::type norm_factor)
{
    Norm_Square<VectorType, MatrixType, PlainVectorType>::get_sq_norm(A, r, block_size, norm_type, block_nrm, norm_factor);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec, class MatrixType, AMGX_VecPrecision t_pod_vecPrec >
class Norm_Square<Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >, MatrixType, Vector<TemplateConfig<AMGX_host, t_pod_vecPrec, t_matPrec, t_indPrec>>>
{
    public:
        typedef Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Vector_h; // host vector of current TConfig(r)
        typedef typename Vector_h::value_type ValueTypeB; // vector's valuetype
        typedef TemplateConfig<AMGX_host, types::PODTypes<ValueTypeB>::vec_prec, MatrixType::TConfig::matPrec, MatrixType::TConfig::indPrec> hvector_type; // TConfig host with pod-values for ValueTypeB
        typedef Vector<hvector_type> HVector; //vectors for saving norms from allgather

        static void get_sq_norm(const MatrixType &A, const Vector_h &r, const int block_size, const NormType norm_type, HVector &block_nrm, typename types::PODTypes<ValueTypeB>::type  norm_factor)
        {
            int bsize = block_nrm.size();
            int offset, size;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);
            std::vector <double> norm(block_size, 0.l);

            if (norm_type == L1 || norm_type == L1_SCALED)
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
                    block_nrm[j] = (norm_type == L1_SCALED) ? norm[j] / norm_factor : norm[j];
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
        static void get_sq_norm(const MatrixType &A, const Vector_d &r, const int block_size, const NormType norm_type, HVector &block_nrm, typename types::PODTypes<ValueTypeB>::type norm_factor)
        {
            int bsize = block_nrm.size();
            int offset, size;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);
            const int ncells = (size * r.get_block_size()) / bsize;

            if ( (size * r.get_block_size()) % bsize != 0)
            {
                FatalError("Size of vector r must be multiple of block size", AMGX_ERR_BAD_PARAMETERS);
            }

            if (norm_type == L1 || norm_type == L1_SCALED)
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

                if (norm_type == L1_SCALED)
                {
                    for (int i = 0; i < block_nrm.size(); ++i)
                    {
                        block_nrm[i] /= norm_factor;
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
void get_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm, typename types::PODTypes<typename VectorType::value_type>::type norm_factor)
{
    if (block_size == 1)
    {
        get_1x1_norm(A, r, block_size, norm_type, block_nrm, norm_factor);
    }
    else
    {
        get_sq_norm(A, r, block_size, norm_type, block_nrm, norm_factor);
    }
}

template <class VectorType, class MatrixType>
class Norm_Factor;

template <class VectorType, class MatrixType>
void compute_norm_factor(MatrixType &A, VectorType &b, VectorType &x, const NormType normType, typename types::PODTypes<typename VectorType::value_type>::type &normFactor)
{
    if(normType == L1_SCALED)
    {
        Norm_Factor<VectorType, MatrixType>::compute_norm_factor(A, b, x, normFactor);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec, class MatrixType >
class Norm_Factor<Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >, MatrixType>
{
    public:
        typedef Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Vector_h;
        typedef typename Vector_h::value_type ValueTypeVec;
        typedef typename types::PODTypes<ValueTypeVec>::type ValueTypeNorm;

        static void compute_norm_factor(MatrixType &A, Vector_h &b, Vector_h &x, ValueTypeNorm& normFactor)
        {
            FatalError("L1 scaled norm not supported with host execution.", AMGX_ERR_NOT_IMPLEMENTED);
        }
};

template<int warpSize, class ValueTypeMat, class IndexTypeVec, class ValueTypeVec, class ValueTypeNorm>
__global__ void scaled_norm_factor_calc(
    int nRows, ValueTypeMat *Avals, IndexTypeVec *Arows, ValueTypeVec *Ax, ValueTypeVec *b, 
    ValueTypeVec xAvg, ValueTypeNorm *localNormFactor, ValueTypeMat *diaVals = nullptr)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ ValueTypeNorm normFactor_s;
    if(threadIdx.x == 0)
    {
        normFactor_s = amgx::types::util<ValueTypeNorm>::get_zero();
    }

    ValueTypeNorm normFactor = 0.0;

    if (r < nRows)
    {
        ValueTypeMat Arow_sum = amgx::types::util<ValueTypeMat>::get_zero();

        // Read in the row
#pragma unroll
        for (int i = Arows[r]; i < Arows[r + 1]; ++i)
        {
            Arow_sum = Arow_sum + Avals[i];
        }
        if(diaVals) Arow_sum = Arow_sum + diaVals[r];

        normFactor =
            types::util<ValueTypeVec>::abs(Ax[r] - Arow_sum * xAvg) +
            types::util<ValueTypeVec>::abs(b[r] - Arow_sum * xAvg);
    }

    // Ensure normFactor_s is initialised
    __syncthreads();

    // Warp-local reduction to lane 0
    for(int i = warpSize/2; i > 0; i /= 2)
    {
        normFactor += utils::shfl_down(normFactor, i);
    }

    // Fast shared atomic add by lane 0 of each warp
    int laneId = threadIdx.x % warpSize;
    if(laneId == 0)
    {
        utils::atomic_add(&normFactor_s, normFactor);
    }

    // Ensure normFactor_s is final
    __syncthreads();

    // Final output of normFactor by first thread of each block
    if(threadIdx.x == 0)
    {
        utils::atomic_add(localNormFactor, normFactor_s);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec, class MatrixType>
class Norm_Factor<Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >, MatrixType>
{
    public:
        typedef Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Vector_d;
        typedef typename Vector_d::value_type ValueTypeVec;
        typedef typename Vector_d::index_type IndexTypeVec;
        typedef typename types::PODTypes<ValueTypeVec>::type ValueTypeNorm;
        typedef TemplateConfig<AMGX_device, types::PODTypes<ValueTypeVec>::vec_prec, MatrixType::TConfig::matPrec, MatrixType::TConfig::indPrec> NormVectorType;
        typedef Vector<NormVectorType> NVector_d;

        static void compute_norm_factor(MatrixType &A, Vector_d &b, Vector_d &x, ValueTypeNorm &normFactor)
        {
            if (A.get_block_dimx() != 1 || A.get_block_dimy() != 1)
            {
                FatalError("L1 scaled norm only supported with scalar matrices", AMGX_ERR_NOT_IMPLEMENTED);
            }

            // Calculate Ax
            int offset, nRows;
            A.getOffsetAndSizeForView(OWNED, &offset, &nRows);

            Vector_d Ax(nRows);
            A.apply(x, Ax);

            // Calculate global average x
            ValueTypeVec xAvg = amgx::thrust::reduce(x.begin(), x.begin() + nRows, amgx::types::util<ValueTypeVec>::get_zero());

            int64_t nr = nRows;
            if (A.is_matrix_distributed())
            {
                A.manager->global_reduce_sum(&xAvg);
                nr = A.manager->num_rows_global;
            }
            amgx::types::util<ValueTypeVec>::divide_by_integer(xAvg, nr);

            // Make a copy of b
            Vector_d bTmp(b);

            // Calculate row sums then the local norm factors
            constexpr int nThreads = 128;
            constexpr int warpSize = 32;
            const int nBlocks = nRows/nThreads + 1;
            NVector_d localNormFactor(1, amgx::types::util<ValueTypeNorm>::get_zero());
            if(A.hasProps(DIAG))
            {
                scaled_norm_factor_calc<warpSize><<<nBlocks, nThreads>>>(
                    nRows,
                    A.values.raw(),
                    A.row_offsets.raw(),
                    Ax.raw(),
                    bTmp.raw(),
                    xAvg,
                    localNormFactor.raw(),
                    A.values.raw() + A.diagOffset()*A.get_block_size());
                cudaCheckError();
            }
            else
            {
                scaled_norm_factor_calc<warpSize><<<nBlocks, nThreads>>>(
                    nRows,
                    A.values.raw(),
                    A.row_offsets.raw(),
                    Ax.raw(),
                    bTmp.raw(),
                    xAvg,
                    localNormFactor.raw());
                cudaCheckError();
            }

            // Fetch the normFactor result and reduce across all ranks
            normFactor = localNormFactor[0];

            if (A.is_matrix_distributed())
            {
                A.manager->global_reduce_sum(&normFactor);
            }

            // Print the norm factor
            std::stringstream info;
            info.precision(12);
            info << "\tAmgX Scaled Norm Factor: " << std::scientific << normFactor << "\n";
            amgx_output(info.str().c_str(), info.str().length());
        }
};

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes< typename Vector<TemplateMode<CASE>::Type>::value_type>::type get_norm(const Matrix<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const NormType norm_type, typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type norm_factor);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes< typename Vector<TemplateMode<CASE>::Type>::value_type>::type get_norm(const Operator<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const NormType norm_type, typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type norm_factor);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  typedef typename Vector< TemplateMode<CASE>::Type >::value_type ValueTypeMB##CASE ;\
  typedef TemplateMode<CASE>::Type::template setMemSpace<AMGX_host>::Type::template setVecPrec< types::PODTypes< ValueTypeMB##CASE >::vec_prec >::Type CurTConfigMB_h##CASE ;\
  template void get_norm(const Matrix<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const int block_size, const NormType norm_type, Vector< CurTConfigMB_h##CASE >& block_nrm, typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type norm_factor); \
  template void compute_norm_factor(Matrix<TemplateMode<CASE>::Type> &A, Vector<TemplateMode<CASE>::Type> &b, Vector<TemplateMode<CASE>::Type> &x, const NormType normType, typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type &normFactor);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  typedef typename Vector< TemplateMode<CASE>::Type >::value_type ValueTypeOB##CASE ;\
  typedef TemplateMode<CASE>::Type::template setMemSpace<AMGX_host>::Type::template setVecPrec< types::PODTypes< ValueTypeOB##CASE >::vec_prec >::Type CurTConfigOB_h##CASE ;\
  template void get_norm(const Operator<TemplateMode<CASE>::Type>& A, const Vector<TemplateMode<CASE>::Type>& r, const int block_size, const NormType norm_type, Vector< CurTConfigOB_h##CASE >& block_nrm, typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type norm_factor);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
