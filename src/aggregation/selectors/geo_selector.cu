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

#include <basic_types.h>
#include <algorithm>
#include <list>
#include <cmath>
#include <aggregation/selectors/geo_selector.h>
#include <error.h>
#include <types.h>
#include <math.h>
#include "util.h"

#include <cutil.h>
#include <util.h>
#include <blas.h>
#include <multiply.h>

#include <matrix_analysis.h>

#include<thrust/count.h> //count
#include<thrust/sort.h> //sort
#include<thrust/binary_search.h> //lower_bound
#include<thrust/unique.h> //unique
#include<cusp/detail/format_utils.h> //offsets_to_indices

#define epsilon 1e-10
namespace amgx
{
namespace aggregation
{

// ------------------------
//  Kernels
// ------------------------

//generate the aggregates labels
template <typename IndexType, typename ValueType>
__global__
void generatelabel1d(IndexType *aggregation, const ValueType *points, const double cord_min, const double cord_max, const int npoint, const int nlevel)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double distance = 1.01 * (cord_max - cord_min);

    for (; idx < npoint; idx += gridDim.x * blockDim.x)
    {
        aggregation[idx] = (int)((points[idx] - cord_min) / distance * nlevel);
    }
}

template <typename IndexType, typename ValueType>
__global__
void generatelabel1dpxb(IndexType *aggregation, const int alpha, const ValueType *points, const double cord_min, const double cord_max, const int npoint, const int nlevel)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double distance = 1.01 * (cord_max - cord_min);

    for (; idx < npoint; idx += gridDim.x * blockDim.x)
    {
        aggregation[idx] = aggregation[idx] + alpha * (int)((points[idx] - cord_min) / distance * nlevel);
    }
}

template <typename IndexType>
__global__
void generatelabel3d(IndexType *aggregation, const int n, const int n1d, const int n1d_orig)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n1d2 = n1d_orig * n1d_orig;
    int x, y, z;

    for (; idx < n; idx += gridDim.x * blockDim.x)
    {
        int label = idx % n1d2;
        x = (label % n1d_orig) / 2;
        y = (label / n1d_orig) / 2;
        z = idx / n1d2 / 2;
        aggregation[idx] = z * n1d * n1d + y * n1d + x;
    }
}

template <typename IndexType>
__global__
void generatelabel2d(IndexType *aggregation, const int n, const int n1d, const int n1d_orig)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x, y;

    for (; idx < n; idx += gridDim.x * blockDim.x)
    {
        x = (idx % n1d_orig) / 2;
        y = (idx / n1d_orig) / 2;
        aggregation[idx] = y * n1d + x;
    }
}


template <typename IndexType >
__global__
void aggregateLevel(IndexType *indices, const IndexType *next_level, const int num_rows)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < num_rows; idx += gridDim.x * blockDim.x)
    {
        indices[idx] = next_level[indices[idx]];
    }
}



// -----------------
//  Methods
// ----------------

// Constructor
template<class T_Config>
GEO_SelectorBase<T_Config>::GEO_SelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
}

template <class T_Config>
bool compx (typename GEO_SelectorBase<T_Config>::p3d i, typename GEO_SelectorBase<T_Config>::p3d j) { return (i.x < j.x); }

template <class T_Config>
bool compy (typename GEO_SelectorBase<T_Config>::p3d i, typename GEO_SelectorBase<T_Config>::p3d j) { return (i.y < j.y); }

template <class T_Config>
bool compz (typename GEO_SelectorBase<T_Config>::p3d i, typename GEO_SelectorBase<T_Config>::p3d j) { return (i.z < j.z); }

template <typename Tuple, typename GeoType, typename IndexType>
struct reduce_functor2 : public amgx::thrust::binary_function< Tuple, Tuple, Tuple >
{
    __host__ __device__
    Tuple operator()(const Tuple x, const Tuple y)
    {
        return amgx::thrust::make_tuple(amgx::thrust::get<0>(x) + amgx::thrust::get<0>(y),
                                  amgx::thrust::get<1>(x) + amgx::thrust::get<1>(y),
                                  amgx::thrust::get<2>(x) + amgx::thrust::get<2>(y)
                                 );
    }
};

template <typename Tuple, typename GeoType, typename IndexType>
struct reduce_functor3 : public amgx::thrust::binary_function< Tuple, Tuple, Tuple >
{
    __host__ __device__
    Tuple operator()(const Tuple x, const Tuple y)
    {
        return amgx::thrust::make_tuple(amgx::thrust::get<0>(x) + amgx::thrust::get<0>(y),
                                  amgx::thrust::get<1>(x) + amgx::thrust::get<1>(y),
                                  amgx::thrust::get<2>(x) + amgx::thrust::get<2>(y),
                                  amgx::thrust::get<3>(x) + amgx::thrust::get<3>(y)
                                 );
    }
};

// interpolates geometric info from the matrix using the matrix aggregation
template<class T_Config>
void GEO_SelectorBase<T_Config>::interpolateGeoinfo( Matrix<T_Config> &A )
{
    const int threads_per_block = 512;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (A.get_num_rows() - 1) / threads_per_block + 1 );
    ngeo_x.resize(A.get_num_rows());
    ngeo_y.resize(A.get_num_rows());

    if (this->dimension == 3)
    {
        ngeo_z.resize(A.get_num_rows());
    }

    if (!(A.hasParameter("aggregates_info")) )
    {
        FatalError("Cannot find information about previous aggregations for GEO selector", AMGX_ERR_BAD_PARAMETERS);
    }

    std::vector<IVector *> *agg_info = A.template getParameterPtr< std::vector<IVector *> >("aggregates_info");
    IVector cur_geo_idx = *(agg_info->at(0));
    IVector v_counter(this->num_origonal, 1), v_new_counter(A.get_num_rows(), 1), r_coord(A.get_num_rows());

    if ( this->num_origonal != cur_geo_idx.size() )
    {
        FatalError("GEO size doesn't match original matrix dimension", AMGX_ERR_BAD_PARAMETERS);
    }

    for (size_t i = 1; i < agg_info->size(); i++)
    {
        aggregateLevel <<< num_blocks, threads_per_block>>>(cur_geo_idx.raw(), agg_info->at(i)->raw(), this->num_origonal);
        cudaCheckError();
    }

    if (this->dimension == 2)
    {
        amgx::thrust::sort_by_key(cur_geo_idx.begin(), cur_geo_idx.end(), amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(this->cord_x->begin(), this->cord_y->begin())));
        amgx::thrust::reduce_by_key(
            cur_geo_idx.begin(),
            cur_geo_idx.end(),
            amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(this->cord_x->begin(), this->cord_y->begin(), v_counter.begin())),
            r_coord.begin(),
            amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(ngeo_x.begin(), ngeo_y.begin(), v_new_counter.begin())),
            amgx::thrust::equal_to<int>(),
            reduce_functor2< amgx::thrust::tuple< ValueType, ValueType, IndexType>, ValueType, IndexType > ()
        );
    }
    else
    {
        amgx::thrust::sort_by_key(cur_geo_idx.begin(), cur_geo_idx.end(), amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(this->cord_x->begin(), this->cord_y->begin(), this->cord_z->begin())));
        amgx::thrust::reduce_by_key(
            cur_geo_idx.begin(),
            cur_geo_idx.end(),
            amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(this->cord_x->begin(), this->cord_y->begin(), this->cord_z->begin(), v_counter.begin())),
            r_coord.begin(),
            amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(ngeo_x.begin(), ngeo_y.begin(), ngeo_z.begin(), v_new_counter.begin())),
            amgx::thrust::equal_to<int>(),
            reduce_functor3< amgx::thrust::tuple< ValueType, ValueType, ValueType, IndexType>, ValueType, IndexType > ()
        );
    }

    cudaCheckError();
}

// Choose the aggregates for csr_matrix_d format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GEO_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_1x1( Matrix_d &A,
        typename Matrix_d::IVector &aggregates, typename Matrix_d::IVector &aggregates_global, int &num_aggregates)
{
    if (this->dimension == 0)
    {
        FatalError("No input geometric information, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    //initialize the aggregates vector
    int n = A.get_num_rows();
    aggregates.resize(n);
    thrust_wrapper::fill<AMGX_device>(aggregates.begin(), aggregates.end(), -1);
    cudaCheckError();
    int nlevel;

    if (this->dimension == 2) { nlevel = floor(log(sqrt((double)n)) / log(2.0)); }

    if (this->dimension == 3) { nlevel = ceil(log(cbrt((double)n)) / log(2.0)); }

    int num_per_row = (int)std::pow(2, nlevel - 1);
    const int threads_per_block = 512;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (n - 1) / threads_per_block + 1 );
    // generate aggregation index in 1d
    IndexType *aggregation_ptr = amgx::thrust::raw_pointer_cast(&aggregates[0]);

    if (A.hasParameter("uniform_based"))
    {
        printf("GEOAggregating %d-rows matrix as uniform\n", A.get_num_rows());
        int npr_orig = 2 * num_per_row;

        if (this->dimension == 3)
        {
            generatelabel3d <<< num_blocks, threads_per_block>>>(aggregation_ptr, n, num_per_row, npr_orig);
            cudaCheckError();
        }
        else
        {
            generatelabel2d <<< num_blocks, threads_per_block>>>(aggregation_ptr, n, num_per_row, npr_orig);
            cudaCheckError();
        }
    }
    else
    {
        std::vector<VVector *> geo_ptrs(3);
        // real geometry source
        geo_ptrs[0] = this->cord_x;
        geo_ptrs[1] = this->cord_y;
        geo_ptrs[2] = this->cord_z;

        // do we need to interpolate from the finest level?
        if (A.get_num_rows() != this->num_origonal)
        {
            this->interpolateGeoinfo(A);

            if (this->dimension == 3)
            {
                geo_ptrs[2] = &this->ngeo_z;
            }

            geo_ptrs[0] = &this->ngeo_x;
            geo_ptrs[1] = &this->ngeo_y;
        }

        //Find the boundary coordinates
        this->xmax = *amgx::thrust::max_element(geo_ptrs[0]->begin(), geo_ptrs[0]->end());
        this->xmin = *amgx::thrust::min_element(geo_ptrs[0]->begin(), geo_ptrs[0]->end());
        this->ymax = *amgx::thrust::max_element(geo_ptrs[1]->begin(), geo_ptrs[1]->end());
        this->ymin = *amgx::thrust::min_element(geo_ptrs[1]->begin(), geo_ptrs[1]->end());

        if (this->dimension == 3)
        {
            this->zmax = *amgx::thrust::max_element(geo_ptrs[2]->begin(), geo_ptrs[2]->end());
            this->zmin = *amgx::thrust::min_element(geo_ptrs[2]->begin(), geo_ptrs[2]->end());
        }

        cudaCheckError();
        ValueType *point_ptr = geo_ptrs[0]->raw();
        generatelabel1d <<< num_blocks, threads_per_block>>>(aggregation_ptr, point_ptr, this->xmin, this->xmax, n, num_per_row);
        cudaCheckError();
        point_ptr = geo_ptrs[1]->raw();
        generatelabel1dpxb <<< num_blocks, threads_per_block>>>(aggregation_ptr, num_per_row, point_ptr, this->ymin, this->ymax, n, num_per_row);
        cudaCheckError();

        if (this->dimension > 2)
        {
            point_ptr = geo_ptrs[2]->raw();
            generatelabel1dpxb <<< num_blocks, threads_per_block>>>(aggregation_ptr, num_per_row * num_per_row, point_ptr, this->zmin, this->zmax, n, num_per_row);
            cudaCheckError();
        }
    }

    num_aggregates = num_per_row * num_per_row;

    if (this->dimension == 3) { num_aggregates *= num_per_row; }

    A.template setParameter < int > ("uniform_based", 1); // not exactly correct. But this is workarond for now, since parameters are instanlty copied for the coarser matrix in aggregation level
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GEO_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(const Matrix_d &A,
        typename Matrix_d::IVector &aggregates, typename Matrix_d::IVector &aggregates_global, int &num_aggregates)
{
    FatalError("GEO_Selector not implemented for non square blocks, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

// Choose aggregates on host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GEO_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_1x1( Matrix_h &A,
        IVector &aggregates,  IVector &aggregates_global, int &num_aggregates)
{
    FatalError("setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

// Choose aggregates on host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GEO_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(const Matrix_h &A,
        IVector &aggregates,  IVector &aggregates_global, int &num_aggregates)
{
    FatalError("setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <class T_Config>
void GEO_SelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    //printf("Selector: %d processing with the GEO\n", A.get_num_rows());
    this->dimension = A.template getParameter<int>("dim");
    cord_x = A.template getParameterPtr< VVector >("geo.x");
    cord_y = A.template getParameterPtr< VVector >("geo.y");
    num_origonal = A.template getParameter<int>("geo_size");

    if (this->dimension == 3)
    {
        cord_z = A.template getParameterPtr< VVector >("geo.z");
    }

    if (this->dimension == 0) { FatalError("GEO_SELECTOR: Problem dimension is not valid.", AMGX_ERR_BAD_PARAMETERS); }

    if (num_origonal == 0) { FatalError("Problem size is not valid.", AMGX_ERR_BAD_PARAMETERS); }

    if (A.get_block_size() == 1 && !A.hasProps(DIAG) )
    {
        setAggregates_1x1( A, aggregates, aggregates_global, num_aggregates );
    }
    else if (A.get_block_dimx() == A.get_block_dimy())
    {
        setAggregates_1x1( A, aggregates, aggregates_global, num_aggregates );
    }
    else
    {
        FatalError("Unsupported block size for GEO_Selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

// ---------------------------
// Explict instantiations
// ---------------------------
#define AMGX_CASE_LINE(CASE) template class GEO_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class GEO_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
