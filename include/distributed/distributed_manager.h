/* Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
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
template <class TConfig> class DistributedManager;
template <class TConfig> class DistributedArranger;
template <class TConfig> class DistributedComms;
template <class TConfig> class CommsMPIDirect;
template <class TConfig> class CommsMPIHostBuffer;
template <class TConfig> class CommsMPIHostBufferStream;
template <class TConfig> class CommsMPIHostBufferAsync;
template <class TConfig> class CommsMultiDeviceBase;
template <class TConfig> class CommsSingleDeviceBase;
namespace aggregation
{
template <class TConfig> class Aggregation_AMG_Level_Base;
}
namespace classical
{
template <class TConfig> class Classical_AMG_Level_Base;
}
namespace energymin
{
template <class TConfig> class Energymin_AMG_Level_Base;
}
}

#include <cstdio>

#include <amgx_cusparse.h>
#include <thrust/sequence.h>
#include <vector.h>
#include <error.h>
#include <distributed/distributed_comms.h>
#include <distributed/distributed_arranger.h>
#include <matrix_distribution.h>

#include "amgx_types/math.h"
#include "amgx_types/util.h"

namespace manager_utils
{
#include <sm_utils.inl>
}

namespace amgx
{

typedef IndPrecisionMap<AMGX_indInt>::Type INDEX_TYPE;

template <class TConfig> class Matrix;
template <class TConfig> class Operator;

template <class T>
__global__ void gatherToBuffer(T *source, INDEX_TYPE *map, T *dest, INDEX_TYPE bsize, INDEX_TYPE total)
{
    int nz_per_block = (blockDim.x / bsize);
    int nz = blockIdx.x * nz_per_block + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= nz_per_block * bsize) { return; }

    while (nz < total)
    {
        dest[nz * bsize + vecIdx] = source[bsize * map[nz] + vecIdx];
        nz += nz_per_block * gridDim.x;
    }
}

template <class T>
__global__ void gatherToBuffer_v3(const T *source, const INDEX_TYPE *map_offsets, INDEX_TYPE **map_ptrs, T **dest_ptrs, INDEX_TYPE bsize, INDEX_TYPE size, INDEX_TYPE num_neighbors)
{
    int nz_per_block = (blockDim.x / bsize);
    int nz = blockIdx.x * nz_per_block + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= nz_per_block * bsize) { return; }

    while (nz < size)
    {
        // Figure out which neighbor I'm responsible for
        int neighbor = 0;

        while (neighbor < num_neighbors && (nz < __ldg(&map_offsets[neighbor]) || nz >= __ldg(&map_offsets[neighbor + 1])))
        {
            neighbor++;
        }

        if (neighbor < num_neighbors && (nz >= __ldg(&map_offsets[neighbor]) && nz < __ldg(&map_offsets[neighbor + 1])))
        {
            INDEX_TYPE offset = map_offsets[neighbor];
            T *dest = dest_ptrs[neighbor];
            INDEX_TYPE *map = map_ptrs[neighbor];
            dest[ (nz - offset)*bsize + vecIdx] = source[bsize * map[nz - offset] + vecIdx];
        }

        nz += nz_per_block * gridDim.x;
    }
}


template <class T>
__global__ void gatherToBufferMultivector(T *source, INDEX_TYPE *map, T *dest,
        INDEX_TYPE num_cols,
        INDEX_TYPE lda,
        INDEX_TYPE B2L_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    while (idx < B2L_size)
    {
        //TODO: optimize removing this loop, like in gatherToBuffer.
        for (int s = 0; s < num_cols; ++s)
        {
            dest[s * B2L_size + idx] = source[s * lda + map[idx]];
        }

        idx += num_threads;
    }
}

template <class T>
__global__ void scatterFromBuffer(T *dest, INDEX_TYPE *map, T *source, INDEX_TYPE bsize, INDEX_TYPE total)
{
    int nz_per_block = (blockDim.x / bsize);
    int nz = blockIdx.x * nz_per_block + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= nz_per_block * bsize) { return; }

    while (nz < total)
    {
        dest[bsize * map[nz] + vecIdx] = dest[bsize * map[nz] + vecIdx] + source[nz * bsize + vecIdx];
        nz += nz_per_block * gridDim.x;
    }
}

template <class T>
__global__ void scatterFromBuffer_v3(T *source, const INDEX_TYPE *map_offsets, INDEX_TYPE **map_ptrs, T **dest_ptrs, INDEX_TYPE bsize, INDEX_TYPE size, INDEX_TYPE num_neighbors)
{
    int nz_per_block = (blockDim.x / bsize);
    int nz = blockIdx.x * nz_per_block + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= nz_per_block * bsize) { return; }

    while ( nz < size )
    {
        int neighbor = 0;

        while (neighbor < num_neighbors && (nz < __ldg(&map_offsets[neighbor]) || nz >= __ldg(&map_offsets[neighbor + 1])))
        {
            neighbor++;
        }

        if (neighbor < num_neighbors && (nz >= __ldg(&map_offsets[neighbor]) && nz < __ldg(&map_offsets[neighbor + 1])))
        {
            INDEX_TYPE offset = map_offsets[neighbor];
            T *dest = dest_ptrs[neighbor];
            INDEX_TYPE *map = map_ptrs[neighbor];
            manager_utils::utils::atomic_add(&source[bsize * map[nz - offset] + vecIdx], dest[(nz - offset)*bsize + vecIdx]);
        }

        //dest[bsize*map[nz]+vecIdx] += source[nz*bsize+vecIdx];
        nz += nz_per_block * gridDim.x;
    }
}


template <class T>
__global__ void scatterFromBufferMin(T *dest, INDEX_TYPE *map, T *source, INDEX_TYPE bsize, INDEX_TYPE total)
{
    int nz_per_block = (blockDim.x / bsize);
    int nz = blockIdx.x * nz_per_block + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= nz_per_block * bsize) { return; }

    while (nz < total)
    {
        INDEX_TYPE mapping = map[nz];
        T old = dest[bsize * mapping + vecIdx];
        dest[bsize * mapping + vecIdx] = min(old, source[nz * bsize + vecIdx]);
        nz += nz_per_block * gridDim.x;
    }
}


template <typename TConfig> class DistributedManagerBase
{
    public:
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::VecPrec  value_type;
        typedef typename TConfig::MatPrec  mat_value_type;
        typedef typename TConfig::IndPrec  index_type;

        typedef typename TConfig::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type vvec_value_type;
        typedef typename TConfig::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_VecPrecision, TConfig::mode)>::Type vvec_value_type_v;
        typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_VecPrecision, TConfig::mode)>::Type vvec_value_type_vh;
        typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_d;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;

        typedef typename ivec_value_type_h::VecPrec VecInt_t;

        typedef Vector<vvec_value_type> VVector;
        typedef Vector<vvec_value_type_v> VVector_v;
        typedef Vector<vvec_value_type_vh> VVector_vh;

        typedef Vector<ivec_value_type> IVector;
        typedef Vector<ivec_value_type_h> IVector_h;
        typedef Vector<ivec_value_type_d> IVector_d;

        typedef Vector<i64vec_value_type> I64Vector;
        typedef Vector<i64vec_value_type_h> I64Vector_h;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        typedef std::vector<IVector_h> IVector_h_vector;
        typedef std::vector<IVector_d> IVector_d_vector;
        typedef Vector<TConfig_h> VVector_h;
        typedef std::vector<VVector_h> VVector_h_vector;

        typedef typename TConfig_h::template setVecPrec< types::PODTypes<typename TConfig::VecPrec >::vec_prec >::Type PODConfig_h;
        typedef typename TConfig_d::template setVecPrec< types::PODTypes<typename TConfig::VecPrec >::vec_prec >::Type PODConfig_d;

        typedef Vector<PODConfig_h> PODVector_h;
        typedef Vector<PODConfig_d> PODVector_d;

        typedef TemplateConfig<TConfig_h::memSpace, AMGX_vecDouble, TConfig_h::matPrec, TConfig_h::indPrec> dvec_value_type_h;
        typedef typename dvec_value_type_h::VecPrec  double_vec_type_h;
        typedef Vector<dvec_value_type_h> DVector_h;

        typedef TemplateConfig<TConfig_h::memSpace, AMGX_vecFloat, TConfig_h::matPrec, TConfig_h::indPrec> fvec_value_type_h;
        typedef typename fvec_value_type_h::VecPrec  float_vec_type_h;
        typedef Vector<fvec_value_type_h> FVector_h;

        typedef TemplateConfig<TConfig_h::memSpace, AMGX_vecComplex, TConfig_h::matPrec, TConfig_h::indPrec> cvec_value_type_h;
        typedef typename cvec_value_type_h::VecPrec  complex_vec_type_h;
        typedef Vector<cvec_value_type_h> CVector_h;

        typedef TemplateConfig<TConfig_h::memSpace, AMGX_vecDoubleComplex, TConfig_h::matPrec, TConfig_h::indPrec> zvec_value_type_h;
        typedef typename zvec_value_type_h::VecPrec  doublecomplex_vec_type_h;
        typedef Vector<zvec_value_type_h> ZVector_h;

        typedef typename i64vec_value_type_h::VecPrec  i64_vec_type_h;

        Matrix<TConfig> *A;  //my part of the matrix

        DistributedManagerBase() : m_fine_level_comms(NULL), _num_interior_nodes(0), m_pinned_buffer(NULL), m_pinned_buffer_size(0), _num_boundary_nodes(0), _comms(NULL), has_B2L(false),
            neighbors(_neighbors), B2L_maps(_B2L_maps), L2H_maps(_L2H_maps),  B2L_rings(_B2L_rings),
            halo_ranges(_halo_ranges), halo_rows_ref_count(0), halo_btl_ref_count(0), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h),  halo_rows(NULL), halo_btl(NULL), m_is_root_partition(false), m_is_glued(false), m_is_fine_level_glued(false), m_is_fine_level_consolidated(false), m_is_fine_level_root_partition(false), m_use_cuda_ipc_consolidation(false), m_fixed_view_size(false)

        {
            cudaEventCreate(&comm_event);
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
        };

        DistributedManagerBase(Matrix<TConfig> &a);

        DistributedManagerBase(Matrix<TConfig> &a,
                               Vector<ivec_value_type_h> neighbors_,
                               INDEX_TYPE interior_nodes_, INDEX_TYPE boundary_nodes_,
                               Vector<ivec_value_type_h> halo_offsets_,
                               amgx::thrust::host_vector<Vector<ivec_value_type> > B2L_maps_) : m_fine_level_comms(NULL),
            A(&a), m_pinned_buffer(NULL), m_pinned_buffer_size(0), _num_interior_nodes(interior_nodes_), _num_boundary_nodes(boundary_nodes_),
            neighbors(_neighbors), halo_offsets(halo_offsets_),
            B2L_maps(_B2L_maps),   L2H_maps(_L2H_maps), B2L_rings(_B2L_rings),
            halo_ranges(_halo_ranges), halo_rows_ref_count(0), halo_btl_ref_count(0), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h), halo_rows(NULL), halo_btl(NULL),
            _comms(NULL), m_is_root_partition(false), m_is_glued(false), m_is_fine_level_glued(false), m_is_fine_level_consolidated(false), m_is_fine_level_root_partition(false), m_use_cuda_ipc_consolidation(false), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            _neighbors = neighbors_;
            _B2L_maps.resize(B2L_maps_.size());

            for (int i = 0; i < B2L_maps_.size(); i++)
            {
                _B2L_maps[i] = B2L_maps_[i];
            }

            cudaEventCreate(&comm_event);
        };


        DistributedManagerBase( INDEX_TYPE my_id,
                                int64_t base_index, INDEX_TYPE index_range,
                                Matrix<TConfig> &a,
                                Vector<ivec_value_type_h> &neighbors_,
                                I64Vector &halo_ranges_,
                                std::vector<IVector > &B2L_maps_,
                                std::vector<std::vector<VecInt_t> > &B2L_rings_,
                                DistributedComms<TConfig> **comms_,
                                std::vector<Matrix<TConfig> > **halo_rows_,
                                std::vector<DistributedManager<TConfig> > **halo_btl_) : m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), neighbors(neighbors_), B2L_maps(B2L_maps_), L2H_maps(_L2H_maps), B2L_rings(B2L_rings_), halo_rows_ref_count(0), halo_btl_ref_count(0), halo_ranges(halo_ranges_), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            DistributedManagerBaseInit(my_id, base_index, index_range, a, comms_, halo_rows_, halo_btl_);
        }

        DistributedManagerBase( INDEX_TYPE my_id,
                                int64_t base_index, INDEX_TYPE index_range,
                                Matrix<TConfig> &a,
                                Vector<ivec_value_type_h> &neighbors_,
                                I64Vector &halo_ranges_,
                                std::vector<IVector > &B2L_maps_,
                                std::vector<IVector > &L2H_maps_,
                                std::vector<std::vector<VecInt_t> > &B2L_rings_,
                                DistributedComms<TConfig> **comms_,
                                std::vector<Matrix<TConfig> > **halo_rows_,
                                std::vector<DistributedManager<TConfig> > **halo_btl_) : m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), neighbors(neighbors_), B2L_maps(B2L_maps_), L2H_maps(_L2H_maps), B2L_rings(B2L_rings_), halo_rows_ref_count(0), halo_btl_ref_count(0), halo_ranges(halo_ranges_), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            DistributedManagerBaseInit(my_id, base_index, index_range, a, comms_, halo_rows_, halo_btl_);
        }

        DistributedManagerBase( INDEX_TYPE my_id,
                                int64_t base_index, INDEX_TYPE index_range,
                                Matrix<TConfig> &a,
                                Vector<ivec_value_type_h> &neighbors_,
                                I64Vector &halo_ranges_,
                                std::vector<IVector > &B2L_maps_,
                                std::vector<std::vector<VecInt_t> > &B2L_rings_,
                                DistributedComms<TConfig> **comms_) : m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), neighbors(neighbors_), halo_ranges(halo_ranges_),
            halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h),
            B2L_maps(B2L_maps_),  L2H_maps(_L2H_maps), B2L_rings(B2L_rings_), m_is_root_partition(false), m_is_glued(false), m_is_fine_level_glued(false), m_is_fine_level_consolidated(false), m_is_fine_level_root_partition(false), m_use_cuda_ipc_consolidation(false), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            DistributedArranger<TConfig> *prep = new DistributedArranger<TConfig>;
            int rings = num_halo_rings(B2L_rings);
            prep->create_B2L_from_maps(a, my_id, rings, base_index, index_range, neighbors,
                                       halo_ranges, B2L_maps, B2L_rings, comms_, &halo_rows, &halo_btl);
            DistributedManagerBaseInit(my_id, base_index, index_range, a, comms_, NULL, NULL);
            delete prep;
        }

        DistributedManagerBase( INDEX_TYPE my_id,
                                Matrix<TConfig> &a, INDEX_TYPE rings,
                                Vector<ivec_value_type_h>  &neighbors_,
                                std::vector<IVector > &B2L_maps_,
                                std::vector<IVector > &L2H_maps_,
                                DistributedComms<TConfig> **comms_) : m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), neighbors(neighbors_), halo_ranges(_halo_ranges),
            halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h),
            B2L_maps(B2L_maps_),  L2H_maps(L2H_maps_), B2L_rings(_B2L_rings), m_is_root_partition(false), m_is_glued(false), m_is_fine_level_glued(false), m_is_fine_level_consolidated(false), m_is_fine_level_root_partition(false), m_use_cuda_ipc_consolidation(false), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            DistributedArranger<TConfig> *prep = new DistributedArranger<TConfig>;
            prep->create_B2L_from_maps(a, my_id, rings, neighbors,
                                       B2L_maps, L2H_maps, B2L_rings, comms_, &halo_rows, &halo_btl);
            DistributedManagerBaseInit(my_id, 0, a.get_num_rows(), a, comms_, NULL, NULL);
            delete prep;
        }

        void exchange1RingHaloRows();

        DistributedManagerBase(
            Matrix<TConfig> &a,
            INDEX_TYPE allocated_halo_depth,
            INDEX_TYPE num_import_rings,
            int max_num_neighbors,
            const VecInt_t *neighbors_);

        void cacheMaps(const VecInt_t *b2l_maps, const VecInt_t *b2l_ptrs, const VecInt_t *l2h_maps, const VecInt_t *l2h_ptrs);

        void cacheMapsOneRing();

        void cacheMapsOneRing(const VecInt_t **b2l_maps, const VecInt_t *b2l_sizes, const VecInt_t **l2h_maps, const VecInt_t *l2h_sizes);

        void setAConsolidationFlags( Matrix<TConfig> &A);

        void uploadMatrix(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag_data, Matrix<TConfig> &A);

        void updateMapsReorder();

        void initializeUploadReorderAll(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag_data, Matrix<TConfig> &A);

        // create comms + initialize basic internal variables 
        void initComms(Resources *rsrc);

        void createComms(Resources *rsrc);

        void destroyComms();

        DistributedManagerBase( INDEX_TYPE my_id, INDEX_TYPE rings,
                                int64_t base_index, INDEX_TYPE index_range,
                                Matrix<TConfig> &a,
                                Vector<ivec_value_type_h> &neighbors_,
                                I64Vector_h &halo_ranges_h_,
                                DistributedComms<TConfig> **comms_) : m_fine_level_comms(NULL), _comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), neighbors(neighbors_), halo_ranges(_halo_ranges), halo_ranges_h(halo_ranges_h_), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h),
            B2L_maps(_B2L_maps),  L2H_maps(_L2H_maps), B2L_rings(_B2L_rings), m_is_root_partition(false), m_is_glued(false), m_is_fine_level_glued(false), m_is_fine_level_consolidated(false), m_is_fine_level_root_partition(false), m_use_cuda_ipc_consolidation(false), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            DistributedArranger<TConfig> *prep = new DistributedArranger<TConfig>;
            prep->create_B2L_from_neighbors(a, my_id, rings, base_index, index_range, neighbors,
                                            halo_ranges_h, halo_ranges, B2L_maps, L2H_maps, B2L_rings, comms_, &halo_rows, &halo_btl);
            DistributedManagerBaseInit(my_id, base_index,  index_range, a, comms_, NULL, NULL);
            delete prep;
        }

        DistributedManagerBase( INDEX_TYPE my_id, INDEX_TYPE rings,
                                int64_t base_index, INDEX_TYPE index_range,
                                Matrix<TConfig> &a,
                                const VecInt_t *neighbors_,
                                const VecInt_t *neighbor_bases,
                                const VecInt_t *neighbor_sizes,
                                int num_neighbors) : m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), neighbors(_neighbors), halo_ranges(_halo_ranges), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h),
            B2L_maps(_B2L_maps),  L2H_maps(_L2H_maps), B2L_rings(_B2L_rings), m_is_root_partition(false), m_is_glued(false), m_is_fine_level_glued(false), m_is_fine_level_consolidated(false), m_is_fine_level_root_partition(false), m_use_cuda_ipc_consolidation(false), m_fixed_view_size(false)
        {
            cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
            this->createComms(A->getResources());
            DistributedArranger<TConfig> *prep = new DistributedArranger<TConfig>;
            neighbors.resize(num_neighbors);
            amgx::thrust::copy(&neighbors_[0], &neighbors_[num_neighbors], neighbors.begin());
            I64Vector_h tmp_halo_ranges_h;
            tmp_halo_ranges_h.resize(2 * num_neighbors);

            for (int i = 0; i < num_neighbors; i++)
            {
                tmp_halo_ranges_h[2 * i] = neighbor_bases[i];
                tmp_halo_ranges_h[2 * i + 1] = neighbor_sizes[i];
            }

            prep->create_B2L_from_neighbors(a, my_id, rings, base_index, index_range, neighbors,
                                            tmp_halo_ranges_h, halo_ranges, B2L_maps, L2H_maps, B2L_rings, &_comms, &halo_rows, &halo_btl);
            DistributedManagerBaseInit(my_id, base_index, index_range, a, &_comms, NULL, NULL);
            delete prep;
        }

        void initializeAfterConsolidation(
            INDEX_TYPE my_id,
            Matrix<TConfig> &A_,
            Vector<ivec_value_type_h> neighbors_,
            INDEX_TYPE interior_nodes_,
            INDEX_TYPE boundary_nodes_,
            INDEX_TYPE total_num_rows,
            Vector<ivec_value_type_h> halo_offsets_,
            std::vector<IVector > &B2L_maps_,
            INDEX_TYPE ring_,
            bool is_root_partition_)
        {
            A = &A_;
            this->set_global_id(my_id);
            _num_interior_nodes = interior_nodes_;
            _num_boundary_nodes = boundary_nodes_;
            neighbors = neighbors_;
            halo_offsets = halo_offsets_;
            B2L_maps = B2L_maps_;
            m_is_root_partition = is_root_partition_;
            this->set_num_halo_rows(total_num_rows - halo_offsets[0]);
            this->set_num_halo_rings(ring_);
        }

        virtual void reorder_matrix() = 0;
        virtual void reorder_matrix_owned() = 0;

        virtual void obtain_shift_l2g_reordering(index_type n, I64Vector &l2g, IVector &p, IVector &q) = 0;
        virtual void unpack_partition(index_type *Bp, index_type *Bc, mat_value_type *Bv) = 0;

        virtual void generatePoisson7pt(int nx, int ny, int nz, int P, int Q, int R) = 0;

        virtual void renumberMatrixOneRing(int update_neighbours = 0) = 0;

        virtual void renumber_P_R(Matrix<TConfig> &P, Matrix<TConfig> &R, Matrix<TConfig> &A) = 0;

        virtual void createOneRingB2Lmaps() = 0;

        virtual void createOneRingHaloRows() = 0;

        void computeDestinationPartitions(INDEX_TYPE upper_threshold, float avg_size, const int num_parts, int &new_num_parts, bool &wantNeighbors);

        void computeDestinationPartitionsWithCons(int my_id, int num_parts, IVector_h &destination_part, DistributedComms<TConfig> *comms);

        Vector<ivec_value_type_h> &getDestinationPartitions()
        {
            return m_destination_partitions;
        }
        Vector<ivec_value_type_h> &getFineDestinationPartitions()
        {
            return m_destination_partitions;
        }
        void setDestinationPartitions(Vector<ivec_value_type_h> &destination_partitions)
        {
            m_destination_partitions = destination_partitions;
        }

        void createNeighToDestPartMap(IVector_h &neigh_to_part, IVector_h &neighbors, IVector_h &destination_part, int num_neighbors);

        void createConsolidatedNeighToPartMap(IVector_h &cons_neigh_to_part, IVector_h &neigh_to_part, int my_destination_part, IVector_h &destination_part, int &num_cons_neighbors);

        void createNeighToConsNeigh(IVector_h &neigh_to_cons_neigh, IVector_h &cons_neigh_to_part, IVector_h &neigh_to_part, int my_destination_part, int &num_neighbors);



        void consolidateB2Lmaps(IVector_h_vector &dest_coarse_B2L_maps, IVector_h_vector &coarse_B2L_maps, IVector_h &fine_neigh_to_coarse_neigh, int num_coarse_neighbors, int num_fine_neighbors);
        void consolidateB2Lmaps(IVector_d_vector &dest_coarse_B2L_maps, IVector_d_vector &coarse_B2L_maps, IVector_h &fine_neigh_to_coarse_neigh, int num_coarse_neighbors, int num_fine_neighbors);

        template <class IVector_hd>
        void consB2Lmaps(std::vector<IVector_hd> &dest_coarse_B2L_maps, std::vector<IVector_hd> &coarse_B2L_maps, IVector_h &fine_neigh_to_coarse_neigh, int num_coarse_neighbors, int num_fine_neighbors);

        void computeConsolidatedOffsets(const int my_id, const int my_destination_part, const bool sis_root_partition, const int num_interior_rows, const int num_boundary_rows, IVector_h_vector &vertex_counts, const IVector_h &parts_to_consolidate, const int num_parts_to_consolidate, int &interior_offset, int &boundary_offset, int &total_interior_rows_in_merged, int &total_boundary_rows_in_merged, int &total_rows_in_merged, DistributedComms<TConfig> *comms);

        void createAggregatesRenumbering(IVector_d &renumbering, IVector_d_vector &B2L_maps, int size, int num_neighbors, int &num_interior_aggregates, int &num_boundary_aggregates, int num_rings);
        void createAggregatesRenumbering(IVector_h &renumbering, IVector_h_vector &B2L_maps, int size, int num_neighbors, int &num_interior_aggregates, int &num_boundary_aggregates, int num_rings);

        template<class IVector_hd>
        void createAggRenumbering(IVector_hd &renumbering, std::vector<IVector_hd> &B2L_maps, int size, int num_neighbors, int &num_interior_aggregates, int &num_boundary_aggregates, int num_rings);


        void consolidateB2LmapsOnRoot(int &num_consolidated_neighbors, IVector_d_vector &consolidated_B2L_maps, IVector_h &consolidated_coarse_ids, IVector_d_vector &dest_coarse_B2L_maps, IVector_h &coarse_neigh_to_fine_part, IVector_h &num_bdy_per_coarse_neigh, IVector_h &fine_parts_to_consolidate, int num_fine_parts_to_consolidate, int my_id, int my_destination_part, bool is_root_partition, int num_coarse_neighbors, DistributedComms<TConfig> *comms);

        void consolidateB2LmapsOnRoot(int &num_consolidated_neighbors, IVector_h_vector &consolidated_B2L_maps, IVector_h &consolidated_coarse_ids, IVector_h_vector &dest_coarse_B2L_maps, IVector_h &coarse_neigh_to_fine_part, IVector_h &num_bdy_per_coarse_neigh, IVector_h &fine_parts_to_consolidate, int num_fine_parts_to_consolidate, int my_id, int my_destination_part, bool is_root_partition, int num_coarse_neighbors, DistributedComms<TConfig> *comms);

        template<class IVector_hd>
        void consB2LmapsOnRoot(int &num_consolidated_neighbors, std::vector<IVector_hd> &consolidated_B2L_maps, IVector_h &consolidated_coarse_ids, std::vector<IVector_hd> &dest_coarse_B2L_maps, IVector_h &coarse_neigh_to_fine_part, IVector_h &num_bdy_per_coarse_neigh, IVector_h &fine_parts_to_consolidate, int num_fine_parts_to_consolidate, int my_id, int my_destination_part, bool is_root_partition, int num_coarse_neighbors, DistributedComms<TConfig> *comms);


        void consolidateAndRenumberHalos(IVector_h &aggregates, const IVector_h &manager_halo_offsets, IVector_h &halo_offsets, const IVector_h &neighbors, int num_fine_neighbors, const IVector_h &consolidated_coarse_ids, int num_consolidated_neighbors, const IVector_h &destination_part, int my_destination_part, bool is_root_partition, IVector_h &fine_parts_to_consolidate, int num_fine_parts_to_consolidate, int num_parts, int my_id, int total_rows_in_merged, int &num_all_aggregates, DistributedComms<TConfig> *comms);

        void consolidateAndRenumberHalos(IVector_d &aggregates, const IVector_h &manager_halo_offsets, IVector_h &halo_offsets, const IVector_h &neighbors, int num_fine_neighbors, const IVector_h &consolidated_coarse_ids, int num_consolidated_neighbors, const IVector_h &destination_part, int my_destination_part, bool is_root_partition, IVector_h &fine_parts_to_consolidate, int num_fine_parts_to_consolidate, int num_parts, int my_id, int total_rows_in_merged, int &num_all_aggregates, DistributedComms<TConfig> *comms);

        template<class IVector_hd>
        void consAndRenumberHalos(IVector_hd &aggregates, const IVector_h &manager_halo_offsets, IVector_h &halo_offsets, const IVector_h &neighbors, int num_fine_neighbors, const IVector_h &consolidated_coarse_ids, int num_consolidated_neighbors, const IVector_h &destination_part, int my_destination_part, bool is_root_partition, IVector_h &fine_parts_to_consolidate, int num_fine_parts_to_consolidate, int num_parts, int my_id, int total_rows_in_merged, int &num_all_aggregates, DistributedComms<TConfig> *comms);

        void ipcExchangePtr(void *&ptr, bool is_root_partition, int num_parts_to_consolidate, IVector_h &parts_to_consolidate, int my_root_partition, int my_id, DistributedComms<TConfig> *comms);

        void ipcWaitForChildren(bool is_root_partition, int num_parts_to_consolidate, IVector_h &parts_to_consolidate, int my_destination_part, int my_id, DistributedComms<TConfig> *comms);

        void ipcWaitForRoot(bool is_root_partition, int num_parts_to_consolidate, IVector_h &parts_to_consolidate, int my_destination_part, int my_id, DistributedComms<TConfig> *comms);

        void remove_boundary(IVector_h &flagArray, IVector_h &B2L_maps, int size);
        void remove_boundary(IVector_d &flagArray, IVector_d &B2L_maps, int size);
        void get_unassigned(IVector_h &flagArray, IVector_h &B2L_maps, IVector_h &partition_flags, int size, int flagArray_size /*, int rank*/);
        void get_unassigned(IVector_d &flagArray, IVector_d &B2L_maps, IVector_d &partition_flags, int size, int flagArray_size /*, int rank*/);
        void set_unassigned(IVector_h &partition_flags, IVector_h &partition_renum, IVector_h &B2L_map, IVector_h &renumbering, int size, int max_element, int renumbering_size /*, int rank*/);
        void set_unassigned(IVector_d &partition_flags, IVector_d &partition_renum, IVector_d &B2L_map, IVector_d &renumbering, int size, int max_element, int renumbering_size /*, int rank*/);

        void exchangeSolveResultsConsolidation(int &num_iters, std::vector<PODVector_h> &res_history, AMGX_STATUS &status, bool store_res_history);

        void flag_halo_ids(int size, IVector_h &scratch, IVector_h &halo_aggregates, VecInt_t min_index_coarse_halo, int max_index, int min_index) ;
        void flag_halo_ids(int size, IVector_d &scratch, IVector_d &halo_aggregates, VecInt_t min_index_coarse_halo, int max_index, int min_index) ;
        void read_halo_ids(int size, IVector_h &scratch, IVector_h &halo_aggregates, VecInt_t min_index_coarse_halo);
        void read_halo_ids(int size, IVector_d &scratch, IVector_d &halo_aggregates, VecInt_t min_index_coarse_halo);

        void checkPinnedBuffer(size_t size);
        // if pointer is host pointer - returns data. If it is device pointer - copies it to the m_pinned_buffer and returns pointer to m_pinned_buffer
        void *getHostPointerForData(void *ptr, size_t size, int *allocated);
        void *getDevicePointerForData(void *ptr, size_t size, int *allocated);
        const void *getHostPointerForData(const void *ptr, size_t size, int *allocated);
        const void *getDevicePointerForData(const void *ptr, size_t size, int *allocated);

        virtual ~DistributedManagerBase();

        //
        // Level 0 API
        //

        DistributedComms<TConfig>  *getComms() { return _comms; }
        DistributedComms<TConfig>  *getFineLevelComms() const { return m_fine_level_comms; }

        void setMatrix(Matrix<TConfig> &A)
        {
            this->A = &A;
        }

        void setComms(DistributedComms<TConfig> &comms_obj)
        {
            if ( _comms != NULL && _comms->decr_ref_count())
            {
                delete _comms;
            }

            _comms = &comms_obj;
            comms_obj.incr_ref_count();
        }

        void setComms(DistributedComms<TConfig> *comms_obj)
        {
            if ( _comms != NULL && _comms->decr_ref_count())
            {
                delete _comms;
            }

            _comms = comms_obj;
            comms_obj->incr_ref_count();
        }

        void set_index_range(INDEX_TYPE i)
        {
            _index_range = i;
        }

        INDEX_TYPE index_range() const
        {
            return _index_range;
        }

        inline void set_base_index(int64_t i)
        {
            _base_index = i;
        }

        inline int64_t base_index() const
        {
            return (int64_t) _base_index;
        }

        bool isRootPartition() const
        {
            return m_is_root_partition;
        }
        bool isGlued() const
        {
            return m_is_glued;
        }
        bool isFineLevelGlued() const
        {
            return m_is_fine_level_glued;
        }
        bool isFineLevelRootPartition() const
        {
            return m_is_fine_level_root_partition;
        }

        bool isFineLevelConsolidated() const
        {
            return m_is_fine_level_consolidated;
        }
        void setIsFineLevelConsolidated(const bool flag)
        {
            m_is_fine_level_consolidated = flag;
        }
        void setIsFineLevelGlued(const bool flag)
        {
            m_is_fine_level_glued = flag;
        }
        void setIsFineLevelRootPartition(const bool flag)
        {
            m_is_fine_level_root_partition = flag;
        }
        void setIsRootPartition(bool flag)
        {
            m_is_root_partition = flag;
        }
        void setIsGlued(const bool flag)
        {
            m_is_glued = flag;
        }
        void fineLevelUpdate()
        {
            m_is_fine_level_root_partition = m_is_root_partition;
            m_num_fine_level_parts_to_consolidate = m_num_parts_to_consolidate;
            m_fine_level_parts_to_consolidate = m_parts_to_consolidate;
            m_my_fine_level_destination_part = m_my_destination_part;
            m_fine_level_comms = _comms;
            m_fine_level_id = _global_id;
            // other data structure used on the finest level
            //fine_level_id
            //get_unconsolidated_size
            //getFineLevelComms
        }

        INDEX_TYPE getMyDestinationPartition()
        {
            return m_my_destination_part;
        }

        INDEX_TYPE getNumPartsToConsolidate()
        {
            return m_num_parts_to_consolidate;
        }

        void setNumPartsToConsolidate(INDEX_TYPE num_fine_parts)
        {
            m_num_parts_to_consolidate = num_fine_parts;
        }
        void setMyDestinationPartition(INDEX_TYPE my_destination_part)
        {
            m_my_destination_part = my_destination_part;
        }

        void setPartsToConsolidate(Vector<ivec_value_type_h> &parts_to_consolidate)
        {
            m_parts_to_consolidate = parts_to_consolidate;
        }

        Vector<ivec_value_type_h> &getPartsToConsolidate(void)
        {
            return m_parts_to_consolidate;
        }

        void setB2Lrings(std::vector<std::vector<VecInt_t> > &par_B2L_rings)
        {
            B2L_rings = par_B2L_rings;
        }

        std::vector<std::vector<VecInt_t> > &getB2Lrings(void)
        {
            return B2L_rings;
        }
        Vector<ivec_value_type_h> &getHaloOffsets(void)
        {
            return halo_offsets;
        }
        std::vector<IVector> &getB2Lmaps(void)
        {
            return B2L_maps;
        }
        void setCoarseToFine(Vector<ivec_value_type_h> &coarse_to_fine_part)
        {
            m_coarse_to_fine_part = coarse_to_fine_part;
        }

        Vector<ivec_value_type_h> &getCoarseToFine(void)
        {
            return m_coarse_to_fine_part;
        }
        void setFineToCoarse(Vector<ivec_value_type_h> &fine_to_coarse_part)
        {
            m_fine_to_coarse_part = fine_to_coarse_part;
        }

        Vector<ivec_value_type_h> &getFineToCoarse(void)
        {
            return m_fine_to_coarse_part;
        }

        void setConsolidationOffsets(int int_off, int int_size, int bndry_off, int bndry_size)
        {
            m_cons_interior_offset = int_off;
            m_cons_interior_size = int_size;
            m_cons_bndry_offset = bndry_off;
            m_cons_bndry_size = bndry_size;
        }

        void getConsolidationOffsets(int *int_off, int *int_size, int *bndry_off, int *bndry_size)
        {
            *int_off = m_cons_interior_offset;
            *int_size = m_cons_interior_size;
            *bndry_off = m_cons_bndry_offset;
            *bndry_size = m_cons_bndry_size;
        }

        void setConsolidationArrayOffsets(std::vector<VecInt_t> &array)
        {
            m_consolidationArrayOffsets = array;
        }

        std::vector<VecInt_t> &getConsolidationArrayOffsets()
        {
            return m_consolidationArrayOffsets;
        }

        void set_fine_level_id(INDEX_TYPE id)
        {
            m_fine_level_id = id;
        }

        void set_global_id(INDEX_TYPE id)
        {
            _global_id = id;
        }

        INDEX_TYPE global_id() const
        {
            return (INDEX_TYPE) _global_id;
        }

        inline void set_num_partitions(INDEX_TYPE num_partitions)
        {
            _num_partitions = num_partitions;
        }

        inline INDEX_TYPE get_num_partitions() const
        {
            return (INDEX_TYPE) _num_partitions;
        }

        inline INDEX_TYPE fine_level_id() const
        {
            return (INDEX_TYPE) m_fine_level_id;
        }

        void set_num_halo_rows(INDEX_TYPE n)
        {
            _num_halo_rows = n;
        }

        INDEX_TYPE num_halo_rows() const
        {
            return (INDEX_TYPE) _num_halo_rows;
        }

        void set_num_halo_rings(INDEX_TYPE n)
        {
            _num_halo_rings = n;
        }

        INDEX_TYPE num_halo_rings() const
        {
            return (INDEX_TYPE) _num_halo_rings;
        }

        INDEX_TYPE num_neighbors() const
        {
            return (INDEX_TYPE) neighbors.size();
        }

        INDEX_TYPE halo_offset(int idx) const
        {
            return (INDEX_TYPE) halo_offsets[idx];
        }

        INDEX_TYPE num_halo_offsets() const
        {
            return (INDEX_TYPE) halo_offsets.size();
        }

        DistributedManagerBase<TConfig> &operator=(const DistributedManagerBase<TConfig_h> &a)
        {
            this->copy(a);
            return *this;
        }

        DistributedManagerBase<TConfig> &operator=(const DistributedManagerBase<TConfig_d> &a)
        {
            this->copy(a);
            return *this;
        }

        template<class TConfig_hd>
        void copy(const DistributedManagerBase<TConfig_hd> &a)
        {
            set_base_index(a.base_index());
            set_global_id(a.global_id());
            set_num_partitions(a.get_num_partitions());
            set_index_range(a.index_range());
            set_num_halo_rows(a.num_halo_rows());
            _num_interior_nodes = a.num_interior_nodes();
            _num_boundary_nodes = a.num_boundary_nodes();
            neighbors = a.neighbors;
            halo_offsets = a.halo_offsets;
            B2L_rings = a.B2L_rings;
            B2L_maps.resize(a.B2L_maps.size());
            L2H_maps.resize(a.L2H_maps.size());
            set_num_halo_rings(a.num_halo_rings());
            halo_ranges = a.halo_ranges;
            halo_ranges_h = a.halo_ranges;
            part_offsets = a.part_offsets;
            part_offsets_h = a.part_offsets_h;
            num_rows_per_part = a.num_rows_per_part;

            for (int i = 0; i < B2L_maps.size(); i++)
            {
                B2L_maps[i] = a.B2L_maps[i];
            }

            for (int i = 0; i < L2H_maps.size(); i++)
            {
                L2H_maps[i] = a.L2H_maps[i];
            }

            m_is_root_partition = a.isRootPartition();
            m_is_glued = a.isGlued();
            m_is_fine_level_glued = a.isFineLevelGlued();
            destroyComms();
            //since you have a copy you should not free the memory
            halo_rows = NULL;
            halo_btl = NULL;
            halo_rows_ref_count = 1;
            halo_btl_ref_count = 1;
        }

        template <class TConfig_hd>
        inline void swap(DistributedManagerBase<TConfig_hd> &a)
        {
            int64_t temp_long;
            temp_long = base_index();
            set_base_index(a.base_index());
            a.set_base_index(temp_long);
            INDEX_TYPE temp;
            temp = index_range();
            set_index_range(a.index_range());
            a.set_index_range(temp);
            temp = num_halo_rows();
            set_num_halo_rows(a.num_halo_rows());
            a.set_num_halo_rows(temp);
            temp = num_halo_rings();
            set_num_halo_rings(a.num_halo_rings());
            a.set_num_halo_rings(temp);
            temp = global_id();
            set_global_id(a.global_id());
            set_num_partitions(a.get_num_partitions());
            neighbors.swap(a.neighbors);
            halo_ranges.swap(a.halo_ranges);
            halo_ranges_h.swap(a.halo_ranges_h);
            part_offsets.swap(a.part_offsets);
            part_offsets_h.swap(a.part_offsets_h);
            num_rows_per_part.swap(a.num_rows_per_part);
            temp = _num_interior_nodes;
            _num_interior_nodes = a.num_interior_nodes();
            temp = _num_boundary_nodes;
            _num_boundary_nodes = a.num_boundary_nodes();
            halo_offsets.swap(a.halo_offsets);
            B2L_maps.swap(a.B2L_maps);
            L2H_maps.swap(a.L2H_maps);
            B2L_rings.swap(a.B2L_rings);
            bool tmp = m_is_root_partition;
            m_is_root_partition = a.isRootPartition();
            a.setIsRootPartition(tmp);
            tmp = m_is_glued;
            m_is_glued = a.isGlued();
            a.setIsGlued(tmp);
            tmp = m_is_fine_level_glued;
            m_is_glued = a.isFineLevelGlued();
            a.setIsFineLevelGlued(tmp);
        }

        void print(char *f, char *s, int trank);

        void printToFile(char *f, char *s);

        int compare(DistributedManagerBase<TConfig> *m2);

        void resize(INDEX_TYPE num_neigbors, INDEX_TYPE rings)
        {
            set_num_halo_rings(rings);
            neighbors.resize(num_neigbors);
            B2L_maps.resize(num_neigbors);
            halo_ranges.resize(num_neigbors * 2);
            B2L_rings.resize(num_neigbors);

            for (int i = 0; i < num_neigbors; i++) { B2L_rings[i].resize(rings + 1); }

            halo_offsets.resize(num_neigbors * num_halo_rings() + 1);
        }

        template <class Vector>
        void exchange_halo(Vector &data, int tag)
        {
            if (_comms == NULL) {data.dirtybit = 0; return;}

            if (this->num_neighbors() == 0) { return; }

            if (_comms->exchange_halo_query(data, *A, comm_event) || data.dirtybit == 0) { return; } //if single node/already arrived/not dirty return
            else if (data.in_transfer & RECEIVING)
            {
                this->exchange_halo_wait(data, tag);  //blocking wait if we are already receiving
                return;
            }

            _comms->setup(data, *A, tag);  //set pointers to buffer
            gather_B2L(data);             //write values to buffer
            _comms->exchange_halo(data, *A, tag, 1); //exchange buffers
            scatter_L2H(data);            //NULL op
        }

        template <class Vector>
        void exchange_halo_v2(Vector &data, int tag)
        {
            if (_comms == NULL) {data.dirtybit = 0; return;}

            if (_comms->exchange_halo_query(data, *A, comm_event) || data.dirtybit == 0 || (data.in_transfer & RECEIVING)) { return; } //if single node/already arrived/not dirty/already receiving return

            _comms->setup(data, *A, tag);  //set pointers to buffer
            gather_B2L_v2(data);             //write values to buffer
            _comms->exchange_halo(data, *A, tag); //begin async send
        }

        template <class Vector>
        void exchange_halo_async(Vector &data, int tag)
        {
            if (_comms == NULL) {data.dirtybit = 0; return;}

            if (_comms->exchange_halo_query(data, *A, comm_event) || data.dirtybit == 0 || (data.in_transfer & RECEIVING)) { return; } //if single node/already arrived/not dirty/already receiving return

            _comms->setup(data, *A, tag);  //set pointers to buffer
            gather_B2L(data);             //write values to buffer
            cudaEventRecord(comm_event);
            _comms->exchange_halo_async(data, *A, comm_event, tag); //begin async send
        }

        template <class Vector>
        void exchange_halo_split_gather(Vector &data, int tag)
        {
            if (_comms == NULL) {data.dirtybit = 0; return;}

            if (_comms->exchange_halo_query(data, *A, comm_event) || data.dirtybit == 0 || (data.in_transfer & RECEIVING)) { return; } //if single node/already arrived/not dirty/already receiving return

            _comms->setup(data, *A, tag);  //set pointers to buffer
            gather_B2L_v2(data);             //write values to buffer
            cudaEventRecord(comm_event);
        }

        template <class Vector>
        void exchange_halo_split_finish(Vector &data, int tag)
        {
            if (_comms == NULL) {data.dirtybit = 0; return;}

            if ((data.in_transfer == IDLE) && (data.dirtybit == 0)) { return; } //if single node/not dirty and no data transfer - return

            if (_comms->exchange_halo_query(data, *A, comm_event) || data.dirtybit == 0 || (data.in_transfer & RECEIVING)) { return; } //if single node/already arrived/not dirty/already receiving return
            _comms->exchange_halo_async(data, *A, comm_event, tag, m_bdy_stream); //begin async send
            _comms->exchange_halo_wait(data, *A, comm_event, tag, m_bdy_stream); //blocking wait
            scatter_L2H(data);            //NULL op
        }

        template <class Vector>
        void exchange_halo_wait(Vector &data, int tag)
        {
            if (_comms == NULL) {data.dirtybit = 0; return;}

            if ((data.in_transfer == IDLE) && (data.dirtybit == 0)) { return; } //if single node/not dirty and no data transfer - return

            _comms->exchange_halo_wait(data, *A, comm_event, tag); //blocking wait
            scatter_L2H(data);            //NULL op
        }

        template <class Vector>
        void exchange_halo_2ring(Vector &data, int tag)
        {
            _comms->setup(data, *A, tag, 2);  //set pointers to buffer
            gather_B2L(data, 2);             //write values to buffer
            _comms->exchange_halo(data, *A, tag, 2); //exchange buffers
            scatter_L2H(data);            //NULL op
        }

        template <class Vector>
        void add_from_halo(Vector &data, int tag, cudaStream_t stream = 0)
        {
            // set num neighbors = size of b2l_rings
            // need to do this because comms might have more neighbors than our matrix knows about
            _comms->set_neighbors(B2L_rings.size());
            _comms->setup_L2H(data, *A);                           //set pointers to buffer
            _comms->gather_L2H(data, *A, 1, stream);            //write values to buffer
            _comms->add_from_halo(data, *A, tag, 1, stream);      //exchange buffers
            scatter_B2L(data);                                     // update values
        }

        template <class Vector>
        void add_from_halo_v2(Vector &data, int tag, cudaStream_t stream = 0)
        {
            // set num neighbors = size of b2l_rings
            // need to do this because comms might have more neighbors than our matrix knows about
            _comms->set_neighbors(B2L_rings.size());
            _comms->setup_L2H(data, *A);                           //set pointers to buffer
            _comms->gather_L2H_v2(data, *A, 1, stream);            //write values to buffer
            _comms->add_from_halo(data, *A, tag, 1, stream);      //exchange buffers
            scatter_B2L(data);                                     // update values
        }

        template <class Vector>
        void add_from_halo_split_gather(Vector &data, int tag, cudaStream_t stream = 0)
        {
            // set num neighbors = size of b2l_rings
            // need to do this because comms might have more neighbors than our matrix knows about
            _comms->set_neighbors(B2L_rings.size());
            _comms->setup_L2H(data, *A);                           //set pointers to buffer
            _comms->gather_L2H_v2(data, *A, 1, stream);            //write values to buffer
        }

        template <class Vector>
        void add_from_halo_split_finish(Vector &data, int tag, cudaStream_t stream = 0)
        {
            _comms->add_from_halo(data, *A, tag, 1, stream);      //exchange buffers
            scatter_B2L_v2(data);                                  // update values
        }

        template <class Vector>
        void min_from_halo(Vector &data, int tag)
        {
            // set num neighbors = size of b2l_rings
            // need to do this because comms might have more neighbors than our matrix knows about
            _comms->set_neighbors(B2L_rings.size());
            _comms->setup_L2H(data, *A);                                //set pointers to buffer
            _comms->gather_L2H(data, *A, 1);                            //write values to buffer
            cudaStream_t null_stream = 0;
            _comms->add_from_halo(data, *A, tag, 1, null_stream);      //exchange buffers
            scatter_B2L_min(data);                                      // update values
        }

        template <class Vector>
        void send_receive_wait(Vector &data, int tag)
        {
            _comms->send_receive_wait(data, *A, comm_event, tag, m_int_stream); //begin async send
        }

        void export_neighbors(VecInt_t *neighbors_e)
        {
            amgx::thrust::copy(neighbors.begin(), neighbors.end(), &neighbors_e[0]);
        }

        void malloc_export_maps(VecInt_t ***b2l_maps_e, VecInt_t **b2l_maps_ptrs_e, VecInt_t ***l2h_maps_e, VecInt_t **l2h_maps_ptrs_e);

        void export_halo_ranges(VecInt_t *halo_ranges_e)
        {
            amgx::thrust::copy(halo_ranges.begin(), halo_ranges.end(), &halo_ranges_e[0]);
        }

        // scalar reductions
        void global_reduce_sum(double_vec_type_h *value)
        {
            DVector_h own(1);
            DVector_h res(1);
            own[0] = *value;
            _comms->global_reduce_sum(res, own, *A, 1);
            *value = res[0];
        }

        void global_reduce_sum(float_vec_type_h *value)
        {
            FVector_h own(1);
            FVector_h res(1);
            own[0] = *value;
            _comms->global_reduce_sum(res, own, *A, 1);
            *value = res[0];
        }

        void global_reduce_sum(index_type *value)
        {
            IVector_h own(1);
            IVector_h res(1);
            own[0] = *value;
            _comms->global_reduce_sum(res, own, *A, 1);
            *value = res[0];
        }

        void global_reduce_sum(i64_vec_type_h *value)
        {
            I64Vector_h own(1);
            I64Vector_h res(1);
            own[0] = *value;
            _comms->global_reduce_sum(res, own, *A, 1);
            *value = res[0];
        }

        void global_reduce_sum(complex_vec_type_h *value)
        {
            CVector_h own(1);
            CVector_h res(1);
            own[0] = *value;
            _comms->global_reduce_sum(res, own, *A, 1);
            *value = res[0];
        }

        void global_reduce_sum(doublecomplex_vec_type_h *value)
        {
            ZVector_h own(1);
            ZVector_h res(1);
            own[0] = *value;
            _comms->global_reduce_sum(res, own, *A, 1);
            *value = res[0];
        }

        //Create renumbering to separate interior/boundary nodes based on B2L_maps
        virtual void createRenumbering(IVector &renumbering) = 0;

        void set_initialized(IVector &row_offsets);

        INDEX_TYPE num_interior_nodes() const
        {
            return _num_interior_nodes;
        }

        INDEX_TYPE num_boundary_nodes() const
        {
            return _num_boundary_nodes;
        }

        INDEX_TYPE num_rows_all() const
        {
            return _num_rows_all;
        }

        INDEX_TYPE num_nz_all() const
        {
            return _num_nz_all;
        }

        inline INDEX_TYPE num_rows_interior() const
        {
            return _num_rows_interior;
        }

        inline INDEX_TYPE num_rows_owned() const
        {
            return _num_rows_owned;
        }

        inline INDEX_TYPE num_rows_full() const
        {
            return _num_rows_full;
        }

        inline void getView(const ViewType type, INDEX_TYPE &num_rows, INDEX_TYPE &num_nz) const
        {
            if (type == INTERIOR)
            {
                num_rows = _num_rows_interior;
                num_nz = _num_nz_interior;
            }
            else if (type == OWNED)
            {
                num_rows = _num_rows_owned;
                num_nz = _num_nz_owned;
            }
            else if (type == FULL)
            {
                num_rows = _num_rows_full;
                num_nz = _num_nz_full;
            }
            else if (type == ALL)
            {
                num_rows = _num_rows_all;
                num_nz = _num_nz_all;
            }
        }

        //return row offset into matrix and number of rows for a given view
        void getOffsetAndSizeForView(ViewType type, int *offset, int *size) const
        {
            if (type == INTERIOR)
            {
                *offset = 0;
                *size = _num_rows_interior;
            }
            else if (type == BOUNDARY)
            {
                *offset = _num_rows_interior;
                *size = _num_rows_owned - _num_rows_interior;
            }
            else if (type == OWNED)
            {
                *offset = 0;
                *size = _num_rows_owned;
            }
            else if (type == HALO1)
            {
                *offset = _num_rows_owned;
                *size = _num_rows_full - _num_rows_owned;
            }
            else if (type == FULL)
            {
                *offset = 0;
                *size = _num_rows_full;
            }
            else if (type == HALO2)
            {
                *offset = _num_rows_full;
                *size = _num_rows_all - _num_rows_full;
            }
            else if (type == ALL)
            {
                *offset = 0;
                *size = _num_rows_all;
            }
            else if (type == (BOUNDARY | HALO1))
            {
                *offset = _num_rows_interior;
                *size = _num_rows_full - _num_rows_interior;
            }
            else if (type == (BOUNDARY | HALO1 | HALO2))
            {
                *offset = _num_rows_interior;
                *size = _num_rows_all - _num_rows_interior;
            }
            else if (type == (HALO1 | HALO2))
            {
                *offset = _num_rows_owned;
                *size = _num_rows_all - _num_rows_owned;
            }
        }

        //return row offset into matrix and number of rows for a given view
        inline void getNnzForView(ViewType type, int *nnz) const
        {
            if (type == INTERIOR)
            {
                *nnz = _num_nz_interior;
            }
            else if (type == BOUNDARY)
            {
                *nnz = _num_nz_owned - _num_nz_interior;
            }
            else if (type == OWNED)
            {
                *nnz = _num_nz_owned;
            }
            else if (type == HALO1)
            {
                *nnz = _num_nz_full - _num_nz_owned;
            }
            else if (type == FULL)
            {
                *nnz = _num_nz_full;
            }
            else if (type == HALO2)
            {
                *nnz = _num_nz_all - _num_nz_full;
            }
            else if (type == ALL)
            {
                *nnz = _num_nz_all;
            }
            else if (type == (BOUNDARY | HALO1))
            {
                *nnz = _num_nz_full - _num_nz_interior;
            }
            else if (type == (BOUNDARY | HALO1 | HALO2))
            {
                *nnz = _num_nz_all - _num_nz_interior;
            }
            else if (type == (HALO1 | HALO2))
            {
                *nnz = _num_nz_all - _num_nz_owned;
            }
        }

    protected:

        //template<CommType> friend void DistributedManager<TConfig>::copy(const CommType&);
        friend class energymin::Energymin_AMG_Level_Base<TConfig>;
        friend class classical::Classical_AMG_Level_Base<TConfig>;
        friend class aggregation::Aggregation_AMG_Level_Base<TConfig>;
        friend class DistributedManagerBase<TConfig_h>;
        friend class DistributedManagerBase<TConfig_d>;

        //
        // Level 0 API related
        //
        DistributedComms<TConfig> *_comms;    //LEVEL 0 - pointer to comms module
        std::vector<IVector > &B2L_maps;       //LEVEL 0 - list of boundary nodes to export to other partitions. Ordering corresponds to "neighbors"
        std::vector<IVector > &L2H_maps;       //LEVEL 0 - list of halo nodes as indexed in our local matrix (during execution this is a sequence, needed for setup). Ordering corresponds to "neighbors"
        std::vector<std::vector<VecInt_t> > &B2L_rings; //LEVEL 0 - offsets int B2L_maps[i]: number of boundary nodes per ring. Ordering corresponds to "neighbors"

        std::vector<IVector_h> cached_B2L_maps;
        std::vector<IVector_h> cached_L2H_maps;

        // Consolidation related
        DistributedComms<TConfig> *m_fine_level_comms;    //LEVEL 0 - pointer to comms module

        bool m_is_fine_level_consolidated;
        bool m_use_cuda_ipc_consolidation;
        bool m_host_transform;
        int m_fine_level_id;
        int m_old_nnz_CONS;


        IVector_h B2L_rings_sizes;
        std::vector<IVector> B2L_maps_offsets;
        device_vector_alloc<int *> B2L_maps_ptrs;

        // For fine level consolidation without IPC
        std::vector<IVector> m_child_row_ids;
        std::vector<IVector> m_child_old_row_offsets;
        IVector m_row_ids_CONS;
        IVector m_old_row_offsets_CONS;
        std::vector<IVector_h> m_child_row_ids_h;
        std::vector<IVector_h> m_child_old_row_offsets_h;
        IVector_h m_row_ids_CONS_h;
        IVector_h m_old_row_offsets_CONS_h;

        void *m_pinned_buffer;
        size_t m_pinned_buffer_size;

        IVector_h m_child_n;
        IVector_h m_child_nnz;
        IVector_h m_child_num_halos;
        int m_child_max_nnz;
        int m_child_max_n;
        // End of level 0 API related

        cudaEvent_t comm_event;

        //
        // Internal variables
        //
        INDEX_TYPE _num_interior_nodes;            //number of interior nodes, row indices: 0->interior_nodes-1
        INDEX_TYPE _num_boundary_nodes;            //number of boundary nodes, row indices: interior_nodes->interior_nodes+boundary_nodes-1

        //[interior boundary n0h0 n1h0 ... n0h1 n1h1 ...]

        IVector old_row_offsets;

        int halo_rows_ref_count;
        int halo_btl_ref_count;
        std::vector<Matrix<TConfig> > *halo_rows;
        std::vector<DistributedManager<TConfig>  > *halo_btl;

        bool has_B2L;

        void DistributedManagerBaseInit(INDEX_TYPE my_id,
                                        int64_t base_index, INDEX_TYPE index_range,
                                        Matrix<TConfig> &a,
                                        DistributedComms<TConfig> **comms_,
                                        std::vector<Matrix<TConfig> > **halo_rows_,
                                        std::vector<DistributedManager<TConfig> > **halo_btl_)
        {
            A = &a;
            _num_interior_nodes = 0;
            _num_boundary_nodes = 0;
            has_B2L = false;
            _comms = *comms_;
            cudaEventCreate(&comm_event);
            set_base_index(base_index);
            set_index_range(index_range);
            set_global_id(my_id);
            int num_neighbors = neighbors.size();
            _comms->set_neighbors(num_neighbors);
            _num_halo_rings = num_halo_rings(B2L_rings);

            if (halo_rows_ != NULL)
            {
                halo_rows = *halo_rows_;
                halo_rows_ref_count = 1;
            }

            if (halo_btl_ != NULL)
            {
                halo_btl = *halo_btl_;
                halo_btl_ref_count = 1;
            }
        }

        INDEX_TYPE num_halo_rings(std::vector<std::vector<VecInt_t> > &B2L_rings)
        {
            int num_neighbors = neighbors.size();
            return  (B2L_rings.size() - 1) / num_neighbors;
        }

        //gather values to buffer
        template <class Vector>
        void gather_B2L(Vector &b, int num_rings = 1)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (neighbors.size() > 0)
                {
                    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
                }
            }
            else
            {
#ifdef DEBUG
                if (b.get_block_size() != 1 || b.get_block_size() != this->A->get_block_dimx() || b.get_block_size() != this->A->get_block_dimy()) { printf("UNRECOGNIZED Vector blocksize!!\n"); }
#endif

                for (int i = 0; i < this->neighbors.size(); i++)
                {
                    int size = this->B2L_rings[i][num_rings];
                    int num_blocks = std::min(4096, (size + 127) / 128);

                    if ( size != 0)
                    {
                        if (b.get_num_cols() == 1)
                        {
                            gatherToBuffer <<< num_blocks, 128>>>(b.raw(), this->B2L_maps[i].raw(), b.linear_buffers[i], b.get_block_size(), size);
                        }
                        else
                        {
                            gatherToBufferMultivector<<<num_blocks, 128>>>(b.raw(), this->B2L_maps[i].raw(), b.linear_buffers[i], b.get_num_cols(), b.get_lda(), size);
                        }

                        cudaCheckError();
                    }
                }
            }
        }

        //gather values to buffer
        template <class Vector>
        void gather_B2L_v2(Vector &b, int num_rings = 1)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (neighbors.size() > 0)
                {
                    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
                }
            }
            else
            {
#ifdef DEBUG

                if (b.get_block_size() != 1 || b.get_block_size() != this->A->get_block_dimx() || b.get_block_size() != this->A->get_block_dimy())
                {
                    printf("UNRECOGNIZED Vector blocksize!!\n");
                }

#endif

                if (b.get_num_cols() != 1)
                {
                    FatalError("num_cols != 1 not supported in gather_B2L_v2\n", AMGX_ERR_NOT_IMPLEMENTED);
                }

                int size = this->B2L_rings_sizes[num_rings - 1];
                int num_blocks = std::min(4096, (size + 127) / 128);
                int num_neighbors = this->neighbors.size();

                if (size != 0)
                {
                    gatherToBuffer_v3 <<< num_blocks, 128>>>(
                        b.raw(),
                        this->B2L_maps_offsets[num_rings - 1].raw(),
                        amgx::thrust::raw_pointer_cast(&B2L_maps_ptrs[0]),
                        amgx::thrust::raw_pointer_cast(&b.linear_buffers_ptrs[0]),
                        b.get_block_size(),
                        size,
                        num_neighbors);
                }
            }
        }


        //add values from buffer
        template <class Vector>
        void scatter_B2L(Vector &b, int num_rings = 1)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (neighbors.size() > 0)
                {
                    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
                }
            }
            else
            {
#ifdef DEBUG

                if (b.get_block_size() != 1 || b.get_block_size() != this->A->get_block_dimx() || b.get_block_size() != this->A->get_block_dimy()) { printf("UNRECOGNIZED Vector blocksize!!\n"); }

#endif

                for (int i = 0; i < this->neighbors.size(); i++)
                {
                    int size = this->B2L_rings[i][num_rings];

                    if (size != 0)
                    {
                        int num_blocks = std::min(4096, (size + 127) / 128);
                        scatterFromBuffer <<< num_blocks, 128>>>(b.raw(), this->B2L_maps[i].raw(), b.linear_buffers[i], b.get_block_size(), size);
                        cudaCheckError();
                    }
                }
            }
        }

        //add values from buffer
        template <class Vector>
        void scatter_B2L_v2(Vector &b, int num_rings = 1)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (neighbors.size() > 0)
                {
                    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
                }
            }
            else
            {
#ifdef DEBUG

                if (b.get_block_size() != 1 || b.get_block_size() != this->A->get_block_dimx() || b.get_block_size() != this->A->get_block_dimy()) { printf("UNRECOGNIZED Vector blocksize!!\n"); }

#endif
                int size = this->B2L_rings_sizes[num_rings - 1];
                int num_blocks = std::min(4096, (size + 127) / 128);
                int num_neighbors = this->neighbors.size();

                if (size != 0)
                {
                    scatterFromBuffer_v3 <<< num_blocks, 128>>>(
                        b.raw(),
                        this->B2L_maps_offsets[num_rings - 1].raw(),
                        amgx::thrust::raw_pointer_cast(&B2L_maps_ptrs[0]),
                        amgx::thrust::raw_pointer_cast(&b.linear_buffers_ptrs[0]),
                        b.get_block_size(),
                        size,
                        num_neighbors);
                    cudaCheckError();
                }
            }
        }

        //add values from buffer
        template <class Vector>
        void scatter_B2L_min(Vector &b, int num_rings = 1)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (neighbors.size() > 0)
                {
                    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
                }
            }
            else
            {
#ifdef DEBUG

                if (b.get_block_size() != 1 || b.get_block_size() != this->A->get_block_dimx() || b.get_block_size() != this->A->get_block_dimy()) { printf("UNRECOGNIZED Vector blocksize!!\n"); }

#endif

                for (int i = 0; i < this->neighbors.size(); i++)
                {
                    int size = this->B2L_rings[i][num_rings];

                    if (size != 0)
                    {
                        int num_blocks = std::min(4096, (size + 127) / 128);
                        scatterFromBufferMin <<< num_blocks, 128>>>(b.raw(), this->B2L_maps[i].raw(), b.linear_buffers[i], b.get_block_size(), size);
                        cudaCheckError();
                    }
                }
            }
        }


        template <class Vector>
        void scatter_L2H(Vector &b, int num_rings = 1)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (neighbors.size() > 0)
                {
                    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
                }
            }
            else
            {
                //do nothing...
            }
        }
    protected:
        int64_t _base_index;                //LEVEL 0 - the index of the first node owned by this partition
        INDEX_TYPE _index_range = 0;               //LEVEL 0 - the number of fine nodes owned by this partition
        INDEX_TYPE _global_id = 0;                 //LEVEL 0 - ID of this node (partition)
        INDEX_TYPE _num_partitions = 0;            //LEVEL 0 - Number of partitions (partition)
        INDEX_TYPE _num_halo_rows = 0;             //LEVEL 0 -  total number of rows in the halo section of the matrix
        INDEX_TYPE _num_halo_rings = 0;            //LEVEL 0 -  number of halo rings

        bool m_is_root_partition;
        bool m_is_glued;
        bool m_is_fine_level_glued;
        INDEX_TYPE m_my_destination_part = 0;
        INDEX_TYPE m_num_parts_to_consolidate = 0;
        INDEX_TYPE m_cons_interior_offset = 0;
        INDEX_TYPE m_cons_interior_size = 0;
        INDEX_TYPE m_cons_bndry_offset = 0;
        INDEX_TYPE m_cons_bndry_size = 0;
        std::vector<VecInt_t> m_consolidationArrayOffsets;
        Vector<ivec_value_type_h> m_destination_partitions;

        Vector<ivec_value_type_h> m_parts_to_consolidate;
        Vector<ivec_value_type_h> m_fine_to_coarse_part;
        Vector<ivec_value_type_h> m_coarse_to_fine_part;

        // fine level consolidation data structures (used both in classical aggregation)
        bool m_is_fine_level_root_partition;
        INDEX_TYPE m_num_fine_level_parts_to_consolidate;
        Vector<ivec_value_type_h> m_fine_level_parts_to_consolidate;
        INDEX_TYPE m_my_fine_level_destination_part;

        //cached sizes for different views of the matrix (set in Matrix::set_initialized(1))
        INDEX_TYPE _num_rows_interior = 0;
        INDEX_TYPE _num_nz_interior = 0;
        INDEX_TYPE _num_rows_owned = 0;
        INDEX_TYPE _num_nz_owned = 0;
        INDEX_TYPE _num_rows_full = 0;
        INDEX_TYPE _num_nz_full = 0;
        INDEX_TYPE _num_rows_all = 0;
        INDEX_TYPE _num_nz_all = 0;
        bool m_fixed_view_size;

        //Containers for Level 0 API:
        std::vector<IVector >_B2L_maps;
        std::vector<IVector >_L2H_maps;
        std::vector<std::vector<VecInt_t> > _B2L_rings;

        I64Vector_h num_rows_per_part {};

    public:

        I64Vector_h &part_offsets_h;
        I64Vector &part_offsets;
        I64Vector_h &halo_ranges_h;
        I64Vector &halo_ranges;

        I64Vector_h _part_offsets_h;
        I64Vector _part_offsets;
        I64Vector_h _halo_ranges_h;
        I64Vector _halo_ranges;

        I64Vector local_to_global_map;

        IVector renumbering;
        IVector inverse_renumbering;

        // Containers to store info of the unglued matrix, we need that glue or unglue vectors
        IVector renumbering_before_glue; // to glue vectors during the solve, used in glue path.
        IVector inverse_renumbering_before_glue; // to unglue vectors before prolongation in fixed cycle, used in glue path.
        // we need that to exchange the halo of unglued vectors (in coarse level consolidation)
        Vector<ivec_value_type_h> neighbors_before_glue;  // just neighbors before glue
        std::vector<IVector > B2L_maps_before_glue;
        std::vector<std::vector<VecInt_t> > B2L_rings_before_glue; //list of boundary nodes to export to other partitions.
        Vector<ivec_value_type_h> halo_offsets_before_glue;

        Vector<ivec_value_type_h> halo_offsets; //offsets (and size) to halos received from different neighbors, size is halo_rings*neighbors.size()+1, first element is already the offset into the matrix

        Vector<ivec_value_type_h> &neighbors;  //LEVEL 0 - list of neighbors with their global index, in the order we store their halos
        Vector<ivec_value_type_h> _neighbors;

        cudaStream_t m_int_stream;
        cudaStream_t m_bdy_stream;
        cudaStream_t null_stream = NULL;

        IVector boundary_rows_list;
        IVector interior_rows_list;
        IVector halo1_rows_list;

        inline cudaStream_t& get_int_stream() { return m_int_stream; }
        inline cudaStream_t& get_bdy_stream() { return m_bdy_stream; }
        inline cudaEvent_t& get_comm_event() { return comm_event; }

        int64_t num_rows_global = 0;

        const I64Vector_h& getNumRowsPerPart()
        {
            if(_comms == nullptr)
            {
                FatalError("Calling getNumRowsPerPart with no communicator", AMGX_ERR_INTERNAL);
            }

            if(_num_rows_owned <= 0)
            {
                FatalError("_num_rows_owned <= 0 when determining num rows per part", AMGX_ERR_INTERNAL);
            }

            if(_num_partitions <= 0)
            {
                _num_partitions = _comms->get_num_partitions();
            }

            // If necessary, populate the number of rows per partition
            if(num_rows_per_part.size() == 0)
            {
                _comms->all_gather(_num_rows_owned, num_rows_per_part, _num_partitions); 
            }

            return num_rows_per_part;
        }

        const IVector &getRowsListForView(ViewType type)
        {
            if (type == INTERIOR)
            {
                return this->interior_rows_list;
            }
            else if (type == BOUNDARY)
            {
                return this->boundary_rows_list;
            }
            else
            {
                FatalError("getRowListForView not implemented for this view", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }

        // Manually set the view sizes
        inline void setViewSizes(int num_interior_nodes, int num_nz_interior, int num_rows_owned, int num_nz_owned, int num_rows_full, int num_nz_full, int num_rows_all, int num_nz_all)
        {
            this->_num_rows_interior = num_interior_nodes;
            this->_num_nz_interior = num_nz_interior;
            this->_num_rows_owned = num_rows_owned;
            this->_num_nz_owned = num_nz_owned;
            this->_num_rows_full = num_rows_full;
            this->_num_nz_full = num_nz_full;
            this->_num_rows_all = num_rows_all;
            this->_num_nz_all = num_nz_all;

            // Avoids the view sizes being overwritten by set_initialized
            this->m_fixed_view_size = true;
        }

        inline bool isViewSizeFixed() { return this->m_fixed_view_size; }

};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class DistributedManager< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public DistributedManagerBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::VecPrec  value_type;
        typedef typename TConfig::MatPrec  mat_value_type;
        typedef typename TConfig::IndPrec  index_type;
        typedef typename DistributedManagerBase<TConfig>::VVector VVector;
        typedef typename DistributedManagerBase<TConfig>::VVector_v VVector_v;
        typedef typename DistributedManagerBase<TConfig>::VVector_vh VVector_vh;
        typedef typename DistributedManagerBase<TConfig>::IVector IVector;
        typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_d;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;

        typedef Matrix<TConfig_h> Matrix_h;

        typedef Vector<ivec_value_type_h> IVector_h;
        typedef Vector<ivec_value_type_d> IVector_d;
        typedef typename ivec_value_type_h::VecPrec VecInt_t;

        typedef Vector<i64vec_value_type_h> I64Vector_h;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        typedef std::vector<IVector_h> IVector_h_vector;
        typedef std::vector<IVector_d> IVector_d_vector;

        void reorder_matrix();
        void reorder_matrix_owned();

        void obtain_shift_l2g_reordering(index_type n, I64Vector_h &l2g, IVector_h &p, IVector_h &q);
        void unpack_partition(index_type *Bp, index_type *Bc, mat_value_type *Bv);

        void generatePoisson7pt(int nx, int ny, int nz, int P, int Q, int R);
        template <typename t_colIndex>
        void loadDistributedMatrix(int num_rows, int num_nonzeros, const int block_dimx, const int block_dimy, const int *row_offsets, const t_colIndex *col_indices, const mat_value_type *values, int num_ranks, int num_rows_global, const void *diag_data, const MatrixDistribution &dist);
        void renumberMatrixOneRing(int update_neighbours = 0);
        void renumber_P_R(Matrix<TConfig_h> &P, Matrix<TConfig_h> &R, Matrix<TConfig_h> &A);
        void createOneRingB2Lmaps();
        void createOneRingHaloRows();
        void consolidateAndUploadAll(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag_data, Matrix<TConfig> &in_A) ;

        void replaceMatrixCoefficientsNoCons(int n, int nnz, const mat_value_type *data, const mat_value_type *diag_data);
        void replaceMatrixCoefficientsWithCons(int n, int nnz, const mat_value_type *data, const mat_value_type *diag_data);
        void transformAndUploadVector(VVector_v &v, const void *data, int n, int block_dim);
        void transformVector(VVector_v &v);
        void transformAndUploadVectorWithCons(VVector_v &v, const void *data, int n, int block_dim);
        void revertAndDownloadVector(VVector_v &v, const void *data, int n, int block_dimy);
        void revertVector(VVector_v &v);
        void revertVector(VVector_v &v_in, VVector_v &v_out);
        void revertAndDownloadVectorWithCons(VVector_v &v, const void *data, int n, int block_dimy);
        void createRenumbering(IVector &renumbering);

        //constructors
        DistributedManager() : DistributedManagerBase<TConfig>() {}
        DistributedManager(Matrix<TConfig> &a) : DistributedManagerBase<TConfig>(a) {}

        DistributedManager( INDEX_TYPE my_id,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_h> &a,
                            Vector<ivec_value_type_h> &neighbors,
                            I64Vector_h &halo_ranges,
                            std::vector<IVector > &B2L_maps,
                            std::vector<std::vector<VecInt_t> > &B2L_rings,
                            DistributedComms<TConfig_h> **comms) {FatalError("Importing data is only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);};

        DistributedManager( INDEX_TYPE my_id, INDEX_TYPE rings,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_h> &a,
                            Vector<ivec_value_type_h> &neighbors,
                            I64Vector_h &halo_ranges_h,
                            DistributedComms<TConfig_h> **comms) {FatalError("Importing data is only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);};

        DistributedManager(
            Matrix<TConfig_h> &a,
            INDEX_TYPE allocated_halo_depth,
            INDEX_TYPE num_import_rings,
            int max_num_neighbors,
            const VecInt_t *neighbors_) {FatalError("Importing data is only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);};

        inline DistributedManager(
            Matrix<TConfig_h> &a,
            INDEX_TYPE nx,
            INDEX_TYPE ny,
            INDEX_TYPE nz,
            INDEX_TYPE P,
            INDEX_TYPE Q,
            INDEX_TYPE R)
        {
            FatalError("Generate distributed poisson only supported on device", AMGX_ERR_NOT_IMPLEMENTED);
        };

        inline DistributedManager( INDEX_TYPE my_id, INDEX_TYPE rings,
                                   int64_t base_index, INDEX_TYPE index_range,
                                   Matrix<TConfig_h> &a,
                                   const VecInt_t *neighbors_h,
                                   const VecInt_t *neighbor_bases,
                                   const VecInt_t *neighbor_sizes,
                                   int num_neighbors)  {FatalError("Importing data is only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);};

        DistributedManager(const DistributedManager<TConfig_h> &a) { DistributedManagerBase<TConfig_h>::copy(a); }
        DistributedManager(const DistributedManager<TConfig_d> &a) { DistributedManagerBase<TConfig_h>::copy(a); }

        //destructor
        virtual ~DistributedManager();
        friend class CommsMultiDeviceBase<TConfig_d>;
        friend class CommsMultiDeviceBase<TConfig_h>;
        friend class CommsMPIHostBuffer<TConfig_d>;
        friend class CommsMPIHostBuffer<TConfig_h>;
        friend class CommsMPIHostBufferStream<TConfig_d>;
        friend class CommsMPIHostBufferStream<TConfig_h>;
        friend class CommsMPIHostBufferAsync<TConfig_d>;
        friend class CommsMPIHostBufferAsync<TConfig_h>;
        friend class CommsMPIDirect<TConfig_d>;
        friend class CommsMPIDirect<TConfig_h>;
        friend class CommsSingleDeviceBase<TConfig_d>;
        friend class CommsSingleDeviceBase<TConfig_h>;
    private:
        std::vector<Matrix<TConfig_h> > *matrix_halos;
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class DistributedManager< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public DistributedManagerBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig::IndPrec  index_type;
        typedef typename TConfig::VecPrec  value_type;
        typedef typename TConfig::MatPrec  mat_value_type;
        typedef typename DistributedManagerBase<TConfig>::VVector VVector;
        typedef typename DistributedManagerBase<TConfig>::VVector_v VVector_v;
        typedef typename DistributedManagerBase<TConfig>::VVector_vh VVector_vh;
        typedef typename DistributedManagerBase<TConfig>::IVector IVector;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_d;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;

        typedef Matrix<TConfig_d> Matrix_d;


        typedef typename ivec_value_type_h::VecPrec VecInt_t;
        typedef Vector<ivec_value_type_h> IVector_h;
        typedef Vector<ivec_value_type_d> IVector_d;

        typedef Vector<i64vec_value_type_h> I64Vector_h;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        typedef std::vector<IVector_h> IVector_h_vector;
        typedef std::vector<IVector_d> IVector_d_vector;

        void consolidateAndUploadAll(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag_data, Matrix<TConfig> &A) ;

        void reorder_matrix();
        void reorder_matrix_owned();

        void obtain_shift_l2g_reordering(index_type n, I64Vector_d &l2g, IVector_d &p, IVector_d &q);
        void unpack_partition(index_type *Bp, index_type *Bc, mat_value_type *Bv);

        void generatePoisson7pt(int nx, int ny, int nz, int P, int Q, int R);
        template <typename t_colIndex>
        void loadDistributedMatrix(int num_rows, int num_nonzeros, const int block_dimx, const int block_dimy, const int *row_offsets, const t_colIndex *col_indices, const mat_value_type *values, int num_ranks, int num_rows_global, const void *diag_data, const MatrixDistribution &dist);
        void renumberMatrixOneRing(int update_neighbours = 0);
        void renumber_P_R(Matrix<TConfig_d> &P, Matrix<TConfig_d> &R, Matrix<TConfig_d> &A);
        void createOneRingB2Lmaps();
        void createOneRingHaloRows();
        void replaceMatrixCoefficientsNoCons(int n, int nnz, const mat_value_type *data_pinned, const mat_value_type *diag_data_pinned);
        void replaceMatrixCoefficientsWithCons(int n, int nnz,  const mat_value_type *data_pinned, const mat_value_type *diag_data_pinned);
        void transformAndUploadVector(VVector_v &v, const void *data, int n, int block_dim);
        void transformVector(VVector_v &v);
        void transformAndUploadVectorWithCons(VVector_v &v, const void *data, int n, int block_dim);

        void revertAndDownloadVector(VVector_v &v, const void *data, int n, int block_dimy);
        void revertVector(VVector_v &v);
        void revertVector(VVector_v &v_in, VVector_v &v_out);
        void revertAndDownloadVectorWithCons(VVector_v &v, const void *data, int n, int block_dimy);

        void createRenumbering(IVector &renumbering);

        //constructors
        DistributedManager() : DistributedManagerBase<TConfig>() {}
        DistributedManager(Matrix<TConfig> &a) : DistributedManagerBase<TConfig>(a) {}

        DistributedManager(const DistributedManager<TConfig_h> &a) { DistributedManagerBase<TConfig_d>::copy(a); }
        DistributedManager(const DistributedManager<TConfig_d> &a) { DistributedManagerBase<TConfig_d>::copy(a); }

        DistributedManager( INDEX_TYPE my_id,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_d> &a,
                            Vector<ivec_value_type_h> &neighbors,
                            I64Vector_d &halo_ranges,
                            std::vector<IVector> &B2L_maps,
                            std::vector<std::vector<VecInt_t> > &B2L_rings,
                            DistributedComms<TConfig_d> **comms,
                            std::vector<Matrix<TConfig_d>  > **halo_rows,
                            std::vector<DistributedManager<TConfig_d> > **halo_btl) : DistributedManagerBase<TConfig_d>(my_id, base_index, index_range, a,
                                        neighbors, halo_ranges, B2L_maps, B2L_rings, comms, halo_rows, halo_btl) {}

        DistributedManager( INDEX_TYPE my_id,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_d> &a,
                            Vector<ivec_value_type_h> &neighbors,
                            I64Vector_d &halo_ranges,
                            std::vector<IVector> &B2L_maps,
                            std::vector<IVector> &L2H_maps,
                            std::vector<std::vector<VecInt_t> > &B2L_rings,
                            DistributedComms<TConfig_d> **comms,
                            std::vector<Matrix<TConfig_d>  > **halo_rows,
                            std::vector<DistributedManager<TConfig_d> > **halo_btl) : DistributedManagerBase<TConfig_d>(my_id, base_index, index_range, a,
                                        neighbors, halo_ranges, B2L_maps, L2H_maps, B2L_rings, comms, halo_rows, halo_btl) {}


        DistributedManager( INDEX_TYPE my_id,
                            Matrix<TConfig_d> &a, INDEX_TYPE rings,
                            Vector<ivec_value_type_h> &neighbors,
                            std::vector<IVector> &B2L_maps,
                            std::vector<IVector> &L2H_maps,
                            DistributedComms<TConfig_d> **comms) : DistributedManagerBase<TConfig_d>(my_id, a, rings,
                                        neighbors, B2L_maps, L2H_maps, comms) {}

        DistributedManager( INDEX_TYPE my_id,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_d> &a,
                            Vector<ivec_value_type_h> &neighbors,
                            I64Vector_d &halo_ranges,
                            std::vector<IVector > &B2L_maps,
                            std::vector<std::vector<VecInt_t> > &B2L_rings,
                            DistributedComms<TConfig_d> **comms) : DistributedManagerBase<TConfig_d>(my_id, base_index, index_range, a,
                                        neighbors, halo_ranges, B2L_maps, B2L_rings, comms) {}

        DistributedManager(
            Matrix<TConfig_d> &a,
            INDEX_TYPE allocated_halo_depth,
            INDEX_TYPE num_import_rings,
            int max_num_neighbors,
            const VecInt_t *neighbors_) : DistributedManagerBase<TConfig_d>(a, allocated_halo_depth, num_import_rings, max_num_neighbors, neighbors_) {};

        DistributedManager( INDEX_TYPE my_id, INDEX_TYPE rings,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_d> &a,
                            Vector<ivec_value_type_h> &neighbors,
                            I64Vector_h &halo_ranges_h,
                            DistributedComms<TConfig_d> **comms) : DistributedManagerBase<TConfig_d>(my_id, rings, base_index, index_range, a,
                                        neighbors, halo_ranges_h, comms)  {}

        DistributedManager( INDEX_TYPE my_id, INDEX_TYPE rings,
                            int64_t base_index, INDEX_TYPE index_range,
                            Matrix<TConfig_d> &a,
                            const VecInt_t *neighbors_h,
                            const VecInt_t *neighbor_bases,
                            const VecInt_t *neighbor_sizes,
                            int num_neighbors) : DistributedManagerBase<TConfig_d>(my_id, rings, base_index, index_range, a,
                                        neighbors_h, neighbor_bases, neighbor_sizes, num_neighbors) {}

        //destructor
        virtual ~DistributedManager();

        friend class DistributedArranger<TConfig_d>;
        friend class CommsMultiDeviceBase<TConfig_d>;
        friend class CommsMultiDeviceBase<TConfig_h>;
        friend class CommsMPIHostBuffer<TConfig_d>;
        friend class CommsMPIHostBuffer<TConfig_h>;
        friend class CommsMPIHostBufferStream<TConfig_d>;
        friend class CommsMPIHostBufferStream<TConfig_h>;
        friend class CommsMPIHostBufferAsync<TConfig_d>;
        friend class CommsMPIHostBufferAsync<TConfig_h>;
        friend class CommsMPIDirect<TConfig_d>;
        friend class CommsMPIDirect<TConfig_h>;
        friend class CommsSingleDeviceBase<TConfig_d>;
        friend class CommsSingleDeviceBase<TConfig_h>;
    private:
        template <typename t_colIndex>
        void loadDistributed_SetOffsets(int num_ranks, int num_rows_global, const t_colIndex* partition_offsets);

        template <typename t_colIndex>
        std::map<t_colIndex, int> loadDistributed_LocalToGlobal(int num_rows, I64Vector_h &off_diag_cols);

        void loadDistributed_InitLocalMatrix(IVector_h local_col_indices, int num_rows, int num_nonzeros, const int block_dimx, const int block_dimy,
            const int *row_offsets, const mat_value_type *values, const void *diag);

        template <typename t_colIndex>
        void loadDistributedMatrixPartitionVec(int num_rows, int num_nonzeros, const int block_dimx, const int block_dimy, 
            const int *row_offsets, const t_colIndex *col_indices, const mat_value_type *values, int num_ranks, int num_rows_global, const void *diag, const int *partition);

        template <typename t_colIndex>
        void loadDistributedMatrixPartitionOffsets( int num_rows, int num_nonzeros, const int block_dimx, const int block_dimy, 
            const int *row_offsets, const t_colIndex *col_indices, const mat_value_type *values, int num_ranks, int num_rows_global, const void *diag, const t_colIndex *partition_offsets);
};
}

