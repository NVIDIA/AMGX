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

#ifdef _WIN32
#ifndef AMGX_API_EXPORTS
#define AMGX_API_EXPORTS
#endif
#endif

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <amg_solver.h>
#include "amg_signal.h"
#include <matrix_io.h>
#include "amgx_c.h"
#include "amgxP_c.h"
#include "../../core/include/version.h"
#include "distributed/distributed_manager.h"
#include "distributed/comms_mpi_gpudirect.h"
#include "distributed/comms_mpi_hostbuffer_stream.h"
#include "distributed/distributed_arranger.h"
#include "distributed/distributed_io.h"
#include "resources.h"
#include "matrix_distribution.h"
#include <amgx_timer.h>
#include "util.h"
#include "reorder_partition.h"
#include <algorithm>
#include <solvers/solver.h>
#include <matrix.h>
#include <vector.h>
#include <thrust_wrapper.h>

#include "amgx_types/util.h"
#include "amgx_types/rand.h"
#include "amgx_c_wrappers.inl"
#include "amgx_c_common.h"
#include "multiply.h"

namespace amgx
{

AMGX_RC getResourcesFromSolverHandle(AMGX_solver_handle slv, Resources **resources)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        AMGX_Mode mode = get_mode_from<AMGX_solver_handle>(slv);

        switch (mode)
        {
#define AMGX_CASE_LINE(CASE) case CASE: { \
          *resources = get_mode_object_from<CASE,AMG_Solver,AMGX_solver_handle>(slv)->getResources();\
          } \
          break;
                AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

            default:
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, NULL);
        }
    }

    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, NULL)
    return AMGX_RC_OK;
}

AMGX_RC getResourcesFromMatrixHandle(AMGX_matrix_handle mtx, Resources **resources)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

        switch (mode)
        {
#define AMGX_CASE_LINE(CASE) case CASE: { \
          *resources = get_mode_object_from<CASE,Matrix,AMGX_matrix_handle>(mtx)->getResources(); \
          } \
          break;
                AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

            default:
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, NULL);
        }
    }

    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, NULL)
    return AMGX_RC_OK;
}

AMGX_RC getResourcesFromVectorHandle(AMGX_vector_handle vec, Resources **resources)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

        switch (mode)
        {
#define AMGX_CASE_LINE(CASE) case CASE: { \
          *resources = get_mode_object_from<CASE,Vector,AMGX_vector_handle>(vec)->getResources(); \
          } \
          break;
                AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

            default:
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, NULL);
        }
    }

    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, NULL)
    return AMGX_RC_OK;
}
}


namespace amgx
{

//function object (functor) for thrust calls (it is a binary subtraction operator)
template<typename T>
class subtract_op
{
    public:
        subtract_op() {}
        __host__ __device__ T operator()(const T &x, const T &y) const
        {
            return y - x;
        }
};

//function object (functor) for thrust calls (it is a unary operator to add a constant)
template<typename T>
class add_constant_op
{
        const T c;
    public:
        add_constant_op(T _c) : c(_c) {}
        __host__ __device__ T operator()(const T &x) const
        {
            return x + c;
        }
};

#ifdef AMGX_WITH_MPI

template<class TConfig>
int create_part_offsets(int &root, int &rank, MPI_Comm &mpicm, Matrix<TConfig> *nv_mtx)
{
    /* WARNING: Notice that part_offsets_h & part_offsets have type int64_t.
                Therefore we need to use MPI_INT64_T (or MPI_LONG_LONG) in MPI_Allgather.
                Also, we need the send & recv buffers to be of the same type, therefore
                we will create a temporary variable n64 of the correct type below. */
    //create TConfig64, which is the same as TConfig, but with index type being int64_t
    typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type TConfig64;
    typedef typename TConfig64::VecPrec t_VecPrec; //t_VecPrec = int64_t
    int n, offset, mpist;
    int nranks = 0; //nv_mtx->manager->get_num_partitions();

    if (nv_mtx->manager != NULL)
    {
        //some initializations
        nv_mtx->getOffsetAndSizeForView(OWNED, &offset, &n);
        MPI_Comm_size(mpicm, &nranks);
        nv_mtx->manager->part_offsets_h.resize(nranks + 1);
        //printf("[%d,%d]: n=%d\n",rank,nranks,n);
        //gather the number of rows per partition on the host (on all ranks)
        t_VecPrec n64 = n;
        nv_mtx->manager->part_offsets_h[0] = 0; //first element is zero (the # of rows is gathered afterwards)

        if (typeid(t_VecPrec) == typeid(int64_t))
        {
            mpist = MPI_Allgather(&n64, 1, MPI_INT64_T, nv_mtx->manager->part_offsets_h.raw() + 1, 1, MPI_INT64_T, mpicm);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        //perform a prefix sum
        thrust::inclusive_scan(nv_mtx->manager->part_offsets_h.begin(), nv_mtx->manager->part_offsets_h.end(), nv_mtx->manager->part_offsets_h.begin());
        //create the corresponding array on device (this is important)
        nv_mtx->manager->part_offsets.resize(nranks + 1);
        thrust::copy(nv_mtx->manager->part_offsets_h.begin(), nv_mtx->manager->part_offsets_h.end(), nv_mtx->manager->part_offsets.begin());
    }

    return 0;
}

template<class TConfig>
int construct_global_matrix(int &root, int &rank, Matrix<TConfig> *nv_mtx, Matrix<TConfig> &gA, int &partition_vector_size, const int *partition_vector)
{
    typedef typename TConfig::IndPrec t_IndPrec;
    typedef typename TConfig::MatPrec t_MatPrec;
    int n, nnz, offset, l, k, i;
    int start, end, shift;
    int mpist;
    MPI_Comm mpicm;
    //MPI call parameters
    t_IndPrec *rc_ptr, *di_ptr;
    t_IndPrec *hli_ptr, *hgi_ptr;
    t_MatPrec *hlv_ptr, *hgv_ptr;
    thrust::host_vector<t_IndPrec> rc;
    thrust::host_vector<t_IndPrec> di;
    //unpacked local matrix on the device and host
    device_vector_alloc<t_IndPrec> Bp;
    device_vector_alloc<t_IndPrec> Bi;
    device_vector_alloc<t_MatPrec> Bv;
    thrust::host_vector<t_IndPrec> hBp;
    thrust::host_vector<t_IndPrec> hBi;
    thrust::host_vector<t_MatPrec> hBv;
    //constructed global matrix on the host
    thrust::host_vector<t_IndPrec> hAp;
    thrust::host_vector<t_IndPrec> hAi;
    thrust::host_vector<t_MatPrec> hAv;
    //WARNING: this routine currently supports matrix only with block size =1 (it can be generalized in the future, though)
    //initialize the defaults
    root = 0;
    rank = 0;
    mpist = MPI_SUCCESS;
    mpicm = MPI_COMM_WORLD;

    if (nv_mtx->manager != NULL)
    {
        // some initializations
        rank = nv_mtx->manager->global_id();

        if (nv_mtx->manager->getComms() != NULL)
        {
            mpicm = *(nv_mtx->getResources()->getMpiComm());
        }

        nv_mtx->getOffsetAndSizeForView(OWNED, &offset, &n);
        nv_mtx->getNnzForView(OWNED, &nnz);

        if (nv_mtx->manager->part_offsets_h.size() == 0)   // create part_offsets_h & part_offsets
        {
            create_part_offsets(root, rank, mpicm, nv_mtx); // (if needed for aggregation path)
        }

        l = nv_mtx->manager->part_offsets_h.size() - 1;    // number of partitions
        k = nv_mtx->manager->part_offsets_h[l];            // global number of rows
        //some allocations/resizing
        Bp.resize(n + 1);
        Bi.resize(nnz);
        Bv.resize(nnz);
        hBp.resize(n + 1);
        hBi.resize(nnz);
        hBv.resize(nnz);

        if (rank == root)
        {
            hAp.resize(k + 1); // extra +1 is needed because row_offsets have one extra element at the end
            //hAi.resize(global nnz); //not known yet
            //hAv.resize(global nnz); //not known yet
            rc.resize(l);
            di.resize(l);
        }

        cudaCheckError();
        //--- unpack the matrix ---
        nv_mtx->manager->unpack_partition(thrust::raw_pointer_cast(Bp.data()),
                                          thrust::raw_pointer_cast(Bi.data()),
                                          thrust::raw_pointer_cast(Bv.data()));
        cudaCheckError();
        //copy to host (should be able to optimize this out later on)
        hBp = Bp;
        hBi = Bi;
        hBv = Bv;
        cudaCheckError();

        // --- construct global matrix ---
        //Step 1. construct global row pointers
        //compute recvcounts and displacements for MPI_Gatherv
        if (rank == root)
        {
            thrust::transform(nv_mtx->manager->part_offsets_h.begin(), nv_mtx->manager->part_offsets_h.end() - 1, nv_mtx->manager->part_offsets_h.begin() + 1, rc.begin(), subtract_op<t_IndPrec>());
            cudaCheckError();
            //thrust::copy(nv_mtx->manager->part_offsets_h.begin(),nv_mtx->manager->part_offsets_h.end(),di.begin());
            thrust::transform(nv_mtx->manager->part_offsets_h.begin(), nv_mtx->manager->part_offsets_h.begin() + l, di.begin(), add_constant_op<t_IndPrec>(1));
            cudaCheckError();
        }

        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        rc_ptr = thrust::raw_pointer_cast(rc.data());
        di_ptr = thrust::raw_pointer_cast(di.data());
        hli_ptr = thrust::raw_pointer_cast(hBp.data());
        hgi_ptr = thrust::raw_pointer_cast(hAp.data());
        cudaCheckError();

        //gather (on the host)
        if (typeid(t_IndPrec) == typeid(int))
        {
            mpist = MPI_Gatherv(hli_ptr + 1, n, MPI_INT,  hgi_ptr, rc_ptr, di_ptr, MPI_INT,  root, mpicm);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        //Step 2. adjust row pointers, construct global column indices and values (recvcounts and displacements were computed above)
        if (rank == root)
        {
            //adjust global row pointers and setup the recvcounts & displacements for subsequent MPI calls
            for (i = 0; i < l; i++)
            {
                start = nv_mtx->manager->part_offsets_h[i];
                end  = nv_mtx->manager->part_offsets_h[i + 1];
                shift = hAp[start];
                //if (rank == 0) printf("# %d %d %d\n",start,end,shift);
                thrust::transform(hAp.begin() + start + 1, hAp.begin() + end + 1, hAp.begin() + start + 1, add_constant_op<t_IndPrec>(shift));
                cudaCheckError();
                di[i] = shift;
                rc[i] = hAp[end] - hAp[start];
                //if (rank == 0) printf("& %d %d %d\n",hAp[start],hAp[end],hAp[end]-hAp[start]);
            }

            //some allocations/resizing
            hAi.resize(hAp[k]); //now we know global nnz and can allocate storage
            hAv.resize(hAp[k]); //now we know global nnz and can allocate storage
        }

        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        rc_ptr = thrust::raw_pointer_cast(rc.data());
        di_ptr = thrust::raw_pointer_cast(di.data());
        hli_ptr = thrust::raw_pointer_cast(hBi.data());
        hgi_ptr = thrust::raw_pointer_cast(hAi.data());
        hlv_ptr = thrust::raw_pointer_cast(hBv.data());
        hgv_ptr = thrust::raw_pointer_cast(hAv.data());
        cudaCheckError();

        //gather (on the host)
        //columns indices
        if (typeid(t_IndPrec) == typeid(int))
        {
            mpist = MPI_Gatherv(hli_ptr, nnz, MPI_INT,  hgi_ptr, rc_ptr, di_ptr, MPI_INT,  root, mpicm);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        //values
        if      (typeid(t_MatPrec) == typeid(float))
        {
            mpist = MPI_Gatherv(hlv_ptr, nnz, MPI_FLOAT,  hgv_ptr, rc_ptr, di_ptr, MPI_FLOAT,  root, mpicm);
        }
        else if (typeid(t_MatPrec) == typeid(double))
        {
            mpist = MPI_Gatherv(hlv_ptr, nnz, MPI_DOUBLE, hgv_ptr, rc_ptr, di_ptr, MPI_DOUBLE, root, mpicm);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        if (rank == root)
        {
            if (partition_vector != NULL)
            {
                //sanity check
                if (partition_vector_size != (hAp.size() - 1))
                {
                    FatalError("partition_vector_size does not match the global vector size", AMGX_ERR_CORE);
                }

                //construct a map (based on partition vector)
                int i, j, nranks;
                MPI_Comm_size(mpicm, &nranks);
                thrust::host_vector<t_IndPrec> c(nranks, 0);
                thrust::host_vector<t_IndPrec> map(hAp.size() - 1);
                thrust::host_vector<t_IndPrec> imap(hAp.size() - 1);

                for (i = 0; i < (hAp.size() - 1); i++)
                {
                    j = partition_vector[i];
                    map[i] = nv_mtx->manager->part_offsets_h[j] + c[j];
                    imap[map[i]] = i;
                    c[j]++;
                }

                //permute rows according to map during copy (host -> host or device depending on vector type)
                hBp.resize(hAp.size());
                hBi.resize(hAi.size());
                hBv.resize(hAv.size());
                reorder_partition_host<t_IndPrec, t_MatPrec, true, true>
                (hAp.size() - 1, hAi.size(),
                 thrust::raw_pointer_cast(hAp.data()),
                 thrust::raw_pointer_cast(hAi.data()),
                 thrust::raw_pointer_cast(hAv.data()),
                 thrust::raw_pointer_cast(hBp.data()),
                 thrust::raw_pointer_cast(hBi.data()),
                 thrust::raw_pointer_cast(hBv.data()),
                 imap.size(), thrust::raw_pointer_cast(imap.data()));
                cudaCheckError();
                gA.addProps(CSR); //need to add this property, so that row_offsets, col_indices & values are resized appropriately in the next call
                gA.resize(hBp.size() - 1, hBp.size() - 1, hBi.size());
                thrust::copy(hBp.begin(), hBp.end(), gA.row_offsets.begin());
                thrust::copy(hBi.begin(), hBi.end(), gA.col_indices.begin());
                thrust::copy(hBv.begin(), hBv.end(), gA.values.begin());
                cudaCheckError();
            }
            else
            {
                //copy (host -> host or device depending on matrix type)
                gA.addProps(CSR); //need to add this property, so that row_offsets, col_indices & values are resized appropriately in the next call
                gA.resize(hAp.size() - 1, hAp.size() - 1, hAi.size());
                thrust::copy(hAp.begin(), hAp.end(), gA.row_offsets.begin());
                thrust::copy(hAi.begin(), hAi.end(), gA.col_indices.begin());
                thrust::copy(hAv.begin(), hAv.end(), gA.values.begin());
                cudaCheckError();
            }
        }
    }
    else
    {
        /* ASSUMPTION: when manager has not been allocated you are running on a single rank */
        gA.addProps(CSR); //need to add this property, so that row_offsets, col_indices & values are resized appropriately in the next call
        gA.resize(nv_mtx->row_offsets.size() - 1, nv_mtx->row_offsets.size() - 1, nv_mtx->col_indices.size());
        thrust::copy(nv_mtx->row_offsets.begin(), nv_mtx->row_offsets.end(), gA.row_offsets.begin());
        thrust::copy(nv_mtx->col_indices.begin(), nv_mtx->col_indices.end(), gA.col_indices.begin());
        thrust::copy(nv_mtx->values.begin(),     nv_mtx->values.end(),      gA.values.begin());

        cudaCheckError();
    }

    return 0;
}

template<class TConfig>
int construct_global_vector(int &root, int &rank, Matrix<TConfig> *nv_mtx, Vector<TConfig> *nv_vec,  Vector<TConfig> &gvec, int &partition_vector_size, const int *partition_vector)
{
    typedef typename TConfig::IndPrec t_IndPrec;
    typedef typename TConfig::VecPrec t_VecPrec;
    int n, nnz, offset, l;
    int mpist;
    MPI_Comm mpicm;
    //MPI call parameters
    t_IndPrec *rc_ptr, *di_ptr;
    t_VecPrec *hv_ptr, *hg_ptr;
    thrust::host_vector<t_IndPrec> rc;
    thrust::host_vector<t_IndPrec> di;
    //unreordered local vector on the host
    thrust::host_vector<t_VecPrec> hv;
    //constructed global vector on the host
    thrust::host_vector<t_VecPrec> hg;
    //WARNING: this routine currently supports vectors only with block size =1 (it can be generalized in the future, though)
    //initialize the defaults
    root = 0;
    rank = 0;
    mpist = MPI_SUCCESS;
    mpicm = MPI_COMM_WORLD;

    if (nv_mtx->manager != NULL)
    {
        // some initializations
        rank = nv_mtx->manager->global_id();

        if (nv_mtx->manager->getComms() != NULL)
        {
            mpicm = *(nv_mtx->getResources()->getMpiComm());
        }

        nv_mtx->getOffsetAndSizeForView(OWNED, &offset, &n);
        nv_mtx->getNnzForView(OWNED, &nnz);

        if (nv_mtx->manager->part_offsets_h.size() == 0)   // create part_offsets_h & part_offsets
        {
            create_part_offsets(root, rank, mpicm, nv_mtx); // (if needed for aggregation path)
        }

        l = nv_mtx->manager->part_offsets_h.size() - 1;    // number of partitions
        //some allocations/resizing
        hv.resize(nv_vec->size());                         // host copy of nv_vec

        if (rank == root)
        {
            hg.resize(nv_mtx->manager->part_offsets_h[l]); // host copy of gvec
            rc.resize(l);
            di.resize(l);
        }

        cudaCheckError();
        //--- unreorder the vector back (just like you did with the matrix, but only need to undo the interior-boundary reordering, because others do not apply) ---
        //Approach 1: just copy the vector (host or device depending on vector type -> host)
        //thrust::copy(nv_vec->begin(),nv_vec->end(),hv.begin());
        //Approach 2: unreorder and copy the vector
        thrust::copy(thrust::make_permutation_iterator(nv_vec->begin(), nv_vec->getManager()->inverse_renumbering.begin()  ),
                     thrust::make_permutation_iterator(nv_vec->begin(), nv_vec->getManager()->inverse_renumbering.begin() + n),
                     hv.begin());
        cudaCheckError();

        // --- construct global vector (rhs/sol) ---
        //compute recvcounts and displacements for MPI_Gatherv
        if (rank == root)
        {
            thrust::transform(nv_mtx->manager->part_offsets_h.begin(), nv_mtx->manager->part_offsets_h.end() - 1, nv_mtx->manager->part_offsets_h.begin() + 1, rc.begin(), subtract_op<t_IndPrec>());
            cudaCheckError();
            thrust::copy(nv_mtx->manager->part_offsets_h.begin(), nv_mtx->manager->part_offsets_h.begin() + l, di.begin());
            cudaCheckError();
        }

        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        rc_ptr = thrust::raw_pointer_cast(rc.data());
        di_ptr = thrust::raw_pointer_cast(di.data());
        hv_ptr = thrust::raw_pointer_cast(hv.data());
        hg_ptr = thrust::raw_pointer_cast(hg.data());
        cudaCheckError();

        //gather (on the host)
        if      (typeid(t_VecPrec) == typeid(float))
        {
            mpist = MPI_Gatherv(hv_ptr, n, MPI_FLOAT,  hg_ptr, rc_ptr, di_ptr, MPI_FLOAT,  root, mpicm);
        }
        else if (typeid(t_VecPrec) == typeid(double))
        {
            mpist = MPI_Gatherv(hv_ptr, n, MPI_DOUBLE, hg_ptr, rc_ptr, di_ptr, MPI_DOUBLE, root, mpicm);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        if (rank == root)
        {
            if (partition_vector != NULL)
            {
                //sanity check
                if (partition_vector_size != hg.size())
                {
                    FatalError("partition_vector_size does not match the global vector size", AMGX_ERR_CORE);
                }

                //construct a map (based on partition vector)
                int i, j, nranks;
                MPI_Comm_size(mpicm, &nranks);
                thrust::host_vector<t_IndPrec> c(nranks, 0);
                thrust::host_vector<t_IndPrec> map(hg.size());
                thrust::host_vector<t_IndPrec> imap(hg.size());

                for (i = 0; i < hg.size(); i++)
                {
                    j = partition_vector[i];
                    map[i] = nv_mtx->manager->part_offsets_h[j] + c[j];
                    imap[map[i]] = i;
                    c[j]++;
                }

                //permute according to map during copy (host -> host or device depending on vector type)
                gvec.resize(hg.size());
                thrust::copy(thrust::make_permutation_iterator(hg.begin(), imap.begin()),
                             thrust::make_permutation_iterator(hg.begin(), imap.end()),
                             gvec.begin());
                cudaCheckError();
            }
            else
            {
                //copy (host -> host or device depending on vector type)
                gvec.resize(hg.size());
                thrust::copy(hg.begin(), hg.end(), gvec.begin());
                cudaCheckError();
            }
        }
    }
    else
    {
        /* ASSUMPTION: when manager has not been allocated you are running on a single rank */
        gvec.resize(nv_vec->size());
        thrust::copy(nv_vec->begin(), nv_vec->end(), gvec.begin());
        cudaCheckError();
    }

    return 0;
}

#endif

typedef CWrapHandle<AMGX_config_handle, AMG_Configuration> ConfigW;
typedef CWrapHandle<AMGX_resources_handle, Resources> ResourceW;
typedef CWrapHandle<AMGX_distribution_handle, MatrixDistribution> MatrixDistributionW;


namespace
{

template<AMGX_Mode CASE,
         template<typename> class SolverType,
         template<typename> class MatrixType>
inline AMGX_ERROR set_solver_with(AMGX_solver_handle slv,
                                  AMGX_matrix_handle mtx,
                                  Resources *resources,
                                  AMGX_ERROR (SolverType<typename TemplateMode<CASE>::Type>::*memf)(MatrixType<typename TemplateMode<CASE>::Type> &))
{
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_solver_handle, SolverLetterT> SolverW;
    typedef MatrixType<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();

    if (wrapA.mode() != wrapSolver.mode() )
    {
        FatalError("Error: mismatch between Matrix mode and Solver Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.getResources() != solver.getResources())
    {
        FatalError("Error: matrix and solver use different resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaSetDevice(solver.getResources()->getDevice(0));
    return (solver.*memf)(A);
}

template<AMGX_Mode CASE,
         template<typename> class SolverType,
         template<typename> class MatrixType>
inline AMGX_ERROR set_solver_with_shared(AMGX_solver_handle slv,
        AMGX_matrix_handle mtx,
        Resources *resources,
        AMGX_ERROR (SolverType<typename TemplateMode<CASE>::Type>::*memf)(std::shared_ptr<MatrixType<typename TemplateMode<CASE>::Type>>))
{
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_solver_handle, SolverLetterT> SolverW;
    typedef MatrixType<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();

    if (wrapA.mode() != wrapSolver.mode() )
    {
        FatalError("Error: mismatch between Matrix mode and Solver Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.getResources() != solver.getResources())
    {
        FatalError("Error: matrix and solver use different resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaSetDevice(solver.getResources()->getDevice(0));
    return (solver.*memf)(wrapA.wrapped());
}

template<AMGX_Mode CASE,
         template<typename> class SolverType,
         template<typename> class VectorType>
inline AMGX_ERROR solve_with(AMGX_solver_handle slv,
                             AMGX_vector_handle rhs,
                             AMGX_vector_handle sol,
                             Resources *resources,
                             bool xIsZero = false)
{
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_solver_handle, SolverLetterT> SolverW;
    typedef VectorType<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();
    //AMGX_STATUS& slv_stat = wrapSolver.last_solve_status(slv);
    VectorW wrapRhs(rhs);
    VectorLetterT &b = *wrapRhs.wrapped();
    VectorW wrapSol(sol);
    VectorLetterT &x = *wrapSol.wrapped();

    if (wrapRhs.mode() != wrapSolver.mode())
    {
        FatalError("Error: mismatch between RHS mode and Solver Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (wrapRhs.mode() != wrapSol.mode())
    {
        FatalError("Error: mismatch between RHS mode and Sol Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if ((b.getResources() != solver.getResources())
            || (x.getResources() != solver.getResources()))
    {
        FatalError("Error: Inconsistency between solver and rhs/sol resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaSetDevice(solver.getResources()->getDevice(0));
    AMGX_ERROR ret = solver.solve(b, x, wrapSolver.last_solve_status(), xIsZero);
    return ret;
}

template<AMGX_Mode CASE,
         template<typename> class MatrixType,
         template<typename> class VectorType>
inline AMGX_ERROR matrix_vector_multiply(AMGX_matrix_handle mtx,
        AMGX_vector_handle x,
        AMGX_vector_handle rhs,
        Resources *resources)
{
    typedef MatrixType<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    typedef VectorType<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    //AMGX_STATUS& slv_stat = wrapSolver.last_solve_status(slv);
    VectorW wrapRhs(rhs);
    VectorLetterT &v_rhs = *wrapRhs.wrapped();
    VectorW wrapX(x);
    VectorLetterT &v_x = *wrapX.wrapped();

    if (wrapX.mode() != wrapA.mode())
    {
        FatalError("Error: mismatch between vector x mode and matrix mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (wrapX.mode() != wrapRhs.mode())
    {
        FatalError("Error: mismatch between vector y mode and vector x mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if ((A.getResources() != v_rhs.getResources())
            || (A.getResources() != v_x.getResources()))
    {
        FatalError("Error: Inconsistency between matrix and vectors resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaSetDevice(resources->getDevice(0));
    // latency hiding disable
    /*if (A.getManager() != NULL)
    {
        A.manager->exchange_halo_wait(v_x, v_x.tag);
        v_x.dirtybit = 0;
    }*/
    multiply(A, v_x, v_rhs);
    return AMGX_OK;
}

template<AMGX_Mode CASE,
         template<typename> class MatrixType,
         template<typename> class VectorType,
         template<typename> class SolverType>
inline AMGX_ERROR solver_calculate_residual( AMGX_solver_handle slv,
        AMGX_matrix_handle mtx,
        AMGX_vector_handle rhs,
        AMGX_vector_handle x,
        Resources *resources,
        AMGX_vector_handle r)
{
    typedef MatrixType<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef VectorType<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_solver_handle, SolverLetterT> SolverW;
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    VectorW wrapRhs(rhs);
    VectorLetterT &v_rhs = *wrapRhs.wrapped();
    VectorW wrapX(x);
    VectorLetterT &v_x = *wrapX.wrapped();
    VectorW wrapR(r);
    VectorLetterT &v_r = *wrapR.wrapped();

    if (wrapX.mode() != wrapA.mode())
    {
        FatalError("Error: mismatch between vector x mode and matrix mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (wrapX.mode() != wrapRhs.mode())
    {
        FatalError("Error: mismatch between vector y mode and vector x mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }
 
    if (wrapX.mode() != wrapR.mode())
    {
        FatalError("Error: mismatch between vector r mode and vector x mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if ((A.getResources() != v_rhs.getResources())
            || (A.getResources() != v_x.getResources()) 
            || (A.getResources() != v_r.getResources()))
    {
        FatalError("Error: Inconsistency between matrix and vectors resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaSetDevice(resources->getDevice(0));
    solver.getSolverObject()->compute_residual(v_rhs, v_x, v_r);
    return AMGX_OK;
}

template<AMGX_Mode CASE,
         template<typename> class MatrixType,
         template<typename> class VectorType,
         template<typename> class SolverType>
inline AMGX_ERROR solver_calculate_residual_norm( AMGX_solver_handle slv,
        AMGX_matrix_handle mtx,
        AMGX_vector_handle rhs,
        AMGX_vector_handle x,
        Resources *resources,
        void *norm_data)
{
    typedef MatrixType<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef VectorType<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_solver_handle, SolverLetterT> SolverW;
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    VectorW wrapRhs(rhs);
    VectorLetterT &v_rhs = *wrapRhs.wrapped();
    VectorW wrapX(x);
    VectorLetterT &v_x = *wrapX.wrapped();

    if (wrapX.mode() != wrapA.mode())
    {
        FatalError("Error: mismatch between vector x mode and matrix mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (wrapX.mode() != wrapRhs.mode())
    {
        FatalError("Error: mismatch between vector y mode and vector x mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if ((A.getResources() != v_rhs.getResources())
            || (A.getResources() != v_x.getResources()))
    {
        FatalError("Error: Inconsistency between matrix and vectors resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaSetDevice(resources->getDevice(0));
    solver.getSolverObject()->compute_residual_norm_external(A, v_rhs, v_x, (typename amgx::types::PODTypes<typename VectorLetterT::value_type>::type *)norm_data);
    return AMGX_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_upload_all(AMGX_matrix_handle mtx,
                                 int n,
                                 int nnz,
                                 int block_dimx,
                                 int block_dimy,
                                 const int *row_ptrs,
                                 const int *col_indices,
                                 const void *data,
                                 const void *diag_data,
                                 Resources *resources)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();

    if (!wrapA.is_valid()
            || n < 1
            || nnz < 0
            || block_dimx < 1
            || block_dimy < 1)
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
        //FatalError("Error: Failure in matrix_upload_all().\n", AMGX_ERR_BAD_PARAMETERS);
        typedef typename MatPrecisionMap<AMGX_GET_MODE_VAL(AMGX_MatPrecision, CASE)>::Type ValueType;

    A.set_initialized(0);
    cudaSetDevice(A.getResources()->getDevice(0));
    A.addProps(CSR);
    A.setColsReorderedByColor(false);
    A.delProps(COO);
    A.delProps(DIAG);

    if (diag_data)
    {
        A.addProps(DIAG);
    }

    /*If manager doesn't exist (single GPU), then upload matrix, otherwise call manager*/
    if (A.manager == NULL)
    {
        int _t = A.resize(n, n, nnz, block_dimx, block_dimy);
        cudaMemcpy(A.row_offsets.raw(), row_ptrs, sizeof(int) * (n + 1), cudaMemcpyDefault);
        cudaMemcpy(A.col_indices.raw(), col_indices, sizeof(int) * nnz, cudaMemcpyDefault);
        cudaMemcpy(A.values.raw(), data, sizeof(ValueType) * nnz * block_dimx * block_dimy, cudaMemcpyDefault);

        if (diag_data)
        {
            cudaMemcpy(A.values.raw() + A.diagOffset()*A.get_block_size(), diag_data, sizeof(ValueType) * n * block_dimx * block_dimy, cudaMemcpyDefault);
        }
        else
        {
            A.computeDiagonal();
        }

        cudaCheckError();
    }
    else
    {
        A.manager->uploadMatrix(n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, data, diag_data, A);
    }

    /* if (A.manager != NULL) A.manager->printToFile("M_clf_ua",""); */
    /* A.printToFile("A_clf_ua","",-1,-1); */
    A.set_initialized(1);
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_replace_coefficients(AMGX_matrix_handle mtx,
        int n,
        int nnz,
        const void *data,
        const void *diag_data,
        Resources *resources)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    cudaSetDevice(A.getResources()->getDevice(0));
    typedef typename MatPrecisionMap<AMGX_GET_MODE_VAL(AMGX_MatPrecision, CASE)>::Type ValueType;

    if (A.manager != NULL &&
            (A.manager->isFineLevelConsolidated() && A.manager->getFineLevelComms()->halo_coloring != LAST ||
             !A.manager->isFineLevelConsolidated() && A.manager->getComms()->halo_coloring != LAST)
       )
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);
    }
    else if (A.manager != NULL && (A.manager->isFineLevelConsolidated() || A.manager->isFineLevelGlued()))
    {
        A.manager->replaceMatrixCoefficientsWithCons(n, nnz, (const ValueType *)data, (const ValueType *)diag_data);
    }
    else if (A.manager != NULL && !A.is_matrix_singleGPU())
    {
        A.manager->replaceMatrixCoefficientsNoCons(n, nnz, (const ValueType *)data, (const ValueType *)diag_data);
    }
    else
    {
        if (n != A.get_num_rows())
        {
            std::string err = "Data passed to replace_coefficients doesn't correspond matrix object";
            amgx_output(err.c_str(), err.length());
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
        }

        if (data)
        {
            cudaMemcpy(A.values.raw(), (ValueType *)data, sizeof(ValueType) * (nnz * A.get_block_size()), cudaMemcpyDefault);
        }

        if (diag_data)
        {
            cudaMemcpy(A.values.raw() + nnz * A.get_block_size(), (ValueType *)diag_data, sizeof(ValueType) * (n * A.get_block_size()), cudaMemcpyDefault);
        }

        cudaCheckError();
    }

    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline void matrix_attach_geometry(AMGX_matrix_handle mtx,
                                   double *geox,
                                   double *geoy,
                                   double *geoz,
                                   int n,
                                   int dimension)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::MVector Vector_h;
    typedef typename Matrix<TConfig>::MVector VVector;
    MatrixLetterT *obj = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    cudaSetDevice(obj->getResources()->getDevice(0));
    Vector_h hgeo_x, hgeo_y, hgeo_z;
    VVector *geo_x = new VVector;
    VVector *geo_y = new VVector;
    hgeo_x.resize(n);
    hgeo_y.resize(n);

    if (dimension == 3)
    {
        VVector *geo_z = new VVector;
        hgeo_z.resize(n);

        for (int i = 0; i < n; i++)
        {
            hgeo_x[i] = geox[i];
            hgeo_y[i] = geoy[i];
            hgeo_z[i] = geoz[i];
        }

        *geo_z = hgeo_z;
        obj->template setParameterPtr< VVector >("geo.z", geo_z);
    }
    else if (dimension == 2)
    {
        for (int i = 0; i < n; i++)
        {
            hgeo_x[i] = geox[i];
            hgeo_y[i] = geoy[i];
        }
    }

    *geo_y = hgeo_y;
    *geo_x = hgeo_x;
    obj->template setParameter<int>("dim", dimension);
    obj->template setParameter<int>("geo_size", n);
    obj->template setParameterPtr< VVector >("geo.x", geo_x);
    obj->template setParameterPtr< VVector >("geo.y", geo_y);
}

template<AMGX_Mode CASE>
inline void matrix_attach_coloring(AMGX_matrix_handle mtx,
                                   int *row_coloring,
                                   int num_rows,
                                   int num_colors)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::IVector IVector_h;
    MatrixLetterT *obj = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    cudaSetDevice(obj->getResources()->getDevice(0));
    IVector_h *row_colors = new IVector_h;
    row_colors->resize(num_rows);

    for (int i = 0; i < num_rows; i++)
    {
        (*row_colors)[i] = row_coloring[i];
    }

    obj->template setParameter<int>("coloring_size", num_rows);
    obj->template setParameter<int>("colors_num", num_colors);
    obj->template setParameterPtr< IVector_h >("coloring", row_colors);
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_sort(AMGX_matrix_handle mtx)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    MatrixLetterT &A = *get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    cudaSetDevice(A.getResources()->getDevice(0));

    if (A.get_block_size() == 1)
    {
        A.sortByRowAndColumn();
        return AMGX_RC_OK;
    }
    else
    {
        return AMGX_RC_NOT_SUPPORTED_BLOCKSIZE;
    }
}

template<AMGX_Mode CASE>
inline AMGX_RC vector_upload(AMGX_vector_handle vec,
                             int n,
                             int block_dim,
                             const void *data)
{
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename VecPrecisionMap<AMGX_GET_MODE_VAL(AMGX_VecPrecision, CASE)>::Type ValueTypeB;
    VectorW wrapV(vec);
    VectorLetterT &v = *wrapV.wrapped();
    cudaSetDevice(v.getResources()->getDevice(0));
    v.set_block_dimx(1);
    v.set_block_dimy(block_dim);

    if (v.getManager() != NULL)
    {
        v.dirtybit = 1;
    }

    if (v.is_transformed())
    {
        v.unset_transformed();
    }

    if (v.getManager() != NULL && !v.is_transformed())
    {
        v.getManager()->transformAndUploadVector(v, data, n, block_dim);
    }
    else
    {
        v.resize(n * block_dim);
        cudaMemcpy(v.raw(), data, sizeof(ValueTypeB) * n * block_dim, cudaMemcpyDefault);
        cudaCheckError();
    }

    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC vector_set_zero(AMGX_vector_handle vec,
                               int n,
                               int block_dim,
                               Resources *resources)
{
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename VecPrecisionMap<AMGX_GET_MODE_VAL(AMGX_VecPrecision, CASE)>::Type ValueTypeB;
    VectorW wrapV(vec);
    VectorLetterT &v = *wrapV.wrapped();

    if (!wrapV.is_valid()
            || n < 0
            || block_dim < 1)
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
        cudaSetDevice(v.getResources()->getDevice(0));

    v.resize(n * block_dim);
    v.set_block_dimy(block_dim);
    thrust::fill(v.begin(), v.end(), types::util<ValueTypeB>::get_zero());
    cudaCheckError();
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC vector_set_random(AMGX_vector_handle vec, int n, Resources *resources)
{
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename VecPrecisionMap<AMGX_GET_MODE_VAL(AMGX_VecPrecision, CASE)>::Type ValueTypeB;
    VectorW wrapV(vec);
    VectorLetterT &v = *wrapV.wrapped();

    if (!wrapV.is_valid()
            || n < 0)
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
        cudaSetDevice(v.getResources()->getDevice(0));

    Vector<typename VectorLetterT::TConfig_h> t_vec(n);

    for (int i = 0; i < n; ++i)
    {
        t_vec[i] = types::get_rand<ValueTypeB>();
    }

    v = t_vec;
    cudaCheckError();
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC vector_download_impl(const AMGX_vector_handle vec,
                                    void *data)
{
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename VecPrecisionMap<AMGX_GET_MODE_VAL(AMGX_VecPrecision, CASE)>::Type ValueTypeB;
    VectorW wrapV(vec);
    VectorLetterT &v = *wrapV.wrapped();
    /*if (!wrapV.is_valid()
        || n < 0
        || block_dim < 1)
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)*/
    cudaSetDevice(v.getResources()->getDevice(0));

    if (v.getManager() != NULL)
    {
        int n, nnz;
        int block_dimy = v.get_block_dimy();
        v.getManager()->getView(OWNED, n, nnz);

        if (v.is_transformed() || v.getManager()->isFineLevelGlued())
        {
            v.getManager()->revertAndDownloadVector(v, data, n, block_dimy);
        }
        else
        {
            cudaMemcpy((ValueTypeB *)data, v.raw(), n * block_dimy * sizeof(ValueTypeB), cudaMemcpyDefault);
            cudaCheckError();
        }
    }
    else
    {
        cudaMemcpy((ValueTypeB *)data, v.raw(), v.size() * sizeof(ValueTypeB), cudaMemcpyDefault);
        cudaCheckError();
    }

    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC vector_get_size(AMGX_vector_handle vec,
                               int *n,
                               int *block_dim)
{
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename VecPrecisionMap<AMGX_GET_MODE_VAL(AMGX_VecPrecision, CASE)>::Type ValueTypeB;
    VectorW wrapV(vec);
    VectorLetterT &v = *wrapV.wrapped();

    //if (!wrapV.is_valid())
    //  AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)

    if (v.getManager() != NULL && (v.getManager()->isFineLevelConsolidated() || v.getManager()->isFineLevelGlued() ) )
    {
        *n = v.get_unconsolidated_size() / v.get_block_dimy();
    }
    else
    {
        *n = v.size() / v.get_block_dimy();
    }

    *block_dim = v.get_block_dimy();
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC read_system(AMGX_matrix_handle mtx,
                           AMGX_vector_handle rhs,
                           AMGX_vector_handle sol,
                           const char *filename,
                           unsigned int props,
                           int block_convert,
                           AMG_Config &amgx_cfg,
                           AMGX_ERROR &read_error)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef  MatrixIO<TConfig> MatrixIOLetterT;
    typedef Vector<TConfig> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef Matrix<TConfig_h> MatrixLetterT0;
    typedef MatrixIO<TConfig_h> MatrixIOLetterT0;
    typedef Vector<TConfig_h> VectorLetterT0;
    MatrixLetterT *mtx_ptr = NULL;
    VectorLetterT *rhs_ptr = NULL;
    VectorLetterT *sol_ptr = NULL;

    //if ((mtx == NULL) || (rhs == NULL) || (sol == NULL))

    if (mtx != NULL)
    {
        mtx_ptr = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    }
    else
    {
        return AMGX_RC_BAD_PARAMETERS;
    }

    if (rhs != NULL)
    {
        rhs_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(rhs);
    }

    if (sol != NULL)
    {
        sol_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(sol);
    }

    //typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    //typedef typename Vector<ivec_value_type_h> IVector_h;

    switch (AMGX_GET_MODE_VAL(AMGX_MemorySpace, CASE))
    {
        case (AMGX_device) :
            {
                MatrixLetterT0 Ah;
                MatrixLetterT0 Ahc;
                Ah.setResources(mtx_ptr->getResources());
                Ahc.setResources(mtx_ptr->getResources());
                VectorLetterT0 bh;
                VectorLetterT0 xh;
                read_error = MatrixIOLetterT0::readSystem(filename, Ah, bh, xh, amgx_cfg, props);

                if (mtx != NULL)
                {
                    if (block_convert != 0)
                    {
                        Ahc.convert(Ah, Ah.getProps(), block_convert, block_convert);
                        Ah.set_initialized(0);
                        Ah = Ahc;
                        Ah.set_initialized(1);

                        if (rhs != NULL)
                        {
                            bh.set_block_dimy(block_convert);
                        }

                        if (sol != NULL)
                        {
                            xh.set_block_dimy(block_convert);
                        }
                    }

                    mtx_ptr->set_initialized(0);
                    *mtx_ptr = Ah;
                    mtx_ptr->set_initialized(1);
                }

                if (rhs != NULL)
                {
                    *rhs_ptr = bh;
                }

                if (sol != NULL)
                {
                    *sol_ptr = xh;
                }
            }
            break;

        case (AMGX_host) :
            {
                std::shared_ptr<MatrixLetterT> Ah;
                std::shared_ptr<VectorLetterT> bh;
                std::shared_ptr<VectorLetterT> xh;

                if (mtx == NULL)
                {
                    Ah.reset(new MatrixLetterT());
                }
                else
                {
                    MatrixW wMtx(mtx);
                    Ah = wMtx.wrapped();
                }

                if (rhs == NULL)
                {
                    bh.reset(new VectorLetterT());
                }
                else
                {
                    VectorW wRhs(rhs);
                    bh = wRhs.wrapped();
                }

                if (sol == NULL)
                {
                    xh.reset(new VectorLetterT());
                }
                else
                {
                    VectorW wSol(sol);
                    xh = wSol.wrapped();
                }

                read_error = MatrixIOLetterT::readSystem((char *)filename, *Ah, *bh, *xh, amgx_cfg, props);

                if (block_convert != 0)
                {
                    if (mtx != NULL)
                    {
                        MatrixLetterT0 Ahc;
                        Ahc.convert(*Ah, Ah->getProps(), block_convert, block_convert);
                        Ah->set_initialized(0);
                        *Ah = Ahc;
                        Ah->set_initialized(1);
                    }

                    if (rhs != NULL)
                    {
                        bh->set_block_dimy(block_convert);
                    }

                    if (sol != NULL)
                    {
                        xh->set_block_dimy(block_convert);
                    }
                }
            }
            break;
    }

    return AMGX_RC_OK;
}

#ifdef AMGX_WITH_MPI
template<AMGX_Mode CASE>
inline AMGX_RC mpi_write_system_distributed(const AMGX_matrix_handle mtx,
        const AMGX_vector_handle rhs,
        const AMGX_vector_handle sol,
        const char *filename,
        int allocated_halo_depth,
        int num_partitions,
        const int *partition_sizes,
        int partition_vector_size,
        const int *partition_vector,
        AMGX_ERROR &rc)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef Vector<TConfig> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;

    if (mtx == NULL && rhs == NULL && sol == NULL)
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
    }

    MatrixLetterT *mtx_ptr = NULL;
    VectorLetterT *rhs_ptr = NULL;
    VectorLetterT *sol_ptr = NULL;

    if (mtx != NULL)
    {
        mtx_ptr = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    }

    if (rhs != NULL)
    {
        rhs_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(rhs);
    }

    if (sol != NULL)
    {
        sol_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(sol);
    }

    /*global objects*/
    MatrixLetterT gA;
    VectorLetterT grhs;
    VectorLetterT gsol;
    int root = 0, rank = 0;

    if (mtx != NULL)
    {
        gA.setResources(mtx_ptr->getResources());
        construct_global_matrix<TConfig>(root, rank, mtx_ptr, gA, partition_vector_size, partition_vector);
    }

    if (rhs != NULL)
    {
        grhs.setResources(rhs_ptr->getResources());
        construct_global_vector<TConfig>(root, rank, mtx_ptr, rhs_ptr, grhs, partition_vector_size, partition_vector);
    }

    if (sol != NULL)
    {
        gsol.setResources(sol_ptr->getResources());
        construct_global_vector<TConfig>(root, rank, mtx_ptr, sol_ptr, gsol, partition_vector_size, partition_vector);
    }

    if (rank == root)
    {
        if (mtx_ptr)
        {
            cudaSetDevice(mtx_ptr->getResources()->getDevice(0));
        }
        else if (rhs_ptr)
        {
            cudaSetDevice(rhs_ptr->getResources()->getDevice(0));
        }
        else
        {
            cudaSetDevice(sol_ptr->getResources()->getDevice(0));
        }

        rc = MatrixIO<TConfig>::writeSystem(filename, &gA, &grhs, &gsol);
    }

    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC read_system_distributed(AMGX_matrix_handle mtx,
                                       AMGX_vector_handle rhs,
                                       AMGX_vector_handle sol,
                                       const char *filename,
                                       int allocated_halo_depth,
                                       int num_partitions,
                                       const int *partition_sizes,
                                       int partition_vector_size,
                                       const int *partition_vector,
                                       std::stringstream &msg,
                                       int &num_ranks,
                                       Resources *resources,
                                       int part,
                                       unsigned int props,
                                       AMGX_ERROR &read_error)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef Vector<TConfig> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    IVector_h partitionVec;
    IVector_h partSize;
    MatrixLetterT *mtx_ptr = NULL;
    VectorLetterT *rhs_ptr = NULL;
    VectorLetterT *sol_ptr = NULL;

    if (mtx != NULL)
    {
        mtx_ptr = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    }

    if (rhs != NULL)
    {
        rhs_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(rhs);
    }

    if (sol != NULL)
    {
        sol_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(sol);
    }

    MPI_Comm *mpi_comm;

    if (mtx != NULL)
    {
        mpi_comm = mtx_ptr->getResources()->getMpiComm();
    }
    else if (rhs != NULL)
    {
        mpi_comm = rhs_ptr->getResources()->getMpiComm();
    }
    else if (sol != NULL)
    {
        mpi_comm = sol_ptr->getResources()->getMpiComm();
    }

    MPI_Comm_size(*mpi_comm, &num_ranks);
    MPI_Comm_rank(*mpi_comm, &part);

    if (partition_vector != NULL)
    {
        if (partition_sizes != NULL && num_partitions != num_ranks)
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
            partitionVec.resize(partition_vector_size);

        thrust::copy(partition_vector, partition_vector + partition_vector_size, partitionVec.begin());
        cudaCheckError();

        if (num_partitions == 0) { num_partitions = num_ranks; }

        if (num_partitions % num_ranks != 0)
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
            partSize.resize(num_ranks);

        if (partition_sizes != NULL)
        {
            thrust::copy(partition_sizes, partition_sizes + num_partitions, partSize.begin());
            cudaCheckError();
        }
        else
        {
            int partsPerRank = num_partitions / num_ranks;
            thrust::fill(partSize.begin(), partSize.end(), 0);
            cudaCheckError();

            for (int i = 0; i < partitionVec.size(); i++)
            {
                int p = partitionVec[i] / partsPerRank;
                partitionVec[i] = p;
                partSize[p]++;
            }
        }

        msg << "Read consolidated partition sizes: ";

        for (int i = 0; i < num_ranks; i++)
        {
            msg << partSize[i] << "  ";
        }

        msg << "n";
    }
    else { num_partitions = num_ranks; }

    switch (AMGX_GET_MODE_VAL(AMGX_MemorySpace, CASE))
    {
        case (AMGX_device) :
            {
                if (rhs != NULL && sol != NULL)
                {
                    read_error = DistributedRead<TConfig>::distributedRead((char *)filename, *mtx_ptr, *rhs_ptr, *sol_ptr, allocated_halo_depth, part, num_ranks, partSize, partitionVec, props);
                }
                else if (rhs != NULL)
                {
                    read_error = DistributedRead<TConfig>::distributedRead((char *)filename, *mtx_ptr, *rhs_ptr, allocated_halo_depth, part, num_ranks, partSize, partitionVec, props);
                }
                else
                {
                    read_error = DistributedRead<TConfig>::distributedRead((char *)filename, *mtx_ptr, *sol_ptr, allocated_halo_depth, part, num_ranks, partSize, partitionVec, props);
                }
            }
            break;

        case (AMGX_host) :
            {
                //local effect only, no need to be allocated on the pool
                //
                std::shared_ptr<MatrixLetterT> Ah;
                std::shared_ptr<VectorLetterT> bh;
                std::shared_ptr<VectorLetterT> xh;

                if (mtx == NULL)
                {
                    Ah.reset(new MatrixLetterT());
                }
                else
                {
                    MatrixW wMtx(mtx);
                    Ah = wMtx.wrapped();
                }

                if (rhs == NULL)
                {
                    bh.reset(new VectorLetterT());
                }
                else
                {
                    VectorW wRhs(rhs);
                    bh = wRhs.wrapped();
                }

                if (sol == NULL)
                {
                    xh.reset(new VectorLetterT());
                }
                else
                {
                    VectorW wSol(sol);
                    xh = wSol.wrapped();
                }

                read_error = DistributedRead<TConfig>::distributedRead((char *)filename, *Ah, *bh, *xh, allocated_halo_depth, part, num_partitions, partSize, partitionVec, props);
            }
            break;
    }

    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC generate_distributed_poisson_7pt(AMGX_matrix_handle mtx,
        AMGX_vector_handle rhs_,
        AMGX_vector_handle sol_,
        int allocated_halo_depth,
        int num_import_rings,
        int nx,
        int ny,
        int nz,
        int px,
        int py,
        int pz)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef Vector<TConfig> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    typedef typename Vector<TConfig>::value_type ValueTypeB;
    MatrixW wrapA(mtx);
    MatrixLetterT &A_part = *wrapA.wrapped();
    VectorW wrapRhs(rhs_);
    VectorLetterT &rhs = *wrapRhs.wrapped();
    VectorW wrapSol(sol_);
    VectorLetterT &sol = *wrapSol.wrapped();
    cudaSetDevice(A_part.getResources()->getDevice(0));
    MPI_Comm *mpi_comm = A_part.getResources()->getMpiComm();
    int num_ranks;
    MPI_Comm_size(*mpi_comm, &num_ranks);

    if ((px * py * pz != num_ranks) || (px * py * pz == 0))
    {
        amgx_printf(" Invalid number of processors or processor topologyn ");
        return AMGX_RC_BAD_PARAMETERS;
    }

    /* Create distributed manager */
    if (A_part.manager != NULL)
    {
        delete A_part.manager;
        A_part.manager = NULL;
    }

    A_part.manager = new DistributedManager<TConfig>(A_part);
    A_part.setManagerExternal();
    /* Generate 7pt Poisson matrix  */
    A_part.manager->generatePoisson7pt(nx, ny, nz, px, py, pz);
    /* Create B2L_maps for comm */
    A_part.manager->renumberMatrixOneRing();

    /* Exchange 1 ring halo rows (for d2 interp) */
    if (num_import_rings == 2)
    {
        A_part.manager->createOneRingHaloRows();
    }

    A_part.manager->getComms()->set_neighbors(A_part.manager->num_neighbors());
    A_part.setView(OWNED);
    A_part.set_initialized(1);
    /* Create rhs and solution */
    rhs.resize(A_part.get_num_rows());
    thrust::fill(rhs.begin(), rhs.end(), types::util<ValueTypeB>::get_one());
    sol.resize(A_part.get_num_rows());
    thrust::fill(sol.begin(), sol.end(), types::util<ValueTypeB>::get_one());
    cudaCheckError();
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_upload_distributed(AMGX_matrix_handle mtx,
                                        int n_global,
                                        int n,
                                        int nnz,
                                        int block_dimx,
                                        int block_dimy,
                                        const int *row_ptrs,
                                        const void *col_indices_global,
                                        const void *data,
                                        const void *diag_data,
                                        AMGX_distribution_handle dist)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef typename MatPrecisionMap<AMGX_GET_MODE_VAL(AMGX_MatPrecision, CASE)>::Type ValueType;
    MatrixDistributionW wrapDist(dist);
    MatrixDistribution &mdist = *wrapDist.wrapped();
    MatrixW wrapA(mtx);
    MatrixLetterT &A_part = *wrapA.wrapped();
    cudaSetDevice(A_part.getResources()->getDevice(0));
    MPI_Comm *mpi_comm = A_part.getResources()->getMpiComm();
    int num_ranks;
    MPI_Comm_size(*mpi_comm, &num_ranks);

    /* Create distributed manager */
    if (A_part.manager != NULL)
    {
        delete A_part.manager;
        A_part.manager = NULL;
    }

    A_part.manager = new DistributedManager<TConfig>(A_part);
    A_part.setManagerExternal();
    /* Load distributed matrix 
        Choose correct overload based on column index type
     */
    if (mdist.get32BitColIndices()) 
    {
        A_part.manager->loadDistributedMatrix(n, nnz, block_dimx, block_dimy, row_ptrs, (int *)col_indices_global,
            (ValueType *)data, num_ranks, n_global, diag_data, mdist);
    }
    else
    {
        A_part.manager->loadDistributedMatrix(n, nnz, block_dimx, block_dimy, row_ptrs, (int64_t *)col_indices_global,
            (ValueType *)data, num_ranks, n_global, diag_data, mdist);
    }
    /* Create B2L_maps for comm */
    A_part.manager->renumberMatrixOneRing();

    /* Exchange 1 ring halo rows (for d2 interp) */
    if (mdist.getNumImportRings() == 2)
    {
        A_part.manager->createOneRingHaloRows();
    }

    A_part.manager->getComms()->set_neighbors(A_part.manager->num_neighbors());
    A_part.setView(OWNED);
    /* if (A_part.manager != NULL) A_part.manager->printToFile("M_clf_gua",""); */
    /* A_part.printToFile("A_clf_gua","",-1,-1); */
    A_part.set_initialized(1);
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_upload_all_global(AMGX_matrix_handle mtx,
                                        int n_global,
                                        int n,
                                        int nnz,
                                        int block_dimx,
                                        int block_dimy,
                                        const int *row_ptrs,
                                        const void *col_indices_global,
                                        const void *data,
                                        const void *diag_data,
                                        int allocated_halo_depth, // TODO: unused parameter
                                        int num_import_rings,
                                        const int *partition_vector)
{
    AMGX_distribution_handle dist;
    AMGX_distribution_create(&dist, NULL);
    MatrixDistributionW wrapDist(dist);
    MatrixDistribution &mdist = *wrapDist.wrapped();
    mdist.setPartitionVec(partition_vector);
    mdist.setNumImportRings(num_import_rings);
    auto rc = matrix_upload_distributed<CASE>(mtx, n_global, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices_global,
        data, diag_data, dist);
    AMGX_distribution_destroy(dist);
    return rc;
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_upload_all_global_32(AMGX_matrix_handle mtx, 
                                           int n_global,
                                           int n,
                                           int nnz,
                                           int block_dimx,
                                           int block_dimy,
                                           const int *row_ptrs,
                                           const void *col_indices_global,
                                           const void *data,
                                           const void *diag_data,
                                           int allocated_halo_depth, // TODO: unused parameter
                                           int num_import_rings,
                                           const int *partition_vector)
{
    AMGX_distribution_handle dist;
    AMGX_distribution_create(&dist, NULL);
    MatrixDistributionW wrapDist(dist);
    MatrixDistribution &mdist = *wrapDist.wrapped();
    mdist.setPartitionVec(partition_vector);
    mdist.setNumImportRings(num_import_rings);
    mdist.set32BitColIndices(true);
    auto rc = matrix_upload_distributed<CASE>(mtx, n_global, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices_global,
        data, diag_data, dist);
    AMGX_distribution_destroy(dist);
    return rc;    
}
#endif

template<AMGX_Mode CASE>
inline AMGX_RC matrix_comm_from_maps(AMGX_matrix_handle mtx, int allocated_halo_depth, int num_import_rings, int max_num_neighbors, const int *neighbors, const int *send_ptrs, int const *send_maps, const int *recv_ptrs, int const  *recv_maps)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A_part = *wrapA.wrapped();
    cudaSetDevice(A_part.getResources()->getDevice(0));

    if (allocated_halo_depth > 1)
    {
        amgx_printf("Allocated_halo_depth > 1 currently not supported");
        return AMGX_RC_BAD_PARAMETERS;
    }

    if (num_import_rings > 1)
    {
        amgx_printf("num_import_rings > 1 currently not supported");
        return AMGX_RC_BAD_PARAMETERS;
    }

    if (allocated_halo_depth != num_import_rings)
    {
        amgx_printf("num_import_rings != allocated_halo_depth currently not supported");
        return AMGX_RC_BAD_PARAMETERS;
    }

    if (A_part.manager != NULL)
    {
        delete A_part.manager;
        A_part.manager = NULL;
    }

    if (max_num_neighbors > 0)
    {
        A_part.manager = new DistributedManager<TConfig>(A_part, allocated_halo_depth, num_import_rings, max_num_neighbors, neighbors);
        A_part.manager->cacheMaps(send_maps, send_ptrs, recv_maps, recv_ptrs);
        A_part.setManagerExternal();
        A_part.manager->createComms(A_part.getResources());
    }

    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline AMGX_RC write_system(const AMGX_matrix_handle mtx,
                            const AMGX_vector_handle rhs,
                            const AMGX_vector_handle sol,
                            const char *filename,
                            AMGX_ERROR &rc)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef Vector<TConfig> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;

    if (mtx == NULL && rhs == NULL && sol == NULL)
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
    }

    MatrixLetterT *mtx_ptr = NULL;
    VectorLetterT *rhs_ptr = NULL;
    VectorLetterT *sol_ptr = NULL;

    if (mtx != NULL)
    {
        mtx_ptr = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx);
    }

    if (rhs != NULL)
    {
        rhs_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(rhs);
    }

    if (sol != NULL)
    {
        sol_ptr = get_mode_object_from<CASE, Vector, AMGX_vector_handle>(sol);
    }

    if (mtx_ptr)
    {
        cudaSetDevice(mtx_ptr->getResources()->getDevice(0));
    }
    else if (rhs_ptr)
    {
        cudaSetDevice(rhs_ptr->getResources()->getDevice(0));
    }
    else
    {
        cudaSetDevice(sol_ptr->getResources()->getDevice(0));
    }

    rc = MatrixIO<TConfig>::writeSystem(filename, mtx_ptr, rhs_ptr, sol_ptr);
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline void solver_get_iterations_number(AMGX_solver_handle slv, int *n)
{
    auto *solver = get_mode_object_from<CASE, AMG_Solver, AMGX_solver_handle>(slv);
    cudaSetDevice(solver->getResources()->getDevice(0));
    *n = solver->get_num_iters();
}

template<AMGX_Mode CASE>
inline AMGX_RC solver_get_iteration_residual(AMGX_solver_handle slv,
        int it,
        int idx,
        double *res)
{
    auto *solver = get_mode_object_from<CASE, AMG_Solver, AMGX_solver_handle>(slv);
    cudaSetDevice(solver->getResources()->getDevice(0));

    if (idx < 0 || idx >= solver->get_residual(it).size())
    {
        amgx_printf("Incorrect block index");
        return AMGX_RC_BAD_PARAMETERS;
    }

    *res = (double)solver->get_residual(it)[idx];
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE>
inline void solver_get_status(AMGX_solver_handle slv, AMGX_SOLVE_STATUS *st)
{
    typedef AMG_Solver<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_solver_handle, SolverLetterT> SolverW;
    SolverW wrapSolver(slv);

    switch (wrapSolver.last_solve_status())
    {
        case AMGX_ST_CONVERGED:
            *st = AMGX_SOLVE_SUCCESS;
            break;

        case AMGX_ST_DIVERGED:
            *st = AMGX_SOLVE_DIVERGED;
            break;

        case AMGX_ST_NOT_CONVERGED:
            *st = AMGX_SOLVE_NOT_CONVERGED;
            break;

        default:
            *st = AMGX_SOLVE_FAILED;
    }
}

template<AMGX_Mode CASE>
inline void matrix_download_all(const AMGX_matrix_handle mtx,
                                int *row_ptrs,
                                int *col_indices,
                                void *data,
                                void **diag_data)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    typedef typename MatPrecisionMap<AMGX_GET_MODE_VAL(AMGX_MatPrecision, CASE)>::Type ValueType;
    cudaSetDevice(A.getResources()->getDevice(0));
    int n, nnz, block_size;
    n = A.get_num_rows();
    block_size = A.get_block_size();
    nnz = A.get_num_nz();

    if (A.hasProps(DIAG))
    {
        int sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, CASE) == AMGX_matDouble)) ? sizeof(double) : sizeof(float);
        *diag_data = get_c_arr_mem_manager().allocate(n * block_size * sizeof_m_val);
        cudaMemcpy((ValueType *)(*diag_data), A.values.raw() + nnz * block_size, n * block_size * sizeof(ValueType), cudaMemcpyDefault);
    }
    else
    {
        *diag_data = NULL;
    }

    cudaMemcpy(row_ptrs, A.row_offsets.raw(), A.row_offsets.size()*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(col_indices, A.col_indices.raw(), A.col_indices.size()*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(data, A.values.raw(), nnz * block_size * sizeof(ValueType), cudaMemcpyDefault);
    cudaCheckError();
}

template<AMGX_Mode CASE>
inline void vector_bind(AMGX_vector_handle vec, const AMGX_matrix_handle mtx)
{
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    VectorW wrapV(vec);
    VectorLetterT &x = *wrapV.wrapped();
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    cudaSetDevice(A.getResources()->getDevice(0));

    if (A.getResources() != x.getResources())
    {
        FatalError("Matrix and vector don't use same resources, exiting", AMGX_ERR_CONFIGURATION);
    }

    if (A.manager != NULL)
    {
        x.setManager(*(A.manager));
    }

    cudaCheckError();
}

template<AMGX_Mode CASE>
inline void read_system_maps_one_ring_impl( const AMGX_matrix_handle A_part,
        int *num_neighbors,
        int **neighbors,
        int **btl_sizes,
        int ***btl_maps,
        int **lth_sizes,
        int ***lth_maps,
        int64_t **local_to_global_map)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef typename TConfig::template setMemSpace<AMGX_device>::Type SType;
    typedef Matrix<SType> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(A_part);
    MatrixLetterT &A = *wrapA.wrapped();
    cudaSetDevice(A.getResources()->getDevice(0));
    A.manager->malloc_export_maps(btl_maps, btl_sizes, lth_maps, lth_sizes);
    *num_neighbors = A.manager->num_neighbors();
    *neighbors = (int *)get_c_arr_mem_manager().allocate((*num_neighbors) * sizeof(int));
    A.manager->export_neighbors(*neighbors);

    /* check if we're in read_system_global */
    if (local_to_global_map != NULL) //??? maybe meant == NULL ?
    {
        /* now setup local to global maps */
        int map_size = A.manager->local_to_global_map.size();
        *local_to_global_map = (int64_t *)get_c_arr_mem_manager().allocate(map_size * sizeof(int64_t));

        for (int i = 0; i < map_size; i++)
        {
            (*local_to_global_map)[i] = A.manager->local_to_global_map[i];
        }
    }
}

template<AMGX_Mode CASE>
inline AMGX_RC matrix_comm_from_maps_one_ring(AMGX_matrix_handle mtx,
        int allocated_halo_depth,
        int max_num_neighbors,
        const int *neighbors,
        const int *send_sizes,
        int const **send_maps,
        const int *recv_sizes,
        int const **recv_maps,
        Resources *resources)
{
    typedef typename TemplateMode<CASE>::Type TConfig;
    typedef Matrix<TConfig> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A_part = *wrapA.wrapped();
    cudaSetDevice(A_part.getResources()->getDevice(0));

    if (allocated_halo_depth > 1)
    {
        amgx_printf("Allocated_halo_depth > 1 currently not supported");
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
    }

    if (A_part.manager != NULL)
    {
        delete A_part.manager;
        A_part.manager = NULL;
    }

    if (max_num_neighbors > 0)
    {
        // Create a new manager, and save neighbor list
        A_part.manager = new DistributedManager<TConfig>(A_part, 1, 1, max_num_neighbors, neighbors);
        // save boundary and halo lists to manager
        A_part.manager->cacheMapsOneRing(send_maps, send_sizes, recv_maps, recv_sizes);
        A_part.setManagerExternal();
        // Create comms module, "communicator" config string either MPI or MPI_DIRECT
        A_part.manager->createComms(A_part.getResources());
    }

    return AMGX_RC_OK;
}


}//end unnamed namespace

AMGX_RC write_system_preamble(const AMGX_matrix_handle mtx,
                              const AMGX_vector_handle rhs,
                              const AMGX_vector_handle sol,
                              Resources *&resources,
                              AMGX_Mode &mode)
{
    if (mtx == NULL && rhs == NULL && sol == NULL)
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
    }

    mode = AMGX_unset;
    AMGX_Mode m_mode = AMGX_unset;
    AMGX_Mode r_mode = AMGX_unset;
    AMGX_Mode s_mode = AMGX_unset;
    resources = NULL;

    if (mtx != NULL)
    {
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        //if (!c_mtx || !c_mtx->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        mode = m_mode = get_mode_from<AMGX_matrix_handle>(mtx);
    }

    if (rhs != NULL)
    {
        if (resources == NULL)
            AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(rhs, &resources)), NULL)
            //no need to check validity here,
            //it's done elsewhere via:
            //get_mode_object_from<...>(...)->
            //CWrapHandle(Envelope) cstrctr ->
            //is_valid()
            //
            //if (!wrapRhs.is_valid()) AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);
            mode = r_mode = get_mode_from<AMGX_vector_handle>(rhs);
    }

    if (sol != NULL)
    {
        if (resources == NULL)
            AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(sol, &resources)), NULL)
            //no need to check validity here,
            //it's done elsewhere via:
            //get_mode_object_from<...>(...)->
            //CWrapHandle(Envelope) cstrctr ->
            //is_valid()
            //
            //if (!wrapSol.is_valid()) AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);
            mode = s_mode = get_mode_from<AMGX_vector_handle>(sol);
    }

    if (mtx != NULL)
    {
        if (m_mode != mode) { AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources); }
    }

    if (rhs != NULL)
    {
        if (r_mode != mode) { AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources); }
    }

    if (sol != NULL)
    {
        if (s_mode != mode) { AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources); }
    }

    return AMGX_RC_OK;
}

AMGX_RC read_system_preamble(const AMGX_matrix_handle mtx,
                             const AMGX_vector_handle rhs,
                             const AMGX_vector_handle sol,
                             Resources *&resources,
                             AMGX_Mode &mode,
                             unsigned int &props,
                             bool try_any = false)
{
    if (mtx == NULL && rhs == NULL && sol == NULL)
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
    }

    mode = AMGX_unset;
    resources = NULL;
    props = io_config::NONE;

    if (mtx != NULL)
    {
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        io_config::addProps(io_config::MTX, props);
        mode = get_mode_from<AMGX_matrix_handle>(mtx);
    }
    else if (!try_any)
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL) // there are no valid resources without Matrix object
        if (rhs != NULL)
        {
            io_config::addProps(io_config::RHS, props);

            if (mode == AMGX_unset)
            {
                mode = get_mode_from<AMGX_vector_handle>(rhs);
            }

            if (try_any)
            {
                if (resources == NULL)
                    AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(rhs, &resources)), NULL)
                }

            //no need to check validity here,
            //it's done elsewhere via:
            //get_mode_object_from<...>(...)->
            //CWrapHandle(Envelope) cstrctr ->
            //is_valid()
            //
            //if (!wrapRhs.is_valid()) AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);
        }

    if (sol != NULL)
    {
        io_config::addProps(io_config::SOLN, props);

        if (mode == AMGX_unset)
        {
            mode = get_mode_from<AMGX_vector_handle>(sol);
        }

        if (try_any)
        {
            if (resources == NULL)
                AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(sol, &resources)), NULL)
            }

        //no need to check validity here,
        //it's done elsewhere via:
        //get_mode_object_from<...>(...)->
        //CWrapHandle(Envelope) cstrctr ->
        //is_valid()
        //
        //if (!wrapSol.is_valid()) AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);
    }

    if (props == io_config::NONE)
    {
        return AMGX_RC_BAD_PARAMETERS;
    }

    return AMGX_RC_OK;
}


} // end namespace

using namespace amgx;
extern "C" {

    void AMGX_API AMGX_abort(AMGX_resources_handle rsc, int err)
    {
        Resources *resources = NULL;

        if (rsc != NULL)
        {
            AMGX_ERROR rc = AMGX_OK;

            try
            {
                ///amgx::CWrapper<AMGX_resources_handle> *c_resources= (amgx::CWrapper<AMGX_resources_handle>*)rsc;
                ResourceW c_r(rsc);

                ///if (!c_resources)
                if (!c_r.wrapped())
                {
                    fprintf(stderr, "AMGX_abort warning: provided wrong resources, using defaults");
                    amgx_error_exit(NULL, err);
                }

                ///resources = (Resources*)(c_resources->hdl);
                resources = c_r.wrapped().get();
            }

            AMGX_CATCHES(rc)
            if (AMGX_OK != rc)
            {
                fprintf(stderr, "AMGX_abort warning: catched %d\n",rc);
            }
        }

        amgx_error_exit(resources, err);
    }

    AMGX_RC AMGX_API AMGX_initialize()
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_initialize " );
        AMGX_CHECK_API_ERROR(amgx::initialize(), NULL);
        return AMGX_RC_OK;
        //return getCAPIerror(amgx::initialize());
    }

    AMGX_RC AMGX_API AMGX_initialize_plugins()
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_initialize_plugins " );
        return getCAPIerror_x(amgx::initializePlugins());
    }

    AMGX_RC AMGX_API AMGX_finalize()
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_finalize " );
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            amgx::finalize();
        }

        AMGX_CATCHES(rc)
        //AMGX_CHECK_API_ERROR(rc, NULL);
        //return AMGX_RC_OK;
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_finalize_plugins()
    {
        AMGX_CPU_PROFILER( "AMGX_finalize_plugins " );
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            amgx::finalizePlugins();
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, NULL);
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_config_create(AMGX_config_handle *cfg_h, const char *options)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_config_create " );
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR err;

        try
        {
            auto *cfg0 = create_managed_object<AMG_Configuration, AMGX_config_handle>(cfg_h);
            err = cfg0->wrapped()->parseParameterString(options);
        }

        AMGX_CATCHES(rc)

        if (rc != AMGX_OK)
            AMGX_CHECK_API_ERROR(rc, NULL)
            //return getCAPIerror(rc);
            else
                AMGX_CHECK_API_ERROR(err, NULL)
                //return getCAPIerror(err);
                return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_config_create_from_file(AMGX_config_handle *cfg_h, const char *param_file)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_config_create_from_file " );
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR err;
        ConfigW *cfg = nullptr;

        try
        {
            ///cfg = get_mem_manager<ConfigW>().allocate<ConfigW>(AMG_Configuration()).get();
            cfg = create_managed_object<AMG_Configuration, AMGX_config_handle>(cfg_h);
            ///err = (*(AMG_Configuration *)cfg->hdl()).parseFile(param_file);
            err = cfg->wrapped()->parseFile(param_file);
            ///*cfg_h = (AMGX_config_handle)cfg;
        }

        AMGX_CATCHES(rc)

        if (rc != AMGX_OK)
            AMGX_CHECK_API_ERROR(rc, NULL)
            //return getCAPIerror(rc);
            else
            {
                AMGX_CHECK_API_ERROR(err, NULL);
            }

        //return getCAPIerror(err);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_config_create_from_file_and_string(AMGX_config_handle *cfg_h, const char *param_file, const char *options)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_config_create_from_file_and_string " );
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR err;
        ConfigW *cfg = nullptr;

        try
        {
            //use create_*()
            //
            ///cfg = get_mem_manager<ConfigW>().allocate<ConfigW>(AMG_Configuration()).get();
            cfg = create_managed_object<AMG_Configuration, AMGX_config_handle>(cfg_h);
            err = cfg->wrapped()->parseParameterStringAndFile(options, param_file);
            *cfg_h = (AMGX_config_handle)cfg;//need (AMGX_config_handle)(cfg->handle().get()) ??? No!
        }

        AMGX_CATCHES(rc)

        if (rc != AMGX_OK)
            AMGX_CHECK_API_ERROR(rc, NULL)
            //return getCAPIerror(rc);
            else
            {
                AMGX_CHECK_API_ERROR(err, NULL);
            }

        //return getCAPIerror(err);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_config_add_parameters(AMGX_config_handle *cfg_h, const char *options)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_config_add_parameters" );
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR err;
        ///CWrapper<AMGX_config_handle>* cfg;

        try
        {
            ///cfg = (CWrapper<AMGX_config_handle>*)(*cfg_h);
            //cfg = (ConfigW*)(*cfg_h);
            ConfigW cfg(*cfg_h);
            ///((AMG_Configuration *)cfg->hdl())->setAllowConfigurationMod(1);
            cfg.wrapped()->setAllowConfigurationMod(1);
            ///err = ((AMG_Configuration *)cfg->hdl())->parseParameterString(options);
            err = cfg.wrapped()->parseParameterString(options);
            ///((AMG_Configuration *)cfg->hdl())->setAllowConfigurationMod(0);
            cfg.wrapped()->setAllowConfigurationMod(0);
        }

        AMGX_CATCHES(rc)

        if (rc != AMGX_OK)
            AMGX_CHECK_API_ERROR(rc, NULL)
            //return getCAPIerror(rc);
            else
                AMGX_CHECK_API_ERROR(err, NULL)
                //return getCAPIerror(err);
                return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_config_get_default_number_of_rings(AMGX_config_handle cfg_h, int *num_import_rings)
    {
        nvtxRange nvrf(__func__);

        std::string s_scope, s_value, p_scope, p_value;
        AlgorithmType s_algorithm, p_algorithm;
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            ///CWrapper<AMGX_config_handle>* cfg = (CWrapper<AMGX_config_handle> *)cfg_h;
            ConfigW cfg(cfg_h);
            ///AMG_Config *cfg_obj = ((AMG_Configuration *)(cfg->hdl))->getConfigObject();
            AMG_Config *cfg_obj = cfg.wrapped()->getConfigObject();

            if (cfg_obj != NULL)
            {
                // find out what solver and preconditioner are being used
                /* WARNING: notice that there is no need to check the smoother,
                   because in order to use a smoother you must have selected,
                   either solver or preconditioner to be AMG[CLASSICAL|AGGREGATION]. */
                cfg_obj->getParameter<std::string>("solver", s_value, "default", s_scope);
                cfg_obj->getParameter<std::string>("preconditioner", p_value, s_scope, p_scope);
                s_algorithm = cfg_obj->getParameter<AlgorithmType>("algorithm", s_scope);
                p_algorithm = cfg_obj->getParameter<AlgorithmType>("algorithm", p_scope);

                /* WARNING: Two assumptions:
                   (i) this routine assumes that you can not mix CLASSICAL
                   and AGGREGATION AMG in the same config string, because they
                   require different number of rings. For example, you can
                   not solve AGGREGATION AMG coarse level with CLASSICAL AMG,
                   and vice-versa. It seems to be a reasonable assumption.
                   (ii) we are only checking two levels of hierarchy, so that if
                   you use CG, preconditioned by CG, preconditioned by AMG,
                   this routine will not check the AMG in the third level
                   of precodnitioning. */
                if (s_value.compare("AMG") == 0)
                {
                    // if solver is AMG than simply check whether
                    // classical or aggregation path is selected
                    if (s_algorithm == CLASSICAL)
                    {
                        *num_import_rings = 2;
                    }
                    else   //(s_alg == AGGREGATION)
                    {
                        *num_import_rings = 1;
                    }
                }
                else
                {
                    // if solver is not AMG than check preconditioner
                    if (p_value.compare("AMG") == 0)
                    {
                        if (p_algorithm == CLASSICAL)
                        {
                            *num_import_rings = 2;
                        }
                        else   //(p_alg == AGGREGATION)
                        {
                            *num_import_rings = 1;
                        }
                    }
                    else
                    {
                        //neither solver nor precondiioner are AMG
                        *num_import_rings = 1;
                    }
                }
            }
            else
            {
                *num_import_rings = 0;
                return AMGX_RC_BAD_CONFIGURATION;
            }
        }

        AMGX_CATCHES(rc);
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_config_destroy(AMGX_config_handle cfg_h)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_config_destroy " );
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            if (!remove_managed_object<AMGX_config_handle, AMG_Configuration>(cfg_h))
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, NULL)
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, NULL);
        return AMGX_RC_OK;
    }


    AMGX_RC AMGX_API AMGX_solver_create(AMGX_solver_handle *slv, AMGX_resources_handle rsc, AMGX_Mode mode, const AMGX_config_handle cfg_h)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_create" );
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc_solver;
        Resources *resources = NULL;

        try
        {
            ResourceW c_r(rsc);
            ConfigW cfg(cfg_h);

            if (!c_r.wrapped())
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
            }

            resources = c_r.wrapped().get();
            cudaSetDevice(resources->getDevice(0));

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {               \
                   auto* solver = create_managed_mode_object<CASE,AMG_Solver,AMGX_solver_handle>(slv, mode, resources, cfg.wrapped().get()); \
                   solver->set_last_solve_status(AMGX_ST_ERROR); \
                   rc_solver = solver->is_valid() ? AMGX_RC_OK : AMGX_RC_UNKNOWN; \
                 }                      \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        return rc_solver;
    }

    AMGX_RC AMGX_API AMGX_solver_calculate_residual_norm(AMGX_solver_handle solver, AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle x, void *norm_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_solve " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      AMGX_ERROR rcs = solver_calculate_residual_norm<CASE, Matrix, Vector, AMG_Solver>(solver, mtx, rhs, x, resources, norm_vector); \
      AMGX_CHECK_API_ERROR(rcs, resources); break;\
          }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }


    AMGX_RC AMGX_API AMGX_solver_destroy(AMGX_solver_handle slv)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_destroy " );
        //cudaSetDevice(...) is called below because
        //device deallocator must be invoked
        //to free device resources
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_solver_handle>(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      cudaSetDevice(get_mode_object_from<CASE,AMG_Solver,AMGX_solver_handle>(slv)->getResources()->getDevice(0));\
      remove_managed_object<CASE,AMG_Solver,AMGX_solver_handle>(slv); \
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_solver_setup(AMGX_solver_handle slv, AMGX_matrix_handle mtx)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_setup " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_solver_handle>(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      typedef TemplateMode<CASE>::Type TConfig; \
      AMGX_ERROR rcs = set_solver_with_shared<CASE,AMG_Solver,Matrix>(slv, mtx, resources, &AMG_Solver<TConfig>::setup_capi); \
      AMGX_CHECK_API_ERROR(rcs, resources); \
      break;\
          }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources) \
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_solver_resetup(AMGX_solver_handle slv, AMGX_matrix_handle mtx)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_resetup " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_solver_handle>(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      typedef TemplateMode<CASE>::Type TConfig; \
      AMGX_ERROR rcs = set_solver_with_shared<CASE,AMG_Solver,Matrix>(slv, mtx, resources, &AMG_Solver<TConfig>::resetup_capi); \
      AMGX_CHECK_API_ERROR(rcs, resources); \
      break;\
            }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources) \
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_solver_solve(AMGX_solver_handle slv, AMGX_vector_handle rhs, AMGX_vector_handle sol)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_solve " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_solver_handle>(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      AMGX_ERROR rcs = solve_with<CASE,AMG_Solver,Vector>(slv, rhs, sol, resources, false); \
      AMGX_CHECK_API_ERROR(rcs, resources); break;\
          }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_solver_solve_with_0_initial_guess(AMGX_solver_handle slv, AMGX_vector_handle rhs, AMGX_vector_handle sol)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_solve_with_0_initial_guess " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_solver_handle>(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      AMGX_ERROR rcs = solve_with<CASE,AMG_Solver,Vector>(slv, rhs, sol, resources, true); \
      AMGX_CHECK_API_ERROR(rcs, resources); break;\
            }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_matrix_create_impl(AMGX_matrix_handle *mtx, AMGX_resources_handle rsc, AMGX_Mode mode)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_create " );
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR rc_mtx = AMGX_OK;
        Resources *resources = NULL;

        try
        {
            ResourceW c_r(rsc);

            if (!c_r.wrapped() )
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
            }

            resources = c_r.wrapped().get();
            cudaSetDevice(resources->getDevice(0));

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      auto* wmtx = create_managed_mode_object<CASE,Matrix,AMGX_matrix_handle>(mtx, mode); \
      rc_mtx = wmtx->is_valid() ? AMGX_OK : AMGX_ERR_UNKNOWN; \
      wmtx->wrapped()->setResources(resources);\
          }\
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
                default: AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc_mtx, resources)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_create(AMGX_matrix_handle *mtx, AMGX_resources_handle rsc, AMGX_Mode mode)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_create_impl(mtx, rsc, mode);
    }

    AMGX_RC AMGX_API AMGX_matrix_vector_multiply(AMGX_matrix_handle mtx, AMGX_vector_handle x, AMGX_vector_handle y)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_solver_solve " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      AMGX_ERROR rcs = matrix_vector_multiply<CASE,Matrix,Vector>(mtx, x, y, resources); \
      AMGX_CHECK_API_ERROR(rcs, resources); break;\
          }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_matrix_destroy_impl(AMGX_matrix_handle mtx)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_destroy " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      cudaSetDevice(get_mode_object_from<CASE,Matrix,AMGX_matrix_handle>(mtx)->getResources()->getDevice(0));\
      remove_managed_matrix<CASE>(mtx); \
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_destroy(AMGX_matrix_handle mtx)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_destroy_impl(mtx);
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_all_impl(AMGX_matrix_handle mtx, int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag_data)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_upload_all " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        // should change to the convert(). this routine will catch possible memory exceptions and return corresponding errors. temporary catch.
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE:{ \
      rc0 = matrix_upload_all<CASE>(mtx,n,nnz,block_dimx,block_dimy,row_ptrs,col_indices,data,diag_data,resources); \
    }                             \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_all(AMGX_matrix_handle mtx, int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag_data)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_upload_all_impl(mtx, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, data, diag_data);
    }

    AMGX_RC AMGX_API AMGX_matrix_replace_coefficients(AMGX_matrix_handle mtx, int n, int nnz, const void *data, const void *diag_data)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_replace_coefficients " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            //if (!c_mtx || !c_mtx->is_valid()) return AMGX_RC_BAD_PARAMETERS;
            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      rc0 = matrix_replace_coefficients<CASE>(mtx,n,nnz,data,diag_data,resources); \
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_matrix_get_size_impl(const AMGX_matrix_handle mtx, int *n, int *block_dimx, int *block_dimy)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_get_size " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE:  {  \
        typedef Matrix<typename TemplateMode<CASE>::Type> MatrixLetterT; \
        MatrixLetterT* mtx_ptr = get_mode_object_from<CASE,Matrix,AMGX_matrix_handle>(mtx); \
        if (mtx_ptr->manager != NULL) \
        {  \
            if (mtx_ptr->manager->isFineLevelGlued())  \
            {   \
                *n = mtx_ptr->manager->halo_offsets_before_glue[0]; \
            }   \
            else  \
            {   \
                *n = mtx_ptr->get_num_rows();                      \
            }   \
        }   \
        else  \
        {   \
            *n = mtx_ptr->get_num_rows();                      \
        }   \
        *block_dimx = mtx_ptr->get_block_dimx();              \
        *block_dimy = mtx_ptr->get_block_dimy();              \
        }   \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //  return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_get_size(const AMGX_matrix_handle mtx, int *n, int *block_dimx, int *block_dimy)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_get_size_impl(mtx, n, block_dimx, block_dimy);
    }

    AMGX_RC AMGX_API AMGX_matrix_attach_geometry( AMGX_matrix_handle mtx, double *geox, double *geoy, double *geoz, int n)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_attach_geometry " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        int dimension = (geoz == NULL ? 2 : 3);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
      { \
        matrix_attach_geometry<CASE>(mtx, geox, geoy, geoz, n,dimension); \
        break; \
      }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    //AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_attach_coloring( AMGX_matrix_handle mtx, int *row_coloring, int num_rows, int num_colors)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_attach_coloring " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
      { \
        matrix_attach_coloring<CASE>(mtx, row_coloring, num_rows, num_colors); \
        break; \
      }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_sort(AMGX_matrix_handle mtx)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_matrix_sort " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_matrix_handle>(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        rc0 = matrix_sort<CASE>(mtx); \
        } \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }


// previously: AMGX_vector_create(AMGX_vector_handle *ret, AMGX_Mode mode)
    AMGX_RC AMGX_vector_create_impl(AMGX_vector_handle *vec, AMGX_resources_handle rsc, AMGX_Mode mode)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_create " );
        Resources *resources = NULL;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR rc_vec = AMGX_OK;

        try
        {
            ResourceW c_r(rsc);

            if (!c_r.wrapped()) { AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL); }

            resources = c_r.wrapped().get();
            cudaSetDevice(resources->getDevice(0));

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      auto* wvec = create_managed_mode_object<CASE,Vector,AMGX_vector_handle>(vec, mode); \
      rc_vec = wvec->is_valid() ? AMGX_OK : AMGX_ERR_UNKNOWN; \
      wvec->wrapped()->setResources(resources); \
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORINTVEC_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc_vec, resources)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_API AMGX_vector_create(AMGX_vector_handle *vec, AMGX_resources_handle rsc, AMGX_Mode mode)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_create_impl(vec, rsc, mode);
    }

    AMGX_RC AMGX_vector_destroy_impl(AMGX_vector_handle vec)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_destroy " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(vec, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      cudaSetDevice(get_mode_object_from<CASE,Vector,AMGX_vector_handle>(vec)->getResources()->getDevice(0));\
      remove_managed_object<CASE,Vector,AMGX_vector_handle>(vec);\
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORINTVEC_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_vector_destroy(AMGX_vector_handle vec)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_destroy_impl(vec);
    }

    AMGX_RC AMGX_vector_upload_impl(AMGX_vector_handle vec, int n, int block_dim, const void *data)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_upload " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(vec, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {\
      rc0 = vector_upload<CASE>(vec, n, block_dim, data); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_vector_upload(AMGX_vector_handle vec, int n, int block_dim, const void *data)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_upload_impl(vec, n, block_dim, data);
    }

    AMGX_RC AMGX_vector_set_zero_impl(AMGX_vector_handle vec, int n, int block_dim)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_set_zero " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(vec, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

//since resize falls directly into the thrust we can only catch bad_alloc here:
        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        rc0 = vector_set_zero<CASE>(vec, n, block_dim, resources); \
        }\
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_vector_set_zero(AMGX_vector_handle vec, int n, int block_dim)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_set_zero_impl(vec, n, block_dim);
    }

    AMGX_RC AMGX_API AMGX_vector_set_random(AMGX_vector_handle vec, int n)
    {
        nvtxRange nvrf(__func__);

        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(vec, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

//since resize falls directly into the thrust we can only catch bad_alloc here:
        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        rc0 = vector_set_random<CASE>(vec, n, resources); \
        }\
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
    }

    AMGX_RC AMGX_vector_download_impl(const AMGX_vector_handle vec, void *data)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_download " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(vec, &resources)), NULL)
        //if (!c_vec || !c_vec->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {\
        rc0 = vector_download_impl<CASE>(vec, data);\
        } \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_vector_download(const AMGX_vector_handle vec, void *data)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_download_impl(vec, data);
    }

    AMGX_RC AMGX_vector_get_size_impl(const AMGX_vector_handle vec, int *n, int *block_dim)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_get_size " );
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromVectorHandle(vec, &resources)), NULL)
        //if (!c_vec || !c_vec->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from<AMGX_vector_handle>(vec);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {\
          rc0 = vector_get_size<CASE>(vec, n, block_dim); \
        }\
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_vector_get_size(const AMGX_vector_handle vec, int *n, int *block_dim)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_get_size_impl(vec, n, block_dim);
    }

#ifdef AMGX_WITH_MPI
    AMGX_RC AMGX_API AMGX_write_system_distributed(const AMGX_matrix_handle mtx, const AMGX_vector_handle rhs, const AMGX_vector_handle sol, const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_write_full_system " );
        AMGX_Mode mode;
        Resources *resources = NULL;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            write_system_preamble(mtx, rhs, sol, resources, mode);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {\
        rc0 = mpi_write_system_distributed<CASE>(mtx, rhs, sol, filename, allocated_halo_depth, num_partitions, partition_sizes, partition_vector_size, partition_vector, rc); \
    }                                           \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }
#else
    AMGX_RC AMGX_API AMGX_write_system_distributed(const AMGX_matrix_handle mtx, const AMGX_vector_handle rhs, const AMGX_vector_handle sol, const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }
#endif

    AMGX_RC AMGX_API AMGX_write_system(const AMGX_matrix_handle mtx, const AMGX_vector_handle rhs, const AMGX_vector_handle sol, const char *filename)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_write_full_system " );
        AMGX_Mode mode;
        Resources *resources = NULL;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            write_system_preamble(mtx, rhs, sol, resources, mode);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {\
        rc0 = write_system<CASE>(mtx, rhs, sol, filename, rc); \
      } break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
                    //return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }



    AMGX_RC AMGX_API AMGX_solver_get_iterations_number(AMGX_solver_handle slv, int *n)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_get_iterations_number " );
        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL)
        //if (!c_solver || !c_solver->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
          solver_get_iterations_number<CASE>(slv, n); \
        } \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    *n = -1;
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_solver_get_iteration_residual(AMGX_solver_handle slv, int it, int idx, double *res)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_get_iteration_residual " );
        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL)
        //if (!c_solver || !c_solver->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        *res = -1.;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
            rc0 = solver_get_iteration_residual<CASE>(slv, it, idx, res); \
        } \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_get_build_info_strings(char **version, char **date, char **time)
    {
        nvtxRange nvrf(__func__);

        *version = const_cast<char *>(__AMGX_BUILD_ID__);
        *date = const_cast<char *>(__AMGX_BUILD_DATE__);
        *time = const_cast<char *>(__AMGX_BUILD_TIME__);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_pin_memory_impl(void *ptr, unsigned int bytes)
    {
        AMGX_CPU_PROFILER( "AMGX_pin_memory " );

        if (ptr == 0)
        {
            return AMGX_RC_OK;
        }

        cudaError_t rc = cudaHostRegister(ptr, bytes, cudaHostRegisterMapped);

        if (cudaSuccess == rc)
        {
            return AMGX_RC_OK;
        }
        else
            AMGX_CHECK_API_ERROR(AMGX_ERR_CUDA_FAILURE, NULL)
            return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_pin_memory(void *ptr, unsigned int bytes)
    {
        nvtxRange nvrf(__func__);

        return AMGX_pin_memory_impl(ptr, bytes);
    }

    AMGX_RC AMGX_unpin_memory_impl(void *ptr)
    {
        AMGX_CPU_PROFILER( "AMGX_unpin_memory " );

        if (ptr == 0)
        {
            return AMGX_RC_OK;
        }

        cudaError_t rc = cudaHostUnregister(ptr);

        if (cudaSuccess == rc)
        {
            return AMGX_RC_OK;
        }
        else
            AMGX_CHECK_API_ERROR(AMGX_ERR_CUDA_FAILURE, NULL)
            return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_unpin_memory(void *ptr)
    {
        nvtxRange nvrf(__func__);

        return AMGX_unpin_memory_impl(ptr);
    }

    AMGX_RC AMGX_API AMGX_solver_get_status(AMGX_solver_handle slv, AMGX_SOLVE_STATUS *st)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromSolverHandle(slv, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(slv);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
            solver_get_status<CASE>(slv, st); \
              } \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_solver_register_print_callback(AMGX_print_callback func)
    {
        nvtxRange nvrf(__func__);

        return AMGX_register_print_callback(func);
    }

    AMGX_RC AMGX_register_print_callback(AMGX_print_callback func)
    {
        nvtxRange nvrf(__func__);

        amgx_output = func;
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_get_error_string(AMGX_RC err, char *buf, int buf_len)
    {
        nvtxRange nvrf(__func__);

        AMGX_GetErrorString(getAMGXerror(err), buf, buf_len);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_install_signal_handler()
    {
        nvtxRange nvrf(__func__);

        SignalHandler::hook();
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_reset_signal_handler()
    {
        nvtxRange nvrf(__func__);

        SignalHandler::unhook();
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_get_api_version(int *major, int *minor)
    {
        nvtxRange nvrf(__func__);

        *major = __AMGX_API_VERSION_MAJOR;
        *minor = __AMGX_API_VERSION_MINOR;
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_matrix_get_nnz_impl(const AMGX_matrix_handle mtx, int *nnz)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        //if (!c_mtx || !c_mtx->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        *nnz = get_mode_object_from<CASE,Matrix,AMGX_matrix_handle>(mtx)->get_num_nz();} \
        break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_get_nnz(const AMGX_matrix_handle mtx, int *nnz)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_get_nnz_impl(mtx, nnz);
    }

    AMGX_RC AMGX_matrix_download_all_impl(const AMGX_matrix_handle mtx, int *row_ptrs, int *col_indices, void *data, void **diag_data)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        //if (!c_mtx || !c_mtx->is_valid()) return AMGX_RC_BAD_PARAMETERS;
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        matrix_download_all<CASE>(mtx, row_ptrs, col_indices, data, diag_data); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_download_all(const AMGX_matrix_handle mtx, int *row_ptrs, int *col_indices, void *data, void **diag_data)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_download_all_impl(mtx, row_ptrs, col_indices, data, diag_data);
    }

    AMGX_RC AMGX_API AMGX_vector_bind_impl(AMGX_vector_handle vec, const AMGX_matrix_handle matrix)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(matrix, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(matrix);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {    \
       vector_bind<CASE>(vec, matrix); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_vector_bind(AMGX_vector_handle vec, const AMGX_matrix_handle mtx)
    {
        nvtxRange nvrf(__func__);

        return AMGX_vector_bind_impl(vec, mtx);
    }

#ifdef AMGX_WITH_MPI
    AMGX_RC AMGX_read_system_distributed_impl(AMGX_matrix_handle mtx,
            AMGX_vector_handle rhs,
            AMGX_vector_handle sol,
            const char *filename,
            int allocated_halo_depth,
            int num_partitions,
            const int *partition_sizes,
            int partition_vector_size,
            const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_read_system_distributed " );
        std::stringstream msg;
        AMGX_Mode mode = AMGX_unset;
        unsigned int props = io_config::NONE;
        Resources *resources = NULL;
        AMGX_ERROR read_error = AMGX_OK;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            rc0 = read_system_preamble(mtx, rhs, sol, resources, mode, props);

            if (rc0 != AMGX_RC_OK)
            {
                return rc0;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)

        //WARNING: num_partitions (= # of ranks) might be set without anything else being set. Removing this error check for now.
        //if (partition_sizes == NULL && num_partitions != 0 || partition_sizes != NULL && num_partitions == 0)
        //  AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)

        if (partition_vector == NULL && partition_vector_size != 0 || partition_vector != NULL && partition_vector_size == 0)
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
            if (partition_vector == NULL && partition_sizes != NULL)
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
                int num_ranks = 1, part = 0;

        if (partition_sizes == NULL && partition_vector != NULL)
        {
            // solve for partition sizes, if they are not provided
            //num_partitions = std::max_element(partition_vector, partition_vector + partition_vector_size);
            num_partitions = 0;

            for (int i = 0; i < partition_vector_size; i++)
            {
                num_partitions = max(num_partitions, partition_vector[i]);
            }

            num_partitions++;
            msg << "Processing partition vector, consisting of " << partition_vector_size << " rows and " <<  num_partitions << " partitions\n";
            //printf("Processing partition vector, consisting of %d rows and %d partitions\n", partition_vector_size, num_partitions); // change output
        }

        rc = AMGX_OK;
        rc0 = AMGX_RC_OK;

        try
        {
            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {    \
       rc0 = read_system_distributed<CASE>(mtx, rhs, sol, filename, allocated_halo_depth, num_partitions, partition_sizes, partition_vector_size, partition_vector, msg, num_ranks, resources, part, props, read_error); \
        } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        amgx_distributed_output(msg.str().c_str(), msg.str().length());
        AMGX_CHECK_API_ERROR(rc, resources)
        AMGX_CHECK_API_ERROR(read_error, resources)
        return rc0;
    }
#else
    AMGX_RC AMGX_read_system_distributed_impl(AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle sol,  const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }
#endif

    AMGX_RC AMGX_matrix_set_boundary_separation_impl(AMGX_matrix_handle mtx,  int boundary_separation)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        //if (!c_mtx) return AMGX_RC_BAD_PARAMETERS;
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE:                                                                             \
    {                                                                                                                \
      get_mode_object_from<CASE,Matrix,AMGX_matrix_handle>(mtx)->set_allow_boundary_separation(boundary_separation);\
    }                                                                                                                \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_set_boundary_separation(AMGX_matrix_handle mtx,  int boundary_separation)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_set_boundary_separation_impl(mtx, boundary_separation);
    }

    AMGX_RC AMGX_read_system_maps_one_ring_impl(int *n,
            int *nnz,
            int *block_dimx,
            int *block_dimy,
            int **row_ptrs,
            int **col_indices,
            void **data,
            void **diag_data,
            void **rhs,
            void **sol,
            int *num_neighbors,
            int **neighbors,
            int **btl_sizes,
            int ***btl_maps,
            int **lth_sizes,
            int ***lth_maps,
            AMGX_resources_handle rsc,
            AMGX_Mode mode,
            const char *filename,
            int allocated_halo_depth,
            int num_partitions,
            const int *partition_sizes,
            int partition_vector_size,
            const int *partition_vector,
            int64_t **local_to_global_map)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_ERROR rc_rs = AMGX_OK;

        try
        {
            ResourceW c_r(rsc);

            if (!c_r.wrapped())
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
            }
            else
            {
                resources = c_r.wrapped().get();
            }
        }

        AMGX_CATCHES(rc_rs)
        AMGX_CHECK_API_ERROR(rc_rs, resources)
        std::string solver_scope, solver_value;
        std::string precond_scope, precond_value;
        AlgorithmType algorithm_s, algorithm_p;
        resources->getResourcesConfig()->getParameter<std::string>("solver", solver_value, "default", solver_scope);
        algorithm_s = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", solver_scope);
        resources->getResourcesConfig()->getParameter<std::string>("preconditioner", precond_value, solver_scope, precond_scope);
        algorithm_p = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", precond_scope);

        if ((local_to_global_map == NULL) // means we're in AMGX_read_system_one_ring not in AMGX_read_system_global
                && algorithm_s == CLASSICAL && algorithm_p == CLASSICAL)
        {
            std::stringstream msg;
            msg << "CLASSICAL is not supported in AMGX_read_system_maps_one_ring.\n";
            amgx_distributed_output(msg.str().c_str(), msg.str().length());
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
        }

        // import data
        AMGX_matrix_handle A_part;
        AMGX_vector_handle b_dev, x_dev;
        AMGX_RC rc = AMGX_matrix_create_impl(&A_part, rsc, mode);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_matrix_set_boundary_separation_impl(A_part, 0); // switch off reordering, since will download later

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_vector_create_impl(&x_dev, rsc, mode);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_vector_create_impl(&b_dev, rsc, mode);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_read_system_distributed_impl(A_part, b_dev, x_dev,  filename, allocated_halo_depth, num_partitions, partition_sizes, partition_vector_size, partition_vector);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_matrix_get_nnz_impl(A_part, nnz);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_matrix_get_size_impl(A_part, n, block_dimx, block_dimy);

        if (rc != AMGX_RC_OK) { return rc; }

        int x_sz, x_block_dim;
        rc = AMGX_vector_get_size_impl(x_dev, &x_sz, &x_block_dim);

        if (rc != AMGX_RC_OK) { return rc; }

        if (x_sz == 0)
        {
            std::stringstream msg;
            msg << "Initializing solution vector with zeroes...\n";
            amgx_distributed_output(msg.str().c_str(), msg.str().length());
            rc = AMGX_vector_set_zero_impl(x_dev, *n, *block_dimy);

            if (rc != AMGX_RC_OK) { return rc; }
        }

        int sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matDouble)) ? sizeof(double) : sizeof(float);
        int sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)) ? sizeof(double) : sizeof(float);
        *sol = get_c_arr_mem_manager().allocate(sizeof_v_val * (*n) * (*block_dimx));
        *rhs = get_c_arr_mem_manager().allocate(sizeof_v_val * (*n) * (*block_dimy));
        rc = AMGX_vector_download_impl(b_dev, *rhs);

        if (rc != AMGX_RC_OK) { return rc; }

        //save partitioned vectors on host
        rc = AMGX_vector_download_impl(x_dev, *sol);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_vector_destroy_impl(x_dev);

        if (rc != AMGX_RC_OK) { return rc; }

        rc = AMGX_vector_destroy_impl(b_dev);

        if (rc != AMGX_RC_OK) { return rc; }

        int block_size = (*block_dimx) * (*block_dimy);
        *row_ptrs = (int *)get_c_arr_mem_manager().allocate((*n + 1) * sizeof(int));
        *col_indices = (int *)get_c_arr_mem_manager().allocate((*nnz) * sizeof(int));
        *data = get_c_arr_mem_manager().allocate((*nnz) * block_size * sizeof_m_val);
        *diag_data = NULL; // will be allocated in AMGX_matrix_download_all if the matrix has DIAG property
        // save matrix before reordering.
        rc = AMGX_matrix_download_all_impl(A_part, *row_ptrs, *col_indices, *data, diag_data);

        if (rc != AMGX_RC_OK) { return rc; }

        AMGX_ERROR nvrc = AMGX_OK;

        try
        {
            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
        read_system_maps_one_ring_impl<CASE>(A_part, num_neighbors, neighbors, btl_sizes, btl_maps, lth_sizes, lth_maps, local_to_global_map); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(nvrc)
        AMGX_CHECK_API_ERROR(nvrc, resources)
        rc = AMGX_matrix_destroy_impl(A_part);

        if (rc != AMGX_RC_OK) { return rc; }

        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_read_system_maps_one_ring(int *n, int *nnz, int *block_dimx, int *block_dimy, int **row_ptrs, int **col_indices, void **data, void **diag_data, void **rhs, void **sol, int *num_neighbors, int **neighbors, int **btl_sizes, int ***btl_maps, int **lth_sizes, int ***lth_maps, AMGX_resources_handle rsc, AMGX_Mode mode, const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        int64_t **local_to_global_map = NULL;
        return AMGX_read_system_maps_one_ring_impl(n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, data, diag_data, rhs, sol, num_neighbors, neighbors, btl_sizes, btl_maps, lth_sizes, lth_maps, rsc, mode, filename, allocated_halo_depth, num_partitions, partition_sizes, partition_vector_size, partition_vector, local_to_global_map);
    }

    AMGX_RC AMGX_API AMGX_read_system_global(int *n, int *nnz, int *block_dimx, int *block_dimy, int **row_ptrs, void **col_indices_global, void **data, void **diag_data, void **rhs, void **sol, AMGX_resources_handle rsc, AMGX_Mode mode, const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        // TODO: we can avoid one-ring construction since we don't need it in this function
        // although the overhead is probably small and this won't be benchmarked anyways
        // so it's more convenient to just reuse AMGX_read_system_maps_one_ring
        int *col_indices;
        int *btl_sizes = NULL;
        int **btl_maps = NULL;
        int *lth_sizes = NULL;
        int **lth_maps = NULL;
        int num_neighbors;
        int *neighbors = NULL;
        int64_t *local_to_global_map = NULL;  // set to flag the following function that we're coming from read_system_global (i.e. we need local_to_global_map values)
        AMGX_RC rc = AMGX_read_system_maps_one_ring_impl(n, nnz, block_dimx, block_dimy, row_ptrs, &col_indices, data, diag_data, rhs, sol, &num_neighbors, &neighbors, &btl_sizes, &btl_maps, &lth_sizes, &lth_maps, rsc, mode, filename, allocated_halo_depth, num_partitions, partition_sizes, partition_vector_size, partition_vector, &local_to_global_map);
        Resources *resources = NULL;
        AMGX_ERROR rc_rs = AMGX_OK;

        try
        {
            ResourceW c_r(rsc);

            if (!c_r.wrapped())
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
            }
            else
            {
                resources = c_r.wrapped().get();
            }
        }

        AMGX_CATCHES(rc_rs)
        AMGX_CHECK_API_ERROR(rc_rs, resources)
        // get global number of rows
        int num_rows_global;
        int my_id;
#ifdef AMGX_WITH_MPI
        MPI_Comm_rank(*resources->getMpiComm(), &my_id);
        MPI_Allreduce(&(*n), &num_rows_global, 1, MPI_INT, MPI_SUM, *resources->getMpiComm());
#else
        my_id = 0;
        num_rows_global = (*n);
#endif
        // setup partition vector
        int num_ranks;
#ifdef AMGX_WITH_MPI
        MPI_Comm_size(*resources->getMpiComm(), &num_ranks);
#else
        num_ranks = 1;
#endif
        int *partitionVec = (int *)get_c_arr_mem_manager().allocate(num_rows_global * sizeof(int));

        if (partition_vector == NULL)
        {
            // initialize equal partitioning
            int *scanPartSize = (int *)get_c_arr_mem_manager().allocate((num_ranks + 1) * sizeof(int));

            for (int p = 0; p < num_ranks; p++)
            {
                scanPartSize[p] = p * num_rows_global / num_ranks;
            }

            scanPartSize[num_ranks] = num_rows_global;
            int p = 0;

            for (int i = 0; i < num_rows_global; i++)
            {
                if (i >= scanPartSize[p + 1]) { p++; }

                partitionVec[i] = p;
            }
        }
        else
        {
            // use existing partition info
            for (int i = 0; i < num_rows_global; i++)
            {
                partitionVec[i] = partition_vector[i];
            }
        }

        // allocate global indices array
        *col_indices_global = get_c_arr_mem_manager().allocate((*nnz) * sizeof(int64_t));
        // global to local mapping for non-halo (interior)
        // compute partition offsets (based on number of elements per partition)
        int64_t *partition_offsets = get_c_arr_mem_manager().callocate<int64_t>(num_ranks + 1);

        for (int i = 0; i < num_rows_global; i++)
        {
            int pvi = partitionVec[i];
            partition_offsets[pvi + 1]++;
        }

        thrust::inclusive_scan(partition_offsets, partition_offsets + num_ranks + 1, partition_offsets);
        // compute partition map (which tells you how the global elements are mapped into the partitions)
        int64_t *partition_map = get_c_arr_mem_manager().callocate<int64_t>(num_rows_global);

        for (int i = 0; i < num_rows_global; i++)
        {
            int     pvi = partitionVec[i];
            int64_t poi = partition_offsets[pvi];
            partition_map[poi] = i;
            //increment used offset/counter for the next iteration
            partition_offsets[pvi]++;
        }

        // restore the offsets back to their original setting
        for (int i = 0; i < num_rows_global; i++)
        {
            int     pvi = partitionVec[i];
            partition_offsets[pvi]--;
        }

        // find global column indices, simply use local_to_global map
        for (int i = 0; i < (*nnz); i++)
        {
            int col = col_indices[i];

            if (col >= (*n))
            {
                ((int64_t *)*col_indices_global)[i] = partition_map[local_to_global_map[col - (*n)]];
            }
            else
            {
                ((int64_t *)*col_indices_global)[i] = partition_map[partition_offsets[my_id] + col];
            }
        }

        // free (temporary) host memory
        get_c_arr_mem_manager().free(partitionVec);
        get_c_arr_mem_manager().free(partition_offsets);
        get_c_arr_mem_manager().free(partition_map);
        return rc;
    }

    AMGX_RC AMGX_free_system_maps_one_ring_impl(int *row_ptrs, int *col_indices, void *data, void *diag_data, void *rhs, void *sol, int num_neighbors, int *neighbors, int *btl_sizes, int **btl_maps, int *lth_sizes, int **lth_maps)
    {
        nvtxRange nvrf(__func__);

        if (row_ptrs != NULL) { get_c_arr_mem_manager().free(row_ptrs); }

        if (col_indices != NULL) { get_c_arr_mem_manager().free(col_indices); }

        if (neighbors != NULL) { get_c_arr_mem_manager().free(neighbors); }

        if (btl_maps != NULL)
        {
            for (int i = 0; i < num_neighbors; i++)
                if (btl_maps[i] != NULL) { free(btl_maps[i]); }

            free(btl_maps);
        }

        if (lth_maps != NULL)
        {
            for (int i = 0; i < num_neighbors; i++)
                if (lth_maps[i] != NULL) { free(lth_maps[i]); }

            free(lth_maps);
        }

        if (btl_sizes != NULL) { free(btl_sizes); }

        if (lth_sizes != NULL) { free(lth_sizes); }

        if (data != NULL) { get_c_arr_mem_manager().free(data); }

        if (diag_data != NULL) { get_c_arr_mem_manager().free(diag_data); }

        if (rhs != NULL) { get_c_arr_mem_manager().free(rhs); }

        if (sol != NULL) { get_c_arr_mem_manager().free(sol); }

        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_free_system_maps_one_ring(int *row_ptrs, int *col_indices, void *data, void *diag_data, void *rhs, void *sol, int num_neighbors, int *neighbors, int *btl_sizes, int **btl_maps, int *lth_sizes, int **lth_maps)
    {
        nvtxRange nvrf(__func__);

        return AMGX_free_system_maps_one_ring_impl(row_ptrs, col_indices, data, diag_data, rhs, sol, num_neighbors, neighbors, btl_sizes, btl_maps, lth_sizes, lth_maps);
    }

    AMGX_RC AMGX_matrix_comm_from_maps_one_ring_impl(   AMGX_matrix_handle mtx,
            int allocated_halo_depth,
            int max_num_neighbors,
            const int *neighbors,
            const int *send_sizes,
            int const **send_maps,
            const int *recv_sizes,
            int const  **recv_maps)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        std::string solver_scope, solver_value;
        std::string precond_scope, precond_value;
        AlgorithmType algorithm_s, algorithm_p;
        resources->getResourcesConfig()->getParameter<std::string>("solver", solver_value, "default", solver_scope);
        algorithm_s = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", solver_scope);
        resources->getResourcesConfig()->getParameter<std::string>("preconditioner", precond_value, solver_scope, precond_scope);
        algorithm_p = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", precond_scope);

        if (algorithm_s == CLASSICAL && algorithm_p == CLASSICAL)
        {
            std::stringstream msg;
            msg << "CLASSICAL is not supported in AMGX_read_system_maps_one_ring.\n";
            amgx_distributed_output(msg.str().c_str(), msg.str().length());
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources)
        }

        AMGX_RC rc0;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      rc0 = matrix_comm_from_maps_one_ring<CASE>(mtx, allocated_halo_depth, max_num_neighbors, neighbors, send_sizes, send_maps, recv_sizes, recv_maps, resources); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_comm_from_maps_one_ring(AMGX_matrix_handle mtx, int allocated_halo_depth, int max_num_neighbors, const int *neighbors, const int *send_sizes, int const **send_maps, const int *recv_sizes, int const  **recv_maps)
    {
        nvtxRange nvrf(__func__);

        return AMGX_matrix_comm_from_maps_one_ring_impl(mtx, allocated_halo_depth, max_num_neighbors, neighbors, send_sizes, send_maps, recv_sizes, recv_maps);
    }

#ifdef AMGX_WITH_MPI
    AMGX_RC AMGX_API AMGX_read_system_distributed(AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle sol,  const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        return AMGX_read_system_distributed_impl(mtx, rhs, sol, filename, allocated_halo_depth, num_partitions, partition_sizes, partition_vector_size, partition_vector);
    }
#else
    AMGX_RC AMGX_API AMGX_read_system_distributed(AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle sol,  const char *filename, int allocated_halo_depth, int num_partitions, const int *partition_sizes, int partition_vector_size, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }
#endif

#ifdef AMGX_WITH_MPI
    AMGX_RC AMGX_API AMGX_generate_distributed_poisson_7pt( AMGX_matrix_handle mtx,
            AMGX_vector_handle rhs,
            AMGX_vector_handle sol,
            int allocated_halo_depth,
            int num_import_rings,
            int nx,
            int ny,
            int nz,
            int px,
            int py,
            int pz)
    {
        nvtxRange nvrf(__func__);

        /* This routine will create 3D (7 point) discretization of the Poisson operator.
           The discretization is performed on a the 3D domain consisting of nx, ny and nz
           points in x-, y- and z-dimension, respectively. This 3D domain will be replicated
           in px, py and pz times in x-, y- and z-dimension. Therefore, creating a large
           "cube", composed of smaller "sub-cubes" each of which is going to be handled on
           a separate ranks/processor. Later on p, q and r will indicate the position of the
           "sub-cube" in the "cube" for a particular rank. Finally, the rhs and solution are
           set to a vector of ones and zeros, respectively. */
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      rc0 = generate_distributed_poisson_7pt<CASE>(mtx, rhs, sol, allocated_halo_depth, num_import_rings, nx, ny, nz, px, py, pz); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        return AMGX_OK != rc ? getCAPIerror_x(rc) : rc0;
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_all_global( AMGX_matrix_handle mtx,
            int n_global,
            int n,
            int nnz,
            int block_dimx,
            int block_dimy,
            const int *row_ptrs,
            const void *col_indices_global,
            const void *data,
            const void *diag_data,
            int allocated_halo_depth,
            int num_import_rings,
            const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      rc0 = matrix_upload_all_global<CASE>(mtx, n_global, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices_global, data, diag_data, allocated_halo_depth, num_import_rings, partition_vector); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        return AMGX_OK != rc ? getCAPIerror_x(rc) : rc0;
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_all_global_32(AMGX_matrix_handle mtx, int n_global, int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const void *col_indices_global, const void *data, const void *diag_data, int allocated_halo_depth, int num_import_rings, const int *partition_vector)
    {
        nvtxRange nvrf(__func__);

        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      rc0 = matrix_upload_all_global_32<CASE>(mtx, n_global, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices_global, data, diag_data, allocated_halo_depth, num_import_rings, partition_vector); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        return AMGX_OK != rc ? getCAPIerror_x(rc) : rc0;
    }   
    
    AMGX_RC AMGX_API AMGX_matrix_upload_distributed(AMGX_matrix_handle mtx,
            int n_global,
            int n,
            int nnz,
            int block_dimx,
            int block_dimy,
            const int *row_ptrs,
            const void *col_indices_global,
            const void *data,
            const void *diag_data,
            AMGX_distribution_handle dist)
    {
        nvtxRange nvrf(__func__);

        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      rc0 = matrix_upload_distributed<CASE>(mtx, n_global, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices_global, data, diag_data, dist); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    return AMGX_RC_BAD_MODE;
            }
        }

        AMGX_CATCHES(rc)
        return AMGX_OK != rc ? getCAPIerror_x(rc) : rc0;
    }

#else
    AMGX_RC AMGX_API AMGX_generate_distributed_poisson_7pt(AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle sol, int allocated_halo_depth, int num_import_rings, int nx, int ny, int nz, int px, int py, int pz)
    {
        nvtxRange nvrf(__func__);

        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_all_global(AMGX_matrix_handle mtx, const int n_global, const int n, const int nnz, const int block_dimx, const int block_dimy, const int *row_ptrs, const void *col_indices_global, const void *data, const void *diag_data, int allocated_halo_depth, int num_import_rings, const int *partition_vector)
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_all_global_int(AMGX_matrix_handle mtx, const int n_global, const int n, const int nnz, const int block_dimx, const int block_dimy, const int *row_ptrs, const void *col_indices_global, const void *data, const void *diag_data, int allocated_halo_depth, int num_import_rings, const int *partition_vector)
    {
        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_matrix_upload_distributed(AMGX_matrix_handle mtx, int n_global, int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const void *col_indices_global, const void *data, const void *diag_data, AMGX_distribution_handle distribution)
    {
        nvtxRange nvrf(__func__);

        AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        return AMGX_RC_OK;
    }
#endif

    AMGX_RC AMGX_API AMGX_matrix_comm_from_maps(AMGX_matrix_handle mtx, int allocated_halo_depth, int num_import_rings, int max_num_neighbors, const int *neighbors, const int *send_ptrs, int const *send_maps, const int *recv_ptrs, int const  *recv_maps)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      rc0 = matrix_comm_from_maps<CASE>(mtx, allocated_halo_depth, num_import_rings, max_num_neighbors, neighbors, send_ptrs, send_maps, recv_ptrs, recv_maps); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return rc0;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_matrix_get_size_neighbors(const AMGX_matrix_handle mtx, int *num_neighbors)
    {
        nvtxRange nvrf(__func__);

        Resources *resources = NULL;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromMatrixHandle(mtx, &resources)), NULL)
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_Mode mode = get_mode_from(mtx);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    { \
      *num_neighbors = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(mtx)->manager->num_neighbors(); \
    } \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
        //return getCAPIerror(rc);
    }

    AMGX_RC AMGX_API AMGX_resources_create(AMGX_resources_handle *rsc, AMGX_config_handle cfg_h, void *comm, int device_num, const int *devices)
    {
        nvtxRange nvrf(__func__);

        if (rsc == NULL || devices == NULL)
        {
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        }

        AMGX_ERROR rc = AMGX_OK;

        try
        {
            ConfigW cfg(cfg_h);
            auto *resources = create_managed_object<Resources, AMGX_resources_handle>(rsc, cfg.wrapped().get(), comm, device_num, devices);
        }

        AMGX_CATCHES(rc)

        if (rc != AMGX_OK)
        {
            AMGX_CHECK_API_ERROR(rc, NULL)
        }

        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_resources_create_simple(AMGX_resources_handle *rsc, AMGX_config_handle cfg_h)
    {
        nvtxRange nvrf(__func__);

        const size_t num_devices = 1;
        const int devices[1] = { 0 };
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            ConfigW cfg(cfg_h);
            auto *resources = create_managed_object<Resources, AMGX_resources_handle>(rsc, cfg.wrapped().get(), nullptr, num_devices, devices);
        }

        AMGX_CATCHES(rc)

        if (rc != AMGX_OK)
        {
            AMGX_CHECK_API_ERROR(rc, NULL)
        }

        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_resources_destroy(AMGX_resources_handle rsc)
    {
        nvtxRange nvrf(__func__);

        AMGX_ERROR rc = AMGX_OK;

        try
        {
            bool found = remove_managed_object<AMGX_resources_handle, Resources>(rsc);
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, NULL);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_distribution_create(AMGX_distribution_handle *dist, AMGX_config_handle cfg)
    {
        nvtxRange nvrf(__func__);

        AMGX_ERROR rc = AMGX_OK;
        try 
        {
            auto *mdist = create_managed_object<MatrixDistribution, AMGX_distribution_handle>(dist);
            if (cfg != NULL)
            {
                int ring;
                rc = getAMGXerror(AMGX_config_get_default_number_of_rings(cfg, &ring));
                mdist->wrapped()->setAllocatedHaloDepth(ring);
                mdist->wrapped()->setNumImportRings(ring);
            }
        }
        AMGX_CATCHES(rc);
        if (rc != AMGX_OK)
        {
            AMGX_CHECK_API_ERROR(rc, NULL);
        }
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_distribution_destroy(AMGX_distribution_handle dist)
    {
        nvtxRange nvrf(__func__);

        AMGX_ERROR rc = AMGX_OK;
        try
        {
            if (!remove_managed_object<AMGX_distribution_handle, MatrixDistribution>(dist)) 
            {
                rc = AMGX_ERR_BAD_PARAMETERS;
            }
        }
        AMGX_CATCHES(rc);
        if (rc != AMGX_OK)
        {
            AMGX_CHECK_API_ERROR(rc, NULL);
        }
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_distribution_set_partition_data(AMGX_distribution_handle dist, AMGX_DIST_PARTITION_INFO info, const void *partition_data)
    {
        nvtxRange nvrf(__func__);

        if (dist == NULL || partition_data == NULL) 
        {
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        }
        typedef CWrapHandle<AMGX_distribution_handle, MatrixDistribution> MatrixDistributionW;
        MatrixDistributionW wrapDist(dist);
        MatrixDistribution &mdist = *wrapDist.wrapped();
        switch (info)
        {
            case AMGX_DIST_PARTITION_VECTOR:
                mdist.setPartitionVec((const int*)partition_data);
                break;
            case AMGX_DIST_PARTITION_OFFSETS:
                mdist.setPartitionOffsets(partition_data);
                break;
            default:
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
                break;
        }
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_distribution_set_32bit_colindices(AMGX_distribution_handle dist, int use32bit)
    {
        nvtxRange nvrf(__func__);

        if (dist == NULL)
        {
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);
        }
        typedef CWrapHandle<AMGX_distribution_handle, MatrixDistribution> MatrixDistributionW;
        MatrixDistributionW wrapDist(dist);
        MatrixDistribution &mdist = *wrapDist.wrapped();
        mdist.set32BitColIndices(use32bit);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_read_system(AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle sol, const char *filename)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_read_system " );
        Resources *resources = NULL;
        AMGX_Mode mode = AMGX_unset;
        unsigned int props = io_config::NONE;
        AMGX_ERROR read_error = AMGX_OK;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            rc0 = read_system_preamble(mtx, rhs, sol, resources, mode, props, true);

            if (rc0 != AMGX_RC_OK)
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);    //return AMGX_RC_BAD_PARAMETERS;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        std::string solver_value, solver_scope;
        resources->getResourcesConfig()->getParameter<std::string>("solver", solver_value, "default", solver_scope);
        int rhs_from_a;
        resources->getResourcesConfig()->getParameter<int>("rhs_from_a", rhs_from_a, "default", solver_scope);

        if (rhs_from_a == 1)
        {
            io_config::addProps(io_config::GEN_RHS, props);
        }

        // reading solution approximation is not available now through C API
        int block_convert;
        resources->getResourcesConfig()->getParameter<int>("block_convert", block_convert, "default", solver_scope);
        AMG_Configuration t_amgx_cfg;
        AMG_Config *amgx_cfg = t_amgx_cfg.getConfigObject();
        rc = AMGX_OK;
        rc0 = AMGX_RC_OK;

        try
        {
            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    rc0 = read_system<CASE>(mtx, rhs, sol, filename, props, block_convert, *amgx_cfg, read_error); \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources);
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        AMGX_CHECK_API_ERROR(read_error, resources);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_write_parameters_description(char *filename, AMGX_GET_PARAMS_DESC_FLAG mode)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_write_parameters_description " );

        switch (mode)
        {
            case AMGX_GET_PARAMS_DESC_JSON_TO_FILE:
                // handles all exceptions inside
                AMGX_CHECK_API_ERROR(AMG_Config::write_parameters_description_json(filename), NULL);
                break;

            default:
                AMGX_CHECK_API_ERROR(AMGX_ERR_NOT_IMPLEMENTED, NULL);
        }

        return AMGX_RC_OK;
    }


    AMGX_RC AMGX_API AMGX_timer_create(const char *label, unsigned int flags) // create new timer
    {
        nvtxRange nvrf(__func__);

        if (getTimers().createTimer(label, flags))
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
            //return AMGX_RC_BAD_PARAMETERS;
            return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_timer_start(const char *label) // start a timer if it's not started yet
    {
        nvtxRange nvrf(__func__);

        if (getTimers().startTimer(label))
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
            //return AMGX_RC_BAD_PARAMETERS;
            return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_timer_elapsed(const char *label, double *sec) // timer continues to run, just get elapsed time since last start() call
    {
        nvtxRange nvrf(__func__);

        *sec = getTimers().elapsedTimer(label);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_timer_get_total(const char *label, double *sec) // retrieves timer's accumulated value
    {
        nvtxRange nvrf(__func__);

        *sec = getTimers().getTotalTime(label);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_timer_stop(const char *label, double *sec) // timer stops, get time since last start() call
    {
        nvtxRange nvrf(__func__);

        *sec = getTimers().stopTimer(label);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_read_geometry( const char *fname, double **geo_x, double **geo_y, double **geo_z, int *dim, int *numrows)
    {
        nvtxRange nvrf(__func__);

        printf("Reading geometry from file: '%s'\n", fname);
        FILE *fin = fopen(fname, "r");

        if (!fin)
        {
            printf("Error opening file '%s'\n", fname);
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
            //exit(1);
        }

        int n, dimension;

        if (2 != fscanf(fin, "%d %d\n", &n, &dimension))
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
            //exit(1);
            //errAndExit("Bad format\n");
            *geo_x = (double *)get_c_arr_mem_manager().allocate(n * sizeof(double));

        *geo_y = (double *)get_c_arr_mem_manager().allocate(n * sizeof(double));

        if (dimension == 3)
        {
            *geo_y = (double *)get_c_arr_mem_manager().allocate(n * sizeof(double));

            for (int i = 0; i < n; i ++)
                if (3 != fscanf(fin, "%lf %lf %lf\n", *geo_x + i, *geo_y + i, *geo_z + i))
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
                    //exit(1);
                    //errAndExit("Bad format\n");
                }
        else if (dimension == 2)
        {
            for (int i = 0; i < n; i ++)
                if ( 2 != fscanf(fin, "%lf %lf\n", *geo_x + i, *geo_y + i))
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
                    //exit(1);
                    //errAndExit("Bad format\n");
                }

        *dim = dimension;
        *numrows = n;
        return AMGX_RC_OK;
    }


    AMGX_RC AMGX_API AMGX_read_coloring( const char *fname, int **row_coloring, int *colored_rows, int *num_colors)
    {
        nvtxRange nvrf(__func__);

        printf("Reading coloring from file: '%s'\n", fname);
        FILE *fin = fopen(fname, "r");

        if (!fin)
        {
            printf("Error opening file '%s'\n", fname);
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
        }

        int n, colors_num;

        if (2 != fscanf(fin, "%d %d\n", &n, &colors_num))
            AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
            //errAndExit("Bad format\n");
            *row_coloring = (int *)get_c_arr_mem_manager().allocate(n * sizeof(int));

        for (int i = 0; i < n; i ++)
            if ( 1 != fscanf(fin, "%d\n", *row_coloring + i))
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL)
                //errAndExit("Bad format\n");
                *colored_rows = n;

        *num_colors = colors_num;
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_API AMGX_read_system_with_cfg(AMGX_matrix_handle mtx, AMGX_vector_handle rhs, AMGX_vector_handle sol, const char *filename, const AMGX_config_handle cfg_h)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_read_system " );
        Resources *resources = NULL;
        AMGX_Mode mode = AMGX_unset;
        unsigned int props = io_config::NONE;
        AMGX_ERROR read_error = AMGX_OK;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc0 = AMGX_RC_OK;

        try
        {
            rc0 = read_system_preamble(mtx, rhs, sol, resources, mode, props, true);

            if (rc0 != AMGX_RC_OK)
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, resources);    //return AMGX_RC_BAD_PARAMETERS;
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        std::string solver_value, solver_scope;
        resources->getResourcesConfig()->getParameter<std::string>("solver", solver_value, "default", solver_scope);
        int rhs_from_a;
        resources->getResourcesConfig()->getParameter<int>("rhs_from_a", rhs_from_a, "default", solver_scope);

        if (rhs_from_a == 1)
        {
            io_config::addProps(io_config::GEN_RHS, props);
        }

        // reading solution approximation is not available now through C API
        int block_convert;
        resources->getResourcesConfig()->getParameter<int>("block_convert", block_convert, "default", solver_scope);
        ConfigW cfg(cfg_h);
        AMG_Config *amgx_cfg = cfg.wrapped().get()->getConfigObject();
        rc = AMGX_OK;
        rc0 = AMGX_RC_OK;

        try
        {
            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: \
    rc0 = read_system<CASE>(mtx, rhs, sol, filename, props, block_convert, *amgx_cfg, read_error); \
    break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources);
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        AMGX_CHECK_API_ERROR(read_error, resources);
        return AMGX_RC_OK;
    }

    int AMGX_Debug_get_resource_count(AMGX_resources_handle rsc)
    {
        return ((ResourceW *)rsc)->wrapped().use_count();
    }

}//extern "C"
