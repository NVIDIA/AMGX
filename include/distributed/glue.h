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

#include <util.h>

#include <amg.h>
#include <basic_types.h>
#include <types.h>
#include <norm.h>
#include <matrix_distribution.h>

#include <iostream>
#include <iomanip>
#include <blas.h>
#include <multiply.h>

#include <amg_level.h>
#include <amgx_c.h>

#include <misc.h>
#include <string>
#include <cassert>
#include <csr_multiply.h>
#include <memory_info.h>

#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust_wrapper.h>

#define COARSE_CLA_CONSO 0 // used enable / disable coarse level consolidation (used in cycles files)

namespace amgx
{

/**********************************************************
 * Glue ( Consolidation)
 *********************************************************/
#ifdef AMGX_WITH_MPI

//------------------------
//---------Matrix-------------
//------------------------
template<class TConfig>
void compute_glue_info(Matrix<TConfig> &A)
{
    // Fill distributed manager fields for consolidation
    // Example
    // destination_part = [0 0 0 0 4 4 4 4 8 8 8 8] (input from manager->computeDestinationPartitions)
    // num_parts_to_consolidate = 4 for partitions 0,4,8 - (0 otherwise)
    // parts_to_consolidate (rank 0)[0 1 2 3] (rank 4)[4 5 6 7] (rank 8)[8 9 10 11]
    //coarse_part_to_fine_part = [0 4 8] num_coarse_partitions = 3
    //fine_part_to_coarse_part = [0 0 0 0 1 1 1 1 2 2 2 2]
    //ConsolidationArrayOffsets constains the offset of the nnz of each partitions in row pointer fashion : 0, pat1.NNZ, pat1.NNZ+part2.NNZ ... NNZ
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef typename ivec_value_type_h::VecPrec VecInt_t;

    bool is_root_partition = false;
    int num_parts_to_consolidate = 0;
    int num_parts = A.manager->getComms()->get_num_partitions();
    int my_id = A.manager->global_id();
    Vector<ivec_value_type_h> parts_to_consolidate;
    Vector<ivec_value_type_h> dest_partitions = A.manager->getDestinationPartitions();

    // compute is_root_partition and num_parts_to_consolidate
    for (int i = 0; i < num_parts; i++)
    {
        if (dest_partitions[i] == my_id)
        {
            is_root_partition = true;
            num_parts_to_consolidate++;
        }
    }

    parts_to_consolidate.resize(num_parts_to_consolidate);
    // parts_to_consolidate
    int count = 0;

    for (int i = 0; i < num_parts; i++)
    {
        if (dest_partitions[i] == my_id)
        {
            parts_to_consolidate[count] = i;
            count++;
        }
    }

    A.manager->setIsRootPartition(is_root_partition);
    A.manager->setNumPartsToConsolidate(num_parts_to_consolidate);
    A.manager->setPartsToConsolidate(parts_to_consolidate);
    // We don't really use the following in the latest version of the glue path but they are useful information
    // coarse_to_fine_part, fine_to_coarse_part
    Vector<ivec_value_type_h> coarse_to_fine_part, fine_to_coarse_part(num_parts);
    coarse_to_fine_part = dest_partitions;
    amgx::thrust::sort(coarse_to_fine_part.begin(), coarse_to_fine_part.end());
    cudaCheckError();
    coarse_to_fine_part.erase(amgx::thrust::unique(coarse_to_fine_part.begin(), coarse_to_fine_part.end()), coarse_to_fine_part.end());
    cudaCheckError();
    amgx::thrust::lower_bound(coarse_to_fine_part.begin(), coarse_to_fine_part.end(), dest_partitions.begin(), dest_partitions.end(), fine_to_coarse_part.begin());
    cudaCheckError();
    A.manager->setCoarseToFine(coarse_to_fine_part);
    A.manager->setFineToCoarse(fine_to_coarse_part);
    Vector<ivec_value_type_h> consolidationArrayOffsets;
    consolidationArrayOffsets.resize(num_parts);
}

template<class TConfig>
MPI_Comm compute_glue_matrices_communicator(Matrix<TConfig> &A)
{
    // Create temporary communicators for each consilidated matrix
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (A.manager->getDestinationPartitions().size() != 0)
    {
        int color = A.manager->getDestinationPartitions()[rank];
        MPI_Comm new_comm;
        // Split a communicator into multiple, non-overlapping communicators by color(each destiation partition has its color)
        // 1. Use MPI_Allgather to get the color and key from each process
        // 2. Count the number of processes with the same color; create a
        //    communicator with that many processes.  If this process has
        //    MPI_UNDEFINED as the color, create a process with a single member.
        // 3. Use key to order the ranks
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
        return new_comm;
    }
    else
    {
        FatalError("NO DESTINATION PARTIONS", AMGX_ERR_CORE);
    }
}

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

template<class TConfig>
int create_part_offsets(MPI_Comm &mpicm, Matrix<TConfig> &nv_mtx)
{
    /* WARNING: Notice that part_offsets_h & part_offsets have type int64_t.
                Therefore we need to use MPI_INT64_T (or MPI_LONG_LONG) in MPI_Allgather.
                Also, we need the send & recv buffers to be of the same type, therefore
                we will create a temporary variable n64 of the correct type below. */
    //create TConfig64, which is the same as TConfig, but with index type being int64_t
    typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type TConfig64;
    typedef typename TConfig64::VecPrec t_VecPrec; //t_VecPrec = int64_t
    int n, offset, mpist;
    int nranks = 0; //nv_mtx.manager->get_num_partitions();

    if (nv_mtx.manager != NULL)
    {
        //some initializations
        nv_mtx.getOffsetAndSizeForView(OWNED, &offset, &n);
        MPI_Comm_size(mpicm, &nranks);
        nv_mtx.manager->part_offsets_h.resize(nranks + 1);
        //gather the number of rows per partition on the host (on all ranks)
        t_VecPrec n64 = n;
        nv_mtx.manager->part_offsets_h[0] = 0; //first element is zero (the # of rows is gathered afterwards)

        if (typeid(t_VecPrec) == typeid(int64_t))
        {
            mpist = MPI_Allgather(&n64, 1, MPI_INT64_T, nv_mtx.manager->part_offsets_h.raw() + 1, 1, MPI_INT64_T, mpicm);
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
        amgx::thrust::inclusive_scan(nv_mtx.manager->part_offsets_h.begin(), nv_mtx.manager->part_offsets_h.end(), nv_mtx.manager->part_offsets_h.begin());
        //create the corresponding array on device (this is important)
        nv_mtx.manager->part_offsets.resize(nranks + 1);
        amgx::thrust::copy(nv_mtx.manager->part_offsets_h.begin(), nv_mtx.manager->part_offsets_h.end(), nv_mtx.manager->part_offsets.begin());
    }

    return 0;
}

template<class TConfig>
int glue_matrices(Matrix<TConfig> &nv_mtx, MPI_Comm &nv_mtx_com, MPI_Comm &temp_com)
{
    typedef typename TConfig::IndPrec t_IndPrec;
    typedef typename TConfig::MatPrec t_MatPrec;
    int n, nnz, offset, l, k = 0, i;
    int start, end, shift;
    int mpist, root = 0;
    //MPI call parameters
    t_IndPrec *rc_ptr, *di_ptr;
    t_IndPrec *hli_ptr, *hgi_ptr, *hgr_ptr, *i_ptr, *r_ptr;
    t_MatPrec *hlv_ptr, *hgv_ptr, *v_ptr;
    amgx::thrust::host_vector<t_IndPrec> rc;
    amgx::thrust::host_vector<t_IndPrec> di;
    //unpacked local matrix on the device and host
    device_vector_alloc<t_IndPrec> Bp;
    device_vector_alloc<t_IndPrec> Bi;
    device_vector_alloc<t_MatPrec> Bv;
    amgx::thrust::host_vector<t_IndPrec> hBp;
    amgx::thrust::host_vector<t_IndPrec> hBi;
    amgx::thrust::host_vector<t_MatPrec> hBv;
    //Consolidated matrices on the host
    amgx::thrust::host_vector<t_IndPrec> hAp;
    amgx::thrust::host_vector<t_IndPrec> hAi;
    amgx::thrust::host_vector<t_MatPrec> hAv;
    //Consolidated matrices on the device
    device_vector_alloc<t_IndPrec> Ap;
    device_vector_alloc<t_IndPrec> Ai;
    device_vector_alloc<t_MatPrec> Av;
    //WARNING: this routine currently supports matrix only with block size =1 (it can be generalized in the future, though)
    //initialize the defaults
    mpist = MPI_SUCCESS;

    if (nv_mtx.manager != NULL)
    {
        //int rank = nv_mtx.manager->global_id();
        nv_mtx.getOffsetAndSizeForView(OWNED, &offset, &n);
        nv_mtx.getNnzForView(OWNED, &nnz);

        if (nv_mtx.manager->part_offsets_h.size() == 0 || nv_mtx.manager->part_offsets.size() == 0)   // create part_offsets_h & part_offsets
        {
            create_part_offsets(nv_mtx_com, nv_mtx);  // (if needed for aggregation path)
        }

        Bp.resize(n + 1);
        Bi.resize(nnz);
        Bv.resize(nnz);
        hBp.resize(n + 1);
        hBi.resize(nnz);
        hBv.resize(nnz);
        //--- unpack the matrix ---
        nv_mtx.manager->unpack_partition(amgx::thrust::raw_pointer_cast(Bp.data()),
                                         amgx::thrust::raw_pointer_cast(Bi.data()),
                                         amgx::thrust::raw_pointer_cast(Bv.data()));
        cudaCheckError();
        //copy to host (should be able to optimize this out later on)
        hBp = Bp;
        hBi = Bi;
        hBv = Bv;
        cudaCheckError();

        // --- Glue matrices ---

        // construct global row pointers
        // compute recvcounts and displacements for MPI_Gatherv
        if (nv_mtx.manager->isRootPartition())
        {
            l = nv_mtx.manager->getNumPartsToConsolidate();      // number of partitions
            rc.resize(l);
            di.resize(l);

            //compute recvcounts and displacements for MPI_Gatherv
            for (i = 0; i < l; i++)
            {
                start = nv_mtx.manager->part_offsets_h[nv_mtx.manager->getPartsToConsolidate()[i]];
                end   = nv_mtx.manager->part_offsets_h[nv_mtx.manager->getPartsToConsolidate()[i] + 1];
                rc[i] = end - start;
                di[i] = k + 1;
                k += rc[i];
            }

            hAp.resize(k + 1); // extra +1 is needed because row_offsets have one extra element at the end
        }

        cudaCheckError();
        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        rc_ptr = amgx::thrust::raw_pointer_cast(rc.data());
        di_ptr = amgx::thrust::raw_pointer_cast(di.data());
        hli_ptr = amgx::thrust::raw_pointer_cast(hBp.data());
        hgr_ptr = amgx::thrust::raw_pointer_cast(hAp.data());
        cudaCheckError();

        //gather (on the host)
        if (typeid(t_IndPrec) == typeid(int))
        {
            mpist = MPI_Gatherv(hli_ptr + 1, n, MPI_INT,  hgr_ptr, rc_ptr, di_ptr, MPI_INT, root, temp_com);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        // Adjust row pointers, construct global column indices and values (recvcounts and displacements were computed above)
        if (nv_mtx.manager->isRootPartition())
        {
            //adjust global row pointers and setup the recvcounts & displacements for subsequent MPI calls
            for (i = 0; i < l; i++)
            {
                start = di[i] - 1;
                end   = di[i] + rc[i] - 1;
                shift = hAp[start];
                thrust_wrapper::transform<TConfig::memSpace>(hAp.begin() + start + 1, hAp.begin() + end + 1, hAp.begin() + start + 1, add_constant_op<t_IndPrec>(shift));
                cudaCheckError();
                di[i] = shift;
                rc[i] = hAp[end] - hAp[start];
            }

            //some allocations/resizing
            hAi.resize(hAp[k]);
            hAv.resize(hAp[k]);
        }

        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        rc_ptr = amgx::thrust::raw_pointer_cast(rc.data());
        di_ptr = amgx::thrust::raw_pointer_cast(di.data());
        hli_ptr = amgx::thrust::raw_pointer_cast(hBi.data());
        hgi_ptr = amgx::thrust::raw_pointer_cast(hAi.data());
        hlv_ptr = amgx::thrust::raw_pointer_cast(hBv.data());
        hgv_ptr = amgx::thrust::raw_pointer_cast(hAv.data());
        cudaCheckError();

        //gather (on the host)
        //columns indices
        if (typeid(t_IndPrec) == typeid(int))
        {
            mpist = MPI_Gatherv(hli_ptr, nnz, MPI_INT,  hgi_ptr, rc_ptr, di_ptr, MPI_INT,  root, temp_com);
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
            mpist = MPI_Gatherv(hlv_ptr, nnz, MPI_FLOAT,  hgv_ptr, rc_ptr, di_ptr, MPI_FLOAT,  root, temp_com);
        }
        else if (typeid(t_MatPrec) == typeid(double))
        {
            mpist = MPI_Gatherv(hlv_ptr, nnz, MPI_DOUBLE, hgv_ptr, rc_ptr, di_ptr, MPI_DOUBLE, root, temp_com);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        // --- Upload matrices ---
        if (nv_mtx.manager->isRootPartition())
        {
            n = hAp.size() - 1;
            nnz = hAi.size();
            Ap.resize(hAp.size());
            Ai.resize(hAi.size());
            Av.resize(hAv.size());
            amgx::thrust::copy(hAp.begin(), hAp.end(), Ap.begin());
            amgx::thrust::copy(hAi.begin(), hAi.end(), Ai.begin());
            amgx::thrust::copy(hAv.begin(), hAv.end(), Av.begin());
            cudaCheckError();
        }
        else
        {
            n = 0;
            nnz  = 0;
            Ap.resize(1); // warning row_ponter size is expected to be n+1.
            Ap.push_back(0);
            Ai.resize(0);
            Av.resize(0);
            cudaCheckError();
        }

        r_ptr = amgx::thrust::raw_pointer_cast(Ap.data());
        i_ptr = amgx::thrust::raw_pointer_cast(Ai.data());
        v_ptr = amgx::thrust::raw_pointer_cast(Av.data());
        cudaCheckError();
        upload_matrix_after_glue(n, nnz, r_ptr, i_ptr, v_ptr, nv_mtx);
    }
    else
    {
        /* ASSUMPTION: when manager has not been allocated you are running on a single rank */
    }

    return 0;
}


template<class TConfig>
int upload_matrix_after_glue(int n, int nnz, int *r_ptr, int *i_ptr, void *v_ptr, Matrix<TConfig> &nv_mtx)
{
    // Using a path similar to AMGX_matrix_upload_all_global
    typedef typename TConfig::IndPrec t_IndPrec;
    typedef typename TConfig::MatPrec t_MatPrec;
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef typename ivec_value_type_h::VecPrec VecInt_t;
    typedef Vector<ivec_value_type> IVector;
    // some parameters
    int block_dimx, block_dimy, num_ranks, n_global, start, end, val;
    t_IndPrec *part_vec_ptr;
    amgx::thrust::host_vector<t_IndPrec> pv;
    // set parameters
    nv_mtx.setView(ALL); // not sure about this
    n_global = nv_mtx.manager->part_offsets_h.back();
    block_dimx = nv_mtx.get_block_dimx();
    block_dimy = nv_mtx.get_block_dimy();
    //MPI_Comm* mpi_comm = nv_mtx.getResources()->getMpiComm();
    //MPI_Comm_size(*mpi_comm, &num_ranks);
    num_ranks = nv_mtx.manager->getComms()->get_num_partitions();
    // WARNING We create an artificial partition vectior that matches the new distribution
    // example n = 8, num_ranks(ie. num partitions) = 4 , DestinationPartitions[0,0,2,2], partvec = [0,0,0,0,2,2,2,2]
    // This might be an issue for the finest level if input_partvect !=NULL
    Vector<ivec_value_type_h> dest_partitions = nv_mtx.manager->getDestinationPartitions();

    for (int i = 0; i < num_ranks; i++)
    {
        val = dest_partitions[i];
        start = nv_mtx.manager->part_offsets_h[i];
        end   = nv_mtx.manager->part_offsets_h[i + 1];

        for (int j = 0; j < end - start; j++)
        {
            pv.push_back(val);
        }
    }

    part_vec_ptr = amgx::thrust::raw_pointer_cast(pv.data());
    cudaCheckError();
    // Save some glue info
    bool is_root_partition = nv_mtx.manager->isRootPartition();
    int dest_part = nv_mtx.manager->getMyDestinationPartition();
    int num_parts_to_consolidate = nv_mtx.manager->getNumPartsToConsolidate();
    Vector<ivec_value_type_h> parts_to_consolidate = nv_mtx.manager->getPartsToConsolidate();
    std::vector<int> cao;

    for (int i = 0; i < nv_mtx.manager->part_offsets_h.size(); i++)
    {
        cao.push_back(nv_mtx.manager->part_offsets_h[i]);
    }

    // WARNING
    // renumbering contains the inverse permutation to unreorder an amgx vector
    // inverse_renumbering contains the permutaion to reorder an amgx vector
    Vector<ivec_value_type> ir = nv_mtx.manager->inverse_renumbering;
    Vector<ivec_value_type> r = nv_mtx.manager->renumbering;
    // We need that to exchange the halo of unglued vectors (in coarse level consolidation)
    Vector<ivec_value_type_h> nei = nv_mtx.manager->neighbors;  // just neighbors before glue
    std::vector<std::vector<VecInt_t> > b2lr = nv_mtx.manager->getB2Lrings(); //list of boundary nodes to export to other partitions.
    Vector<ivec_value_type_h> ho = nv_mtx.manager->halo_offsets;
    std::vector<IVector > b2lm = nv_mtx.manager->getB2Lmaps();
    cudaCheckError();

    // Create a fresh distributed manager
    if (nv_mtx.manager != NULL )
    {
        delete nv_mtx.manager;
    }

    nv_mtx.manager = new DistributedManager<TConfig>(nv_mtx);
    nv_mtx.set_initialized(0);
    nv_mtx.delProps(DIAG);
    // Load distributed matrix
    MatrixDistribution mdist;
    mdist.setPartitionVec(part_vec_ptr);
    nv_mtx.manager->loadDistributedMatrix(n, nnz, block_dimx, block_dimy, r_ptr, i_ptr, (t_MatPrec *) v_ptr, num_ranks, n_global, NULL, mdist);
    // Create B2L_maps for comm
    nv_mtx.manager->renumberMatrixOneRing();
    // WARNING WE SHOULD GET THE NUMBER OF RINGS AND DO THE FOLLOWING ONLY IF THERE ARE 2 RINGS
    // Exchange 1 ring halo rows (for d2 interp)
    // if (num_import_rings == 2)
    nv_mtx.manager->createOneRingHaloRows();
    nv_mtx.manager->getComms()->set_neighbors(nv_mtx.manager->num_neighbors());
    nv_mtx.setView(OWNED);
    nv_mtx.set_initialized(1);
    cudaCheckError();
    // restore some glue info to consolidate the vectors in the future
    nv_mtx.manager->setDestinationPartitions(dest_partitions);
    nv_mtx.manager->setIsRootPartition(is_root_partition);
    nv_mtx.manager->setNumPartsToConsolidate(num_parts_to_consolidate);
    nv_mtx.manager->setPartsToConsolidate(parts_to_consolidate);
    nv_mtx.manager->setIsGlued(true);
    nv_mtx.manager->setMyDestinationPartition(dest_part);
    nv_mtx.manager->setConsolidationArrayOffsets(cao); // partions_offest before consolidation
    // set fine level data structures, this is used to match the former distribution when we upload / download from the API

    if (nv_mtx.amg_level_index == 0)
    {
        // just small copies inside fineLevelUpdate.
        nv_mtx.manager->fineLevelUpdate();
    }

    nv_mtx.manager->renumbering_before_glue = r;
    nv_mtx.manager->inverse_renumbering_before_glue = ir;
    nv_mtx.manager->neighbors_before_glue = nei;
    nv_mtx.manager->halo_offsets_before_glue = ho;
    nv_mtx.manager->B2L_rings_before_glue = b2lr;
    nv_mtx.manager->B2L_maps_before_glue = b2lm;
    cudaCheckError();
    return 0;
}

template<class TConfig>
int glue_vector(Matrix<TConfig> &nv_mtx, MPI_Comm &A_comm, Vector<TConfig> &nv_vec, MPI_Comm &temp_com)
{
    // glu vecots based on dest_partitions (which contains partitions should be merged together)
    typedef typename TConfig::IndPrec t_IndPrec;
    typedef typename TConfig::VecPrec t_VecPrec;
    int n, l, mpist, start, end, k = 0, root = 0, rank = 0;
    //MPI call parameters
    t_IndPrec *rc_ptr, *di_ptr;
    t_VecPrec *hv_ptr, *hg_ptr, *v_ptr;
    amgx::thrust::host_vector<t_IndPrec> rc;
    amgx::thrust::host_vector<t_IndPrec> di;
    //unreordered local vector on the host
    amgx::thrust::host_vector<t_VecPrec> hv;
    //constructed global vector on the host
    amgx::thrust::host_vector<t_VecPrec> hg;
    //constructed global vector on the device
    device_vector_alloc<t_VecPrec> v;
    //WARNING: this routine currently supports vectors only with block size =1 (it can be generalized in the future, though)
    //initialize the defaults
    mpist = MPI_SUCCESS;

    if (nv_mtx.manager != NULL)
    {
        // some initializations
        rank = nv_mtx.manager->global_id();

        if (nv_mtx.manager->getComms() != NULL)
        {
            nv_mtx.manager->getComms()->get_mpi_comm();
        }

        n = nv_mtx.manager->getConsolidationArrayOffsets()[rank + 1] - nv_mtx.manager->getConsolidationArrayOffsets()[rank];

        if (nv_mtx.manager->getConsolidationArrayOffsets().size() == 0)
        {
            std::cout << "ERROR part_offsets in glue path" << std::endl;
        }

        l = nv_mtx.manager->getNumPartsToConsolidate();      // number of partitions
        //some allocations/resizing
        hv.resize(nv_mtx.manager->renumbering_before_glue.size());                         // host copy of nv_vec

        if (nv_mtx.manager->isRootPartition())
        {
            // This works with neighbours only
            hg.resize(nv_mtx.manager->getConsolidationArrayOffsets()[rank + l] - nv_mtx.manager->getConsolidationArrayOffsets()[rank]); // host copy of cvec
            rc.resize(l);
            di.resize(l);
        }

        cudaCheckError();
        //--- unreorder the vector back (just like you did with the matrix, but only need to undo the interior-boundary reordering, because others do not apply) ---
        // unreorder and copy the vector
        // WARNING
        // renumbering contains the inverse permutation to unreorder an amgx vector
        // inverse_renumbering contains the permutaion to reorder an amgx vector
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(nv_vec.begin(), nv_mtx.manager->renumbering_before_glue.begin()  ),
                     amgx::thrust::make_permutation_iterator(nv_vec.begin(), nv_mtx.manager->renumbering_before_glue.begin() +  nv_mtx.manager->renumbering_before_glue.size()),
                     hv.begin());
        cudaCheckError();
        hv.resize(n);

        // --- construct global vector (rhs/sol) ---
        //compute recvcounts and displacements for MPI_Gatherv
        if (nv_mtx.manager->isRootPartition())
        {
            l = nv_mtx.manager->getNumPartsToConsolidate();      // number of partitions

            //compute recvcounts and displacements for MPI_Gatherv
            for (int i = 0; i < l; i++)
            {
                start = nv_mtx.manager->getConsolidationArrayOffsets()[nv_mtx.manager->getPartsToConsolidate()[i]];
                end   = nv_mtx.manager->getConsolidationArrayOffsets()[nv_mtx.manager->getPartsToConsolidate()[i] + 1];
                rc[i] = end - start;
                di[i] = k;
                k += rc[i];
            }
        }

        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        rc_ptr = amgx::thrust::raw_pointer_cast(rc.data());
        di_ptr = amgx::thrust::raw_pointer_cast(di.data());
        hv_ptr = amgx::thrust::raw_pointer_cast(hv.data());
        hg_ptr = amgx::thrust::raw_pointer_cast(hg.data());
        cudaCheckError();

        //gather (on the host)
        if      (typeid(t_VecPrec) == typeid(float))
        {
            mpist = MPI_Gatherv(hv_ptr, n, MPI_FLOAT,  hg_ptr, rc_ptr, di_ptr, MPI_FLOAT,  root, temp_com);
        }
        else if (typeid(t_VecPrec) == typeid(double))
        {
            mpist = MPI_Gatherv(hv_ptr, n, MPI_DOUBLE, hg_ptr, rc_ptr, di_ptr, MPI_DOUBLE, root, temp_com);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        // clean
        nv_vec.in_transfer = IDLE;

        //nv_vec.dirtybit = 0;
        if (nv_vec.buffer != NULL)
        {
            delete nv_vec.buffer;
            nv_vec.buffer = NULL;
            nv_vec.buffer_size = 0;
        }

        if (nv_vec.linear_buffers_size != 0)
        {
            amgx::memory::cudaFreeHost(&(nv_vec.linear_buffers[0]));
            nv_vec.linear_buffers_size = 0;
        }

        if (nv_vec.explicit_host_buffer)
        {
            amgx::memory::cudaFreeHost(nv_vec.explicit_host_buffer);
            nv_vec.explicit_host_buffer = NULL;
            nv_vec.explicit_buffer_size = 0;
            cudaEventDestroy(nv_vec.mpi_event);
        }

        // resize
        if (nv_mtx.manager->isRootPartition())
        {
            n = hg.size();
            v.resize(hg.size());
            amgx::thrust::copy(hg.begin(), hg.end(), v.begin());
            cudaCheckError();
        }
        else
        {
            n = 0;
            v.resize(0);
            cudaCheckError();
        }

        // upload
        v_ptr = amgx::thrust::raw_pointer_cast(v.data());
        cudaCheckError();
        upload_vector_after_glue(n, v_ptr, nv_vec, nv_mtx);
    }
    else
    {
        // ASSUMPTION: when manager has not been allocated you are running on a single rank
    }

    return 0;
}


template<class TConfig>
int upload_vector_after_glue(int n, void *v_ptr, Vector<TConfig> &nv_vec, Matrix<TConfig> &nv_mtx)
{
    typedef typename TConfig::VecPrec t_VecPrec;
    // vector bind
    nv_vec.unsetManager();
    cudaCheckError();

    if (nv_mtx.manager != NULL)
    {
        nv_vec.setManager(*(nv_mtx.manager));
    }

    cudaCheckError();

    if (nv_vec.is_transformed())
    {
        nv_vec.unset_transformed();
    }

    nv_vec.set_block_dimx(1);
    nv_vec.set_block_dimy(nv_mtx.get_block_dimy());
    // the dirtybit has to be set to one here in order to have correct results to ensure an exachange halo before the solve
    // this is particulary important when the number of consolidated partitions is greater than 1 on large matrices such as drivaer9M
    nv_vec.dirtybit = 1;

    if (nv_mtx.manager != NULL)
    {
        nv_vec.getManager()->transformAndUploadVector(nv_vec, (t_VecPrec *)v_ptr, n, nv_vec.get_block_dimy());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

template<class TConfig>
int unglue_vector(Matrix<TConfig> &nv_mtx, MPI_Comm &A_comm, Vector<TConfig> &nv_vec, MPI_Comm &temp_com, Vector<TConfig> &nv_vec_unglued)
{
    // glue vecots based on dest_partitions (which contains partitions should be merged together)
    typedef typename TConfig::IndPrec t_IndPrec;
    typedef typename TConfig::VecPrec t_VecPrec;
    int n_loc, l, mpist, start, end, k = 0, root = 0, rank = 0;
    //MPI call parameters
    t_IndPrec *sc_ptr, *di_ptr;
    t_VecPrec *hv_ptr, *hg_ptr;
    amgx::thrust::host_vector<t_IndPrec> sc;
    amgx::thrust::host_vector<t_IndPrec> di;
    //unreordered local vector on the host
    amgx::thrust::host_vector<t_VecPrec> hv;
    //constructed global vector on the host
    amgx::thrust::host_vector<t_VecPrec> hg;
    //constructed global vector on the device
    device_vector_alloc<t_VecPrec> v;
    //WARNING: this routine currently supports vectors only with block size =1 (it can be generalized in the future, though)
    //initialize the defaults
    mpist = MPI_SUCCESS;

    if (nv_mtx.manager != NULL)
    {
        // some initializations
        rank = nv_mtx.manager->global_id();

        if (nv_mtx.manager->getComms() != NULL)
        {
            nv_mtx.manager->getComms()->get_mpi_comm();
        }

        n_loc = nv_mtx.manager->getConsolidationArrayOffsets()[rank + 1] - nv_mtx.manager->getConsolidationArrayOffsets()[rank];

        if (nv_mtx.manager->getConsolidationArrayOffsets().size() == 0)
        {
            printf("ERROR part_offsets\n");
        }

        l = nv_mtx.manager->getNumPartsToConsolidate();      // number of partitions
        //some allocations/resizing
        hv.resize(n_loc);                         // host copy

        if (nv_mtx.manager->isRootPartition())
        {
            hg.resize(nv_vec.size()); // host copy of cvec
        }

        sc.resize(l);
        di.resize(l);
        cudaCheckError();
        // Exchange_halo before unreordering
        // Do we need that?
        nv_mtx.manager->exchange_halo(nv_vec, nv_vec.tag);

        // unreorder the vector
        if (nv_mtx.manager->isRootPartition())
        {
            // WARNING
            // renumbering contains the inverse permutation to unreorder an amgx vector
            // inverse_renumbering contains the permutaion to reorder an amgx vector
            amgx::thrust::copy(amgx::thrust::make_permutation_iterator(nv_vec.begin(), nv_mtx.manager->renumbering.begin()  ),
                         amgx::thrust::make_permutation_iterator(nv_vec.begin(), nv_mtx.manager->renumbering.begin() + nv_mtx.manager->renumbering.size()),
                         hg.begin());
            cudaCheckError();
            hg.resize(nv_mtx.manager->getConsolidationArrayOffsets()[rank + l] - nv_mtx.manager->getConsolidationArrayOffsets()[rank]);
        }

        // --- construct local vector (sol) ---
        //compute sendcounts and displacements for MPI_Gatherv
        for (int i = 0; i < l; i++)
        {
            start = nv_mtx.manager->getConsolidationArrayOffsets()[nv_mtx.manager->getPartsToConsolidate()[i]];
            end   = nv_mtx.manager->getConsolidationArrayOffsets()[nv_mtx.manager->getPartsToConsolidate()[i] + 1];
            sc[i] = end - start;
            di[i] = k;
            k += sc[i];
        }

        //alias raw pointers to thrust vector data (see thrust example unwrap_pointer for details)
        sc_ptr = amgx::thrust::raw_pointer_cast(sc.data());
        di_ptr = amgx::thrust::raw_pointer_cast(di.data());
        hv_ptr = amgx::thrust::raw_pointer_cast(hv.data());
        hg_ptr = amgx::thrust::raw_pointer_cast(hg.data());
        cudaCheckError();

        // Scatter (on the host)
        if      (typeid(t_VecPrec) == typeid(float))
        {
            mpist = MPI_Scatterv(hg_ptr, sc_ptr, di_ptr, MPI_FLOAT,  hv_ptr, n_loc, MPI_FLOAT,  root, temp_com);
        }
        else if (typeid(t_VecPrec) == typeid(double))
        {
            mpist = MPI_Scatterv(hg_ptr, sc_ptr, di_ptr, MPI_DOUBLE, hv_ptr, n_loc, MPI_DOUBLE, root, temp_com);
        }
        else
        {
            FatalError("MPI_Gatherv of the vector has failed - incorrect vector data type", AMGX_ERR_CORE);
        }

        if (mpist != MPI_SUCCESS)
        {
            FatalError("MPI_Gatherv of the vector has failed - detected incorrect MPI return code", AMGX_ERR_CORE);
        }

        // --- Manual upload ---
        // Cleaning
        nv_vec_unglued.in_transfer = IDLE;

        if (nv_vec_unglued.buffer != NULL)
        {
            delete nv_vec_unglued.buffer;
            nv_vec_unglued.buffer = NULL;
            nv_vec_unglued.buffer_size = 0;
        }

        if (nv_vec_unglued.linear_buffers_size != 0)
        {
            amgx::memory::cudaFreeHost(&(nv_vec_unglued.linear_buffers[0]));
            nv_vec_unglued.linear_buffers_size = 0;
        }

        if (nv_vec_unglued.explicit_host_buffer)
        {
            amgx::memory::cudaFreeHost(nv_vec_unglued.explicit_host_buffer);
            nv_vec_unglued.explicit_host_buffer = NULL;
            nv_vec_unglued.explicit_buffer_size = 0;
            cudaEventDestroy(nv_vec_unglued.mpi_event);
        }

        // We should avoid copies between nv_vec and hv here
        nv_vec_unglued.resize( nv_mtx.manager->inverse_renumbering_before_glue.size());
        thrust_wrapper::fill( nv_vec_unglued.begin(), nv_vec_unglued.end(), 0.0 );
        amgx::thrust::copy(hv.begin(), hv.end(), nv_vec_unglued.begin());
        hv.resize( nv_mtx.manager->inverse_renumbering_before_glue.size());
        cudaCheckError();
        //  Manual reordering
        //  Upload_vector_after_glue is not going to work because matrix managers has been modified during glued matrices, and don't match the new, glued, topology.
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(nv_vec_unglued.begin(), nv_mtx.manager->inverse_renumbering_before_glue.begin()  ),
                     amgx::thrust::make_permutation_iterator(nv_vec_unglued.begin(), nv_mtx.manager->inverse_renumbering_before_glue.begin() + nv_mtx.manager->inverse_renumbering_before_glue.size()),
                     hv.begin());
        cudaCheckError();
        thrust_wrapper::fill( nv_vec_unglued.begin(), nv_vec_unglued.end(), 0.0 );
        amgx::thrust::copy(hv.begin(), hv.end(), nv_vec_unglued.begin());
    }
    else
    {
        // ASSUMPTION: when manager has not been allocated you are running on a single rank
        printf("Glue was called on a single rank\n");
    }

    return 0;
}

#if 0

// The folowing code is to perform an exhange halo using data that doesn't matches the topology of the matrix stored in its distributed manager
// We use instead other containers stored in the distributed manager. They are suffixed by "_before_glue"
// This allows to exchange vector halo between unglued vectors from glued matrices

template <class TConfig>
void exchange_halo_after_unglue(const Matrix<TConfig> &A, Vector<TConfig>  &data, int tag, int num_ring = 1)
{
    setup_after_unglue(data, A, num_ring);  //set pointers to buffer
    gather_B2L_after_unglue(A, data, num_ring);             //write values to buffer
    exchange_halo_after_unglue(data, A, num_ring); //exchange buffers
    //scatter_L2H(data);            //NULL op
}
/*
template <class TConfig>
void CommsMPIHostBufferStream<T_Config>::setup(DVector &b, const Matrix<TConfig> &m, int num_rings) { do_setup_after_unglue((b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings) {  do_exchange_halo_after_unglue((b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup_after_unglue((b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings) {  do_exchange_halo_after_unglue((b, m, num_rings);}
*/
template <class TConfig>
void setup_after_unglue(Vector<TConfig> &b, const Matrix<TConfig> &m, int num_rings)
{
    /*
       amgx::thrust::copy( m.manager->neighbors_before_glue.begin(),  m.manager->neighbors_before_glue.end(), std::ostream_iterator<int64_t>(std::cout, " "));
       amgx::thrust::copy( m.manager->halo_offsets_before_glue.begin(),  m.manager->halo_offsets_before_glue.end(), std::ostream_iterator<int64_t>(std::cout, " "));
    for (int i = 0; i < m.manager->B2L_rings_before_glue.size(); ++i)
    {
    amgx::thrust::copy( m.manager->B2L_rings_before_glue[i].begin(),  m.manager->B2L_rings_before_glue[i].end(), std::ostream_iterator<int64_t>(std::cout, " "));
    }

    for (int i = 0; i < m.manager->B2L_maps_before_glue.size(); ++i)
    {
    amgx::thrust::copy( m.manager->B2L_maps_before_glue[i].begin(),  m.manager->B2L_maps_before_glue[i].end(), std::ostream_iterator<int64_t>(std::cout, " "));
    }
    */
    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("MPI Comms module no implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
#ifdef AMGX_WITH_MPI
        int bsize = b.get_block_size();
        int num_cols = b.get_num_cols();

        if (bsize != 1 && num_cols != 1)
            FatalError("Error: vector cannot have block size and subspace size.",
                       AMGX_ERR_INTERNAL);

        // set num neighbors = size of B2L_rings_before_glue
        // need to do this because comms might have more neighbors than our matrix knows about
        int neighbors = m.manager->B2L_rings_before_glue.size();
        m.manager->getComms()->set_neighbors(m.manager->B2L_rings_before_glue.size());

        if (b.in_transfer & SENDING)
        {
            b.in_transfer = IDLE;
        }

        typedef typename TConfig::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type value_type;
        b.requests.resize(2 * neighbors); //first part is sends second is receives
        b.statuses.resize(2 * neighbors);

        for (int i = 0; i < 2 * neighbors; i++)
        {
            b.requests[i] = MPI_REQUEST_NULL;
        }

        int total = 0;

        for (int i = 0; i < neighbors; i++)
        {
            total += m.manager->B2L_rings_before_glue[i][num_rings] * bsize * num_cols;
        }

        b.buffer_size = total;

        if (b.buffer == NULL)
        {
            b.buffer = new Vector<TConfig>(total);
        }
        else
        {
            if (total > b.buffer->size())
            {
                b.buffer->resize(total);
            }
        }

        if (b.linear_buffers_size < neighbors)
        {
            if (b.linear_buffers_size != 0) { amgx::memory::cudaFreeHost(b.linear_buffers); }

            amgx::memory::cudaMallocHost((void **) & (b.linear_buffers), neighbors * sizeof(value_type *));
            b.linear_buffers_size = neighbors;
        }

        cudaCheckError();
        total = 0;
        bool linear_buffers_changed = false;

        for (int i = 0; i < neighbors; i++)
        {
            if (b.linear_buffers[i] != b.buffer->raw() + total)
            {
                linear_buffers_changed = true;
            }

            b.linear_buffers[i] = b.buffer->raw() + total;
            total += m.manager->B2L_rings_before_glue[i][num_rings] * bsize * num_cols;
        }

        // Copy to device
        if (linear_buffers_changed)
        {
            b.linear_buffers_ptrs.resize(neighbors);
            //amgx::thrust::copy(b.linear_buffers.begin(),b.linear_buffers.end(),b.linear_buffers_ptrs.begin());
            cudaMemcpyAsync(amgx::thrust::raw_pointer_cast(&b.linear_buffers_ptrs[0]), &(b.linear_buffers[0]), neighbors * sizeof(value_type *), cudaMemcpyHostToDevice);
            cudaCheckError();
        }

        int size = 0;
        size  = total + (m.manager->halo_offsets_before_glue[num_rings * neighbors] - m.manager->halo_offsets_before_glue[0]) * bsize * num_cols;

        if (size > 0)
        {
            if (b.explicit_host_buffer == NULL)
            {
                b.host_buffer.resize(1);
                cudaEventCreateWithFlags(&b.mpi_event, cudaEventDisableTiming);
                cudaCheckError();
                amgx::memory::cudaMallocHost((void **)&b.explicit_host_buffer, size * sizeof(value_type));
                cudaCheckError();
            }
            else if (size > b.explicit_buffer_size)
            {
                amgx::memory::cudaFreeHost(b.explicit_host_buffer);
                cudaCheckError();
                amgx::memory::cudaMallocHost((void **)&b.explicit_host_buffer, size * sizeof(value_type));
                cudaCheckError();
            }

            cudaCheckError();
            b.explicit_buffer_size = size;
        }

#else
        FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
    }
}
template <class TConfig>
void gather_B2L_after_unglue(const Matrix<TConfig> &m, Vector<TConfig> &b, int num_rings = 1)
{
    if (TConfig::memSpace == AMGX_host)
    {
        if (m.manager->neighbors_before_glue.size() > 0)
        {
            FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
        }
    }
    else
    {
        for (int i = 0; i < m.manager->neighbors_before_glue.size(); i++)
        {
            int size = m.manager->B2L_rings_before_glue[i][num_rings];
            int num_blocks = min(4096, (size + 127) / 128);

            if ( size != 0)
            {
                if (b.get_num_cols() == 1)
                {
                    gatherToBuffer <<< num_blocks, 128>>>(b.raw(), m.manager->B2L_maps_before_glue[i].raw(), b.linear_buffers[i], b.get_block_size(), size);
                }
                else
                {
                    gatherToBufferMultivector <<< num_blocks, 128>>>(b.raw(), m.manager->B2L_maps_before_glue[i].raw(), b.linear_buffers[i], b.get_num_cols(), b.get_lda(), size);
                }

                cudaCheckError();
            }
        }
    }
}
template <class TConfig>
void exchange_halo_after_unglue(Vector<TConfig> &b, const Matrix<TConfig> &m, int num_rings)
{
    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Halo exchanges not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
#ifdef AMGX_WITH_MPI
        typedef typename TConfig::VecPrec VecPrec;
        cudaCheckError();
        int bsize = b.get_block_size();
        int num_cols = b.get_num_cols();
        int offset = 0;
        int neighbors = m.manager->B2L_rings_before_glue.size();
        MPI_Comm mpi_comm = m.manager->getComms()->get_mpi_comm();

        if (b.buffer_size != 0)
        {
            cudaMemcpy(&(b.explicit_host_buffer[0]), b.buffer->raw(), b.buffer_size * sizeof(typename TConfig::VecPrec), cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < neighbors; i++)
        {
            int size = m.manager->B2L_rings_before_glue[i][num_rings] * bsize * num_cols;

            if (size != 0)
            {
                MPI_Isend(&(b.explicit_host_buffer[offset]), size * sizeof(typename TConfig::VecPrec), MPI_BYTE, m.manager->neighbors_before_glue[i], m.manager->global_id(), mpi_comm, &b.requests[i]);
            }
            else
            {
                MPI_Isend(&(b.host_buffer[0]), size * sizeof(typename TConfig::VecPrec), MPI_BYTE, m.manager->neighbors_before_glue[i], m.manager->global_id(), mpi_comm, &b.requests[i]);
            }

            offset += size;
        }

        b.in_transfer = RECEIVING | SENDING;
        offset = 0;

        for (int i = 0; i < neighbors; i++)
        {
            // Count total size to receive from one neighbor
            int size = 0;

            for (int j = 0; j < num_rings; j++)
            {
                size += m.manager->halo_offsets_before_glue[j * neighbors + i + 1] * bsize * num_cols - m.manager->halo_offsets_before_glue[j * neighbors + i] * bsize * num_cols;
            }

            if (size != 0)
            {
                MPI_Irecv(&(b.explicit_host_buffer[b.buffer_size + offset]), size * sizeof(typename TConfig::VecPrec), MPI_BYTE, m.manager->neighbors_before_glue[i], m.manager->neighbors_before_glue[i], mpi_comm, &b.requests[neighbors + i]);
            }
            else
            {
                MPI_Irecv(&(b.host_buffer[0]), size * sizeof(typename TConfig::VecPrec), MPI_BYTE, m.manager->neighbors_before_glue[i], m.manager->neighbors_before_glue[i], mpi_comm, &b.requests[neighbors + i]);
            }

            offset += size;
            int required_size = m.manager->halo_offsets_before_glue[0] * bsize * num_cols + offset;

            if (required_size > b.size())
            {
                // happen because we have 2 ring
                // required_size correspond to "n" in the FULL view of the unconsolidated matrix.
                // In exchange halo this is a fatal error since it should never happen.
                b.resize(required_size);
            }
        }

        MPI_Waitall(2 * neighbors, &b.requests[0], /*&b.statuses[0]*/ MPI_STATUSES_IGNORE); //I only wait to receive data, I can start working before all my buffers were sent
        b.dirtybit = 0;
        b.in_transfer = IDLE;

        // copy on host ring by ring
        if (num_rings == 1)
        {
            if (num_cols == 1)
            {
                if (offset != 0)
                {
                    cudaMemcpy(b.raw() + m.manager->halo_offsets_before_glue[0]*bsize, &(b.explicit_host_buffer[b.buffer_size]), offset * sizeof(typename TConfig::VecPrec), cudaMemcpyHostToDevice);
                }
            }
            else
            {
                int lda = b.get_lda();
                VecPrec *rank_start = &(b.explicit_host_buffer[b.buffer_size]);

                for (int i = 0; i < neighbors; ++i)
                {
                    int halo_size = m.manager->halo_offsets_before_glue[i + 1] - m.manager->halo_offsets_before_glue[i];

                    for (int s = 0; s < num_cols; ++s)
                    {
                        VecPrec *halo_start = b.raw() + lda * s + m.manager->halo_offsets_before_glue[i];
                        VecPrec *received_halo = rank_start + s * halo_size;
                        cudaMemcpy(halo_start, received_halo, halo_size * sizeof(VecPrec), cudaMemcpyHostToDevice);
                    }

                    rank_start += num_cols * halo_size;
                }
            }
        }
        else
        {
            if (num_cols == 1)
            {
                offset = 0;

                // Copy into b, one neighbor at a time, one ring at a time
                for (int i = 0 ; i < neighbors ; i++)
                {
                    for (int j = 0; j < num_rings; j++)
                    {
                        int size = m.manager->halo_offsets_before_glue[j * neighbors + i + 1] * bsize - m.manager->halo_offsets_before_glue[j * neighbors + i] * bsize;

                        if (size != 0)
                        {
                            cudaMemcpy(b.raw() + m.manager->halo_offsets_before_glue[j * neighbors + i]*bsize, &(b.explicit_host_buffer[b.buffer_size + offset]), size * sizeof(typename TConfig::VecPrec), cudaMemcpyHostToDevice);
                        }

                        offset += size;
                    }
                }
            }
            else
            {
                FatalError("num_rings != 1 && num_cols != 1 not supported\n", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }

#else
        FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
    }
}
#endif
// if 0
#endif
//MPI
} // namespace amgx
