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

#include <distributed/distributed_io.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust_wrapper.h>
#include <amgx_types/util.h>

namespace amgx
{

template <class T>
AMGX_ERROR free_maps_one_ring(T num_neighbors, T *neighbors, T *btl_sizes, T **btl_maps, T *lth_sizes, T **lth_maps)
{
    if (neighbors   != NULL) { free(neighbors); }

    if (btl_maps != NULL)
    {
        for (T i = 0; i < num_neighbors; i++)
            if (btl_maps[i] != NULL) { free(btl_maps[i]); }

        free(btl_maps);
    }

    if (lth_maps != NULL)
    {
        for (T i = 0; i < num_neighbors; i++)
            if (lth_maps[i] != NULL) { free(lth_maps[i]); }

        free(lth_maps);
    }

    if (btl_sizes   != NULL) { free(btl_sizes); }

    if (lth_sizes   != NULL) { free(lth_sizes); }

    return AMGX_OK;
}

namespace
{
// partitioning routines
void create_part_offsets_equal_rows(INDEX_TYPE num_part, int64_t num_rows_total, int64_t *part_offsets_h)
{
    for (int i = 0; i < num_part; i++)
    {
        part_offsets_h[i] = i * num_rows_total / num_part;
    }

    part_offsets_h[num_part] = num_rows_total;
}

template <class T_Config_src, class T_Config_dst>
void transfer_values(const INDEX_TYPE my_id, INDEX_TYPE num_part, const int64_t *part_offsets, const Matrix<T_Config_src> &A, Matrix<T_Config_dst> &A_part)
{
    if (A_part.values.size() == 0 || num_part == 0 || A_part.row_offsets.size() == 0)
    {
        FatalError("Partitioning was not performed", AMGX_ERR_BAD_PARAMETERS);
    }

    thrust::copy(A.values.begin() + (INDEX_TYPE)A.row_offsets[part_offsets[my_id]]*A.get_block_size(), A.values.begin() + (INDEX_TYPE)A.row_offsets[part_offsets[my_id + 1]]*A.get_block_size(), A_part.values.begin());

    if (A.hasProps(DIAG))
    {
        thrust::copy(A.values.begin() + (A.diag[0] + part_offsets[my_id])*A.get_block_size(), A.values.begin() + (A.diag[0] + part_offsets[my_id + 1])*A.get_block_size(), A_part.values.begin() + A_part.row_offsets[A_part.row_offsets.size() - 1]*A.get_block_size());
        cudaCheckError();
    }
}

template <class T_Config_src, class T_Config_dst>
void copyPartition(const INDEX_TYPE my_id, INDEX_TYPE num_part, const int64_t *part_offsets, const INDEX_TYPE block_dimx, const INDEX_TYPE block_dimy, const Vector<T_Config_src> &b, const Vector<T_Config_src> &x, Vector<T_Config_dst> &b_part, Vector<T_Config_dst> &x_part)
{
    if (b.is_vector_read_partitioned())
    {
        b_part.set_block_dimx(1);
        b_part.set_block_dimy(block_dimy);
        x_part.set_block_dimx(1);
        x_part.set_block_dimy(block_dimx);
        b_part.resize(b.size());
        x_part.resize(x.size());
        thrust::copy(b.begin(), b.end(), b_part.begin());
        thrust::copy(x.begin(), x.end(), x_part.begin());
        cudaCheckError();
    }
    else
    {
        FatalError("Partitioning mismatch", AMGX_ERR_BAD_PARAMETERS);
    }
}

template <class T_Config_src, class T_Config_dst>
void partition(const INDEX_TYPE my_id, INDEX_TYPE num_part, const int64_t *part_offsets, const INDEX_TYPE block_dimx, const INDEX_TYPE block_dimy, const Vector<T_Config_src> &b, const Vector<T_Config_src> &x, Vector<T_Config_dst> &b_part, Vector<T_Config_dst> &x_part)
{
    b_part.set_block_dimx(1);
    b_part.set_block_dimy(block_dimy);
    x_part.set_block_dimx(1);
    x_part.set_block_dimy(block_dimx);
    b_part.resize((part_offsets[my_id + 1] - part_offsets[my_id])*block_dimy);
    x_part.resize((part_offsets[my_id + 1] - part_offsets[my_id])*block_dimx);
    thrust::copy(b.begin() + part_offsets[my_id]*block_dimy, b.begin() + part_offsets[my_id + 1]*block_dimy, b_part.begin());
    thrust::copy(x.begin() + part_offsets[my_id]*block_dimx, x.begin() + part_offsets[my_id + 1]*block_dimx, x_part.begin());
    cudaCheckError();
}

template <class T_Config_src, class T_Config_dst>
void copyPartition(const INDEX_TYPE my_id, INDEX_TYPE num_part, const int64_t *part_offsets, const Matrix<T_Config_src> &A, Matrix<T_Config_dst> &A_part)
{
    if (A.is_matrix_read_partitioned())
    {
        A_part.addProps(CSR);

        if (A.hasProps(DIAG)) { A_part.addProps(DIAG); }

        A_part.resize(A.get_num_rows(), A.get_num_cols(), A.get_num_nz(), A.get_block_dimy(), A.get_block_dimx(), 1);
        thrust::copy(A.col_indices.begin(), A.col_indices.end(), A_part.col_indices.begin());
        thrust::copy(A.row_offsets.begin(), A.row_offsets.end(), A_part.row_offsets.begin());
        thrust::copy(A.values.begin(), A.values.end(), A_part.values.begin());

        if (A.hasProps(DIAG))
        {
            /*
             thrust::copy(A.values.begin() + A.get_num_nz()*A.get_block_size(), A.values.end(), A_part.values.begin()+A_part.row_offsets[A_part.row_offsets.size()-1]*A.get_block_size());*/
            A_part.addProps(DIAG);
        }

        cudaCheckError();
    }
    else
    {
        FatalError("Partitioning mismatch", AMGX_ERR_BAD_PARAMETERS);
    }
}

template <class T_Config_src, class T_Config_dst>
void partition(const INDEX_TYPE my_id, INDEX_TYPE num_part, const int64_t *part_offsets, const Matrix<T_Config_src> &A, Matrix<T_Config_dst> &A_part)
{
    if (num_part == 0)
    {
        FatalError("Partitioning scheme is not set", AMGX_ERR_BAD_PARAMETERS);
    }

    A_part.addProps(CSR);

    if (A.hasProps(DIAG)) { A_part.addProps(DIAG); }

    int device = -1;
    cudaGetDevice( &device );
    printf("Processing partition %d/%d size: %ld offset %d nnz %d on device %d\n", my_id + 1, num_part, part_offsets[my_id + 1] - part_offsets[my_id], (int)part_offsets[my_id], (int)(A.row_offsets[part_offsets[my_id + 1]] - A.row_offsets[part_offsets[my_id]]), device);
    A_part.resize(part_offsets[my_id + 1] - part_offsets[my_id], A.get_num_cols(), (INDEX_TYPE)A.row_offsets[part_offsets[my_id + 1]] - (INDEX_TYPE)A.row_offsets[part_offsets[my_id]], A.get_block_dimy(), A.get_block_dimx(), 1);
    thrust::copy(A.col_indices.begin() + (INDEX_TYPE)A.row_offsets[part_offsets[my_id]], A.col_indices.begin() + (INDEX_TYPE)A.row_offsets[part_offsets[my_id + 1]], A_part.col_indices.begin());
    thrust::copy(A.row_offsets.begin() + part_offsets[my_id], A.row_offsets.begin() + part_offsets[my_id + 1] + 1, A_part.row_offsets.begin());
    thrust::transform(A_part.row_offsets.begin(), A_part.row_offsets.end(), thrust::constant_iterator<INDEX_TYPE>(A.row_offsets[part_offsets[my_id]]), A_part.row_offsets.begin(), thrust::minus<INDEX_TYPE>());
    cudaCheckError();
    transfer_values(my_id, num_part, part_offsets, A, A_part);
}
} // end of partitioning routines

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::genRowPartitionsEqual(int partitions, IVector_h &partSize, int n_rows, IVector_h &partitionVec)
{
    int i, p;
    IVector_h scanPartSize(partitions + 1);

    //Initialize partSize, partitionVec by equal partitioning
    for (p = 0; p < partitions; p++)
    {
        uint64_t t = (uint64_t)p * (uint64_t)n_rows / (uint64_t)partitions;
        scanPartSize[p] = t;
    }

    scanPartSize[partitions] = n_rows;
    partSize.resize(partitions);
    partitionVec.resize(n_rows);
    p = 0;

    for (i = 0; i < n_rows; i++)
    {
        if (i >= scanPartSize[p + 1]) { p++; }

        partitionVec[i] = p;
    }

    for (p = 0; p < partitions; p++)
    {
        partSize[p] = scanPartSize[p + 1] - scanPartSize[p];
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::consolidatePartitions(IVector_h &partSize, IVector_h &partitionVec, int partitions)
{
    std::stringstream msg;
    int read_partitions = 0;

    for (int i = 0; i < partitionVec.size(); i++) { read_partitions = max(read_partitions, partitionVec[i]); }

    read_partitions++;

    if (read_partitions % partitions != 0)
    {
        FatalError("Only integer number of partitions per rank is supported", AMGX_ERR_IO);
    }

    if (read_partitions != partitions)
    {
        msg << "Found " << read_partitions << " performing consolidation\n";
    }

    int partsPerRank = read_partitions / partitions;
    partSize.resize(partitions);
    thrust::fill(partSize.begin(), partSize.end(), 0);
    cudaCheckError();

    for (int i = 0; i < partitionVec.size(); i++)
    {
        int p = partitionVec[i] / partsPerRank;
        partitionVec[i] = p;
        partSize[p]++;
    }

    msg << "Read consolidated partition sizes: ";

    for (int i = 0; i < partitions; i++)
    {
        msg << partSize[i] << "  ";
    }

    msg << "\n";
    amgx_output(msg.str().c_str(), 0);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::readRowPartitions(const char *fnamec, int num_partitions, IVector_h &partSize, IVector_h &partitionVec)
{
    /*
      Partition vector format
      vector of int - global array of partition ids, mapping each row_id to a partition id
    */
    std::string err, fname(fnamec);
    int N;
    std::stringstream msg;
    FILE *fin_rowpart = fopen(fname.c_str(), "rb");
    const int size_of_int = sizeof(int);
    char buf[size_of_int];
    int c;
    int chr_cnt = 0;

    while ((c = fgetc(fin_rowpart)) != EOF)
    {
        buf[chr_cnt] = (char) c;
        chr_cnt++;

        if (chr_cnt % size_of_int == 0)
        {
            partitionVec.push_back(int(*buf));
            chr_cnt = 0;
        }
    }

    N = partitionVec.size();
    msg << "Finished reading partition vector, consisting of " << N << " rows\n";
    amgx_output(msg.str().c_str(), 0);
    consolidatePartitions(partSize, partitionVec, num_partitions);
    //amgx_output(msg.str().c_str(), 0);
    fclose(fin_rowpart);
}

// Remap column indices: column indices that correspond to rows
// belonging to the same partition will be consecutive after remapping.
// For instance if partition is [0 1 0 1 0 1], remapping of columns will be:
// 0 -> 0
// 1 -> 3
// 2 -> 1
// 3 -> 4
// 4 -> 2
// 5 -> 5
// In other words:
// 0 - 2 is partition 0.
// 3 - 5 is partition 1.
// This way it is easy to determine if we have an edge to another partition just by using the column index.
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::genMapRowPartitions(int rank, const IVector_h &partSize, IVector_h &partitionVec, IVector_h  &partRowVec)
{
    // partitionVec will contain Column Map on exit
    if (partitionVec.size() == 0) { return; }

    int num_part = partSize.size();
    IVector_h scanPartSize(num_part + 1); // scan of partition sizes
    IVector_h partCount(num_part, 0);// partition counters
    IVector_h &colMapVec = partitionVec; // map for column indices old_glb_i-> new_glb_i, reusing the same vector
    int p;
    thrust::inclusive_scan(partSize.begin(), partSize.end(), &scanPartSize[1]);
    cudaCheckError();
    scanPartSize[0] = 0;

    for (int old_glb_i = 0; old_glb_i < partitionVec.size(); old_glb_i++)
    {
        //printf("partitionVec[%d] = %d\n", old_glb_i, partitionVec[old_glb_i]);
        if (partitionVec[old_glb_i] >= num_part)
        {
            FatalError("Bad partition vector", AMGX_ERR_IO);
        }

        p = partitionVec[old_glb_i];

        if (p == rank) { partRowVec.push_back(old_glb_i); }

        int new_loc_i = partCount[p];
        int new_glb_i = scanPartSize[p] + new_loc_i;
        colMapVec[old_glb_i] = new_glb_i;
        partCount[p]++;
    }

    bool is_err = (partRowVec.size() != scanPartSize[rank + 1] - scanPartSize[rank]);

    for (p = 0; p < num_part; p++)
    {
        is_err = is_err || (partCount[p] != scanPartSize[p + 1] - scanPartSize[p]);
    }

    std::string err = "Error: reading row offsets";

    if (is_err) { FatalError(err, AMGX_ERR_IO); }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::remapReadColumns(Matrix<TConfig_h> &A, IVector_h &colMapVec)
{
    for (int i = 0; i < A.get_num_nz(); i++)
    {
        A.col_indices[i] = colMapVec[A.col_indices[i]];
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMGX_ERROR DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::distributedRead(const char *fnamec, Matrix<TConfig_h> &A, Vector<TConfig_h> &b, Vector<TConfig_h> &x, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props)
{
    AMG_Configuration t_amgx_cfg;
    AMG_Config *amgx_cfg = t_amgx_cfg.getConfigObject();

    // Call the reader, but only read the dimensions of the matrix.
    // Matrix A will have its dimensions set, but its content will be empty.
    if (AMGX_OK != MatrixIO<TConfig_h>::readSystem(fnamec, A, *amgx_cfg, io_config::SIZE))
    {
        return AMGX_ERR_IO;
    }

    int n_rows = A.get_num_rows();
    // Discard the matrix.
    A.set_initialized(0);
    A.resize(0, 0, 0, 1, 1); //Have to do this for proper resize later on
    A.set_initialized(1);

    // No partition was provided, use naive partitioning.
    if (partitionVec.size() == 0)
    {
        genRowPartitionsEqual(partitions, partSize, n_rows, partitionVec);
    }

    if (n_rows != partitionVec.size())
    {
        FatalError("partition vector size does not match with matrix dimensions.", AMGX_ERR_CONFIGURATION);
    }

    IVector_h partRowVec;

    if (partSize.size() == 0)
    {
        consolidatePartitions(partSize, partitionVec, partitions);
    }

    genMapRowPartitions(part, partSize, partitionVec, partRowVec); //partitionVec contains the map now
    // partitionVec now contains columns remapping information (see remapReadColumns below).
    // partRowVec now contains the list of owned rows.

    // Call the distributed reader, the system will be read into A, b and x.
    // Entries are filtered during reading using partRowVec. Entries
    // that do not belong to the current partition are discarded.
    if (AMGX_OK != MatrixIO<TConfig_h>::readSystem(fnamec, A, b, x, *amgx_cfg, props, partRowVec))
    {
        return AMGX_ERR_IO;
    }

    remapReadColumns(A, partitionVec);

    return AMGX_OK;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMGX_ERROR DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::distributedRead(const char *fnamec, Matrix<TConfig_h> &A, Vector<TConfig_h> &b, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props)
{
    Vector<TConfig_h> v = Vector<TConfig_h>(0);

    if (io_config::hasProps(io_config::RHS, props))
    {
        return distributedRead(fnamec, A, b, v, allocated_halo_depth, part, partitions, partSize, partitionVec, props);
    }
    else
    {
        return distributedRead(fnamec, A, v, b, allocated_halo_depth, part, partitions, partSize, partitionVec, props);
    }
}

// Service function to partition matrix and vectors on host,then upload matrix partition  to device and compute maps internally without boundary separation
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMGX_ERROR DistributedRead<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::distributedReadDeviceInit(const char *fnamec, Matrix<TConfig_h> &Ah_part, Matrix<TConfig_d> &A, Vector<TConfig_h> &bh_part, Vector<TConfig_h> &xh_part, I64Vector_h &part_offsets_h, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props)
{
    Matrix<TConfig_h> Ah;
    Vector<TConfig_h> bh;
    Vector<TConfig_h> xh;
    typedef typename VecPrecisionMap<t_vecPrec>::Type ValueTypeB;
    Ah_part.setResources(A.getResources());
    Ah.setResources(A.getResources());
    AMGX_ERROR err = DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::distributedRead(fnamec, Ah, bh, xh, allocated_halo_depth, part, partitions, partSize, partitionVec, props);

    if (err != AMGX_OK)
    {
        return err;
    }

    //cudaSetDevice(A.getResources()->getDevice(0));
    part_offsets_h.resize(partitions + 1);
    int64_t num_rows = Ah.get_num_rows();

    if (!Ah.is_matrix_read_partitioned())
    {
        create_part_offsets_equal_rows(partitions, num_rows, part_offsets_h.raw());
        partition<TConfig_h, TConfig_h>(part, partitions, part_offsets_h.raw(), Ah, Ah_part);
        partition<TConfig_h, TConfig_h>(part, partitions, part_offsets_h.raw(), Ah.get_block_dimx(), Ah.get_block_dimy(), bh, xh, bh_part, xh_part);
    }
    else
    {
        Ah_part.swap(Ah);
        bh_part.swap(bh);
        xh_part.swap(xh);
        thrust::inclusive_scan(partSize.begin(), partSize.end(), &part_offsets_h[1]);
        cudaCheckError();
        part_offsets_h[0] = 0;
    }

    Ah_part.set_is_matrix_read_partitioned(true);
    bh_part.set_is_vector_read_partitioned(true);
    xh_part.set_is_vector_read_partitioned(true);
    // upload a copy to device
    copyPartition<TConfig_h, TConfig_d>(part, partitions, part_offsets_h.raw(), Ah_part, A);

    if (xh_part.size() == 0)
    {
        if (part == 0)
        {
            printf("Initializing solution vector with zeroes...\n");
        }

        xh_part.resize(num_rows * A.get_block_dimx());
        thrust::fill(xh_part.begin(), xh_part.end(), types::util<ValueTypeB>::get_zero());
    }

    if (A.manager == NULL )
    {
        A.manager = new DistributedManager<TConfig_d>(A);
    }
    else
    {
        A.setManagerExternal();
    }

    A.manager->createComms(A.getResources());
    // copy part offsets
    A.manager->part_offsets_h.resize(partitions + 1);

    for (int i = 0; i < partitions + 1; i++)
    {
        A.manager->part_offsets_h[i] = part_offsets_h[i];
    }

    A.manager->num_rows_global = A.manager->part_offsets_h[partitions];
    A.manager->part_offsets = A.manager->part_offsets_h;

    if (partitions > 1)
    {
        DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
        prep->set_part_offsets(partitions, part_offsets_h.raw());
        prep->create_B2L(A, part, 1);
        delete prep;
    }

    return AMGX_OK;
}

template <class Mat>
void getConsolidationFlags( const Mat *A, int *consolidate_flag, int *cuda_ipc_flag)
{
    AMG_Config *rsrc_cfg = A->getResources()->getResourcesConfig();
    std::string scope;
    rsrc_cfg->getParameter<int>("fine_level_consolidation", *consolidate_flag, "default", scope);
    rsrc_cfg->getParameter<int>("use_cuda_ipc_consolidation", *cuda_ipc_flag, "default", scope);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMGX_ERROR DistributedRead<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::distributedRead(const char *fnamec, Matrix<TConfig_d> &A, Vector<TConfig_d> &b, Vector<TConfig_d> &x, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props)
{
    Resources *resources = A.getResources();
    cudaSetDevice(resources->getDevice(0));
    Matrix<TConfig_h> Ah;
    Vector<TConfig_h> bh;
    Vector<TConfig_h> xh;
    I64Vector_h part_offsets_h;
    std::string solver_scope, solver_value;
    std::string precond_scope, precond_value;
    AlgorithmType algorithm_s, algorithm_p;
    resources->getResourcesConfig()->getParameter<std::string>("solver", solver_value, "default", solver_scope);
    algorithm_s = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", solver_scope);
    resources->getResourcesConfig()->getParameter<std::string>("preconditioner", precond_value, solver_scope, precond_scope);
    algorithm_p = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", precond_scope);
    bool isClassical = false;

    // Detect whether one does CLASSICAL or AGGREGATION
    if (algorithm_s == CLASSICAL && algorithm_p == CLASSICAL)
    {
        if (allocated_halo_depth > 2)
        {
            FatalError("allocated_halo_depth > 2 not supported in CLASSICAL", AMGX_ERR_BAD_PARAMETERS);
        }

        isClassical = true;
    }
    else
    {
        if (allocated_halo_depth > 1)
        {
            FatalError("allocated_halo_depth > 1 not supported in AGGREGATION", AMGX_ERR_BAD_PARAMETERS);
        }
    }

    //Matrix<TConfig_d>* Ad = new Matrix<TConfig_d>();
    //Ad->setResources(A.getResources());
    Matrix<TConfig_d> *Ad;
    int consolidate_flag, cuda_ipc_flag;
    getConsolidationFlags( &A, &consolidate_flag, &cuda_ipc_flag);

    if (consolidate_flag != 0 && partitions > 1 && A.get_allow_boundary_separation())
    {
        Ad = new Matrix<TConfig_d>;
        Ad->setResources(resources);
    }
    else
    {
        Ad = &A;
    }

    if (isClassical && consolidate_flag)
    {
        FatalError("Fine level consolidation not supported in CLASSICAL", AMGX_ERR_BAD_PARAMETERS);
    }

    // Reset distributed manager
    if (A.manager != NULL )
    {
        delete A.manager;
        A.manager = NULL;
    }

    A.manager = new DistributedManager<TConfig_d>(A);
    A.setManagerExternal();
    AMGX_ERROR err = DistributedRead<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::distributedReadDeviceInit(fnamec, Ah, *Ad, bh, xh, part_offsets_h, allocated_halo_depth, part, partitions, partSize, partitionVec, props);

    if (err != AMGX_OK)
    {
        return err;
    }

    int64_t num_rows = Ah.get_num_rows();
    int block_dimx = Ah.get_block_dimx();
    int block_dimy = Ah.get_block_dimy();
    int block_size = block_dimx * block_dimy;
    void *x_host = xh.raw(), *b_host = bh.raw();
    int sizeof_v_val = (t_vecPrec == AMGX_vecDouble) ? sizeof(double) : sizeof(float);
    //int sizeof_v_val =  sizeof(typename VecPrecisionMap<t_vecPrec>::Type);
    cudaHostRegister(x_host, num_rows * block_dimx * sizeof_v_val, cudaHostRegisterMapped);
    cudaHostRegister(b_host, num_rows * block_dimy * sizeof_v_val, cudaHostRegisterMapped);
    cudaCheckError();

    if (!isClassical)
    {
        // AGGREGATION path
        if (partitions > 1)
        {
            if (Ad->get_allow_boundary_separation())
            {
                // TODO: This can be done without exporting of maps
                int *btl_sizes = NULL;
                int **btl_maps  = NULL;
                int *lth_sizes = NULL;
                int **lth_maps = NULL;
                int *neighbors = NULL;
                int num_neighbors = Ad->manager->num_neighbors();
                neighbors = (int *)malloc((num_neighbors) * sizeof(int));
                Ad->manager->malloc_export_maps(&btl_maps, &btl_sizes, &lth_maps, &lth_sizes);
                Ad->manager->export_neighbors(neighbors);

                if (A.manager != NULL )
                {
                    delete A.manager;
                }

                A.manager = new DistributedManager<TConfig_d>(A, 1, 1, num_neighbors, neighbors);
                A.manager->cacheMapsOneRing((int const **) btl_maps, (const int *)btl_sizes, (int const **)lth_maps, (const int *)lth_sizes);
                A.setManagerExternal();
                A.manager->createComms(A.getResources());
                A.manager->setAConsolidationFlags(A);

                if (A.manager->isFineLevelConsolidated())
                {
                    A.addProps(CSR);
                    A.setColsReorderedByColor(false);
                    A.delProps(COO);
                    A.delProps(DIAG);
                    A.setColsReorderedByColor(false);

                    if (Ah.hasProps(DIAG))
                    {
                        A.addProps(DIAG);
                    }

                    int nnz = Ah.get_num_nz();
                    typedef typename MatPrecisionMap<t_matPrec>::Type ValueType;
                    int *row_ptrs = NULL, *col_indices = NULL;
                    void *values, *diag_data = NULL;
                    // Use local column indicies now
                    col_indices = Ad->col_indices.raw();
                    //row_ptrs = Ad->row_offsets.raw();
                    // values = Ad->values.raw();
                    //col_indices = Ad->col_indices.raw();
                    // row offsets are still global and not reordered
                    row_ptrs = Ah.row_offsets.raw();
                    values = Ah.values.raw();
                    // Do pinning of some buffers since fine level consolidation crashes when only one GPU is used
                    int sizeof_m_val = sizeof(ValueType);
                    cudaHostRegister(values, nnz * block_size * sizeof_m_val, cudaHostRegisterMapped);
                    cudaCheckError();

                    if (Ah.hasProps(DIAG))
                    {
                        //diag_data = (Ad->values.raw() + nnz*block_size);
                        diag_data = (Ah.values.raw() + nnz * block_size);
                        cudaHostRegister((void *)diag_data, num_rows * block_size * sizeof_m_val, cudaHostRegisterMapped);
                        cudaCheckError();
                    }

                    /*
                    cudaHostRegister(col_indices,nnz*sizeof(int),cudaHostRegisterMapped);
                    cudaCheckError();
                    */
                    cudaHostRegister(row_ptrs, (num_rows + 1)*sizeof(int), cudaHostRegisterMapped);
                    cudaCheckError();
                    cudaSetDevice(A.getResources()->getDevice(0));
                    A.manager->consolidateAndUploadAll(num_rows, nnz, block_dimx, block_dimy, row_ptrs, col_indices, values, diag_data, A);
                    A.set_initialized(1);
                    cudaSetDevice(A.getResources()->getDevice(0));

                    if (diag_data != NULL)
                    {
                        cudaHostUnregister(diag_data);
                    }

                    cudaHostUnregister(values);
                    cudaHostUnregister(row_ptrs);
                    //cudaHostUnregister(col_indices);
                    cudaCheckError();
                    delete Ad;
                }
                else
                {
                    A.manager->createComms(A.getResources());
                    A.manager->updateMapsReorder();
                } // End consolidation check

                free_maps_one_ring<int>(num_neighbors, neighbors, btl_sizes, btl_maps, lth_sizes, lth_maps);
                A.set_is_matrix_read_partitioned(true);
                b.set_is_vector_read_partitioned(true);
                x.set_is_vector_read_partitioned(true);
                x.setManager(*(A.manager));
                b.setManager(*(A.manager));
                b.set_block_dimx(1);
                b.set_block_dimy(block_dimy);
                x.set_block_dimx(1);
                x.set_block_dimy(block_dimx);
                A.manager->transformAndUploadVector(b, b_host, num_rows, b.get_block_dimy());
                A.manager->transformAndUploadVector(x, x_host, num_rows, x.get_block_dimy());
                cudaHostUnregister(b_host);
                cudaHostUnregister(x_host);
                cudaDeviceSynchronize();
                return AMGX_OK;
            } // end of boundary sparation
        }
        else
        {
            A.computeDiagonal();
            A.set_initialized(1);
            A.setView(OWNED);
        } // End if partitions>1
    }
    else
    {
        // CLASSICAL
        /* WARNING: in the classical path, even if a single partition is used it needs
           to setup the distributed data structures, because they are later used in the
           code. For instance, halo_offsets must be set correctly, otherwise
           generateInterpolationMatrix_1x1 -> exchange_halo_2ring -> setup -> do_setup
           will fail when accessing them. Therefore, classical path can not be encapsulated
           in the if (npartitions > 1) { ... } statement and cl-116816 moves it out. */
        if (Ad->get_allow_boundary_separation())
        {
            A.set_initialized(0);
            A.manager->neighbors.resize(0);
            A.manager->renumberMatrixOneRing();
            A.manager->createOneRingHaloRows();
            A.manager->getComms()->set_neighbors(A.manager->num_neighbors());
            A.setView(OWNED);
            A.set_initialized(1);
            A.set_is_matrix_read_partitioned(true);
            b.set_is_vector_read_partitioned(true);
            x.set_is_vector_read_partitioned(true);
            x.setManager(*(A.manager));
            b.setManager(*(A.manager));
            b.set_block_dimx(1);
            b.set_block_dimy(block_dimy);
            x.set_block_dimx(1);
            x.set_block_dimy(block_dimx);
            A.manager->transformAndUploadVector(b, b_host, num_rows, b.get_block_dimy());
            A.manager->transformAndUploadVector(x, x_host, num_rows, x.get_block_dimy());
            cudaHostUnregister(b_host);
            cudaHostUnregister(x_host);
            cudaDeviceSynchronize();
            return AMGX_OK;
        }
    }

    // just copy remaining data to device
    copyPartition<TConfig_h, TConfig_d>(part, partitions, part_offsets_h.raw(), Ah.get_block_dimx(), Ah.get_block_dimy(), bh, xh, b, x);
    cudaHostUnregister(b_host);
    cudaHostUnregister(x_host);
    return AMGX_OK;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMGX_ERROR DistributedRead<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::distributedRead(const char *fnamec, Matrix<TConfig_d> &A, Vector<TConfig_d> &b, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props)
{
    Vector<TConfig_d> v = Vector<TConfig_d>(0);

    if (io_config::hasProps(io_config::RHS, props))
    {
        return distributedRead(fnamec, A, b, v, allocated_halo_depth, part, partitions, partSize, partitionVec, props);
    }
    else
    {
        return distributedRead(fnamec, A, v, b, allocated_halo_depth, part, partitions, partSize, partitionVec, props);
    }
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class DistributedRead<TemplateMode<CASE>::Type >;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}
