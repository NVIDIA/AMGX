// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <distributed/comms_mpi_gpudirect.h>
#include <basic_types.h>
#include <cutil.h>

namespace amgx
{

/***************************************
 * Source Definitions
 ***************************************/

template <class T_Config>
void CommsMPIDirect<T_Config>::exchange_matrix_halo(IVector_Array &row_offsets,
        I64Vector_Array &col_indices,
        MVector_Array &values,
        I64Vector_Array &halo_row_ids,
        IVector_h &neighbors_list,
        int global_id)
{
    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("MPI Comms module no implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
#ifdef AMGX_WITH_MPI
        int total = 0;
        MPI_Comm mpi_comm = CommsMPIHostBufferStream<T_Config>::get_mpi_comm();
        std::vector<MPI_Request> &requests = CommsMPIHostBufferStream<T_Config>::get_requests();
        MPI_Comm_size( mpi_comm, &total );
        int num_neighbors = neighbors_list.size();
        IVector_Array local_row_offsets(num_neighbors);
        I64Vector_Array local_col_indices(num_neighbors);
        MVector_Array local_values(num_neighbors);
        I64Vector_Array local_row_ids(0);

        if (halo_row_ids.size() != 0)
        {
            local_row_ids.resize(num_neighbors);
        }

        // send metadata
        std::vector<INDEX_TYPE> metadata(num_neighbors * 2); // num_rows+1, num_nz

        for (int i = 0; i < num_neighbors; i++)
        {
            metadata[i * 2 + 0] = row_offsets[i].size();
            metadata[i * 2 + 1] = col_indices[i].size();
            MPI_Isend(&metadata[i * 2 + 0], 2, MPI_INT, neighbors_list[i], 0, mpi_comm, &requests[i]);
        }

        // receive metadata
        std::vector<INDEX_TYPE> metadata_recv(2);

        for (int i = 0; i < num_neighbors; i++)
        {
            MPI_Recv(&metadata_recv[0], 2, MPI_INT, neighbors_list[i], 0, mpi_comm, MPI_STATUSES_IGNORE);
            local_row_offsets[i].resize(metadata_recv[0]);
            local_col_indices[i].resize(metadata_recv[1]);
            local_values[i].resize(metadata_recv[1]);

            if (local_row_ids.size() != 0)
            {
                if (metadata_recv[0] - 1 > 0)
                {
                    local_row_ids[i].resize(metadata_recv[0] - 1);    // row_ids is one smaller than row_offsets
                }
            }
        }

        MPI_Waitall(num_neighbors, &requests[0], MPI_STATUSES_IGNORE); // data is already received, just closing the handles
        // receive matrix data
        typedef typename T_Config::MatPrec mvalue;

        for (int i = 0; i < num_neighbors; i++)
        {
            MPI_Irecv(local_row_offsets[i].raw(), local_row_offsets[i].size(), MPI_INT, neighbors_list[i], 10 * neighbors_list[i] + 0, mpi_comm, &requests[3 * num_neighbors + i]);
            MPI_Irecv(local_col_indices[i].raw(), local_col_indices[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * neighbors_list[i] + 1, mpi_comm, &requests[4 * num_neighbors + i]);
            MPI_Irecv(local_values[i].raw(), local_values[i].size()*sizeof(mvalue), MPI_BYTE, neighbors_list[i], 10 * neighbors_list[i] + 2, mpi_comm, &requests[5 * num_neighbors + i]);

            if (halo_row_ids.size() != 0)
            {
                MPI_Irecv(local_row_ids[i].raw(), local_row_ids[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * neighbors_list[i] + 3, mpi_comm, &requests[7 * num_neighbors + i]);
            }
        }

        // send matrix: row offsets, col indices, values
        for (int i = 0; i < num_neighbors; i++)
        {
            MPI_Isend(row_offsets[i].raw(), row_offsets[i].size(), MPI_INT, neighbors_list[i], 10 * global_id + 0, mpi_comm, &requests[i]);
            MPI_Isend(col_indices[i].raw(), col_indices[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * global_id + 1, mpi_comm, &requests[num_neighbors + i]);
            MPI_Isend(values[i].raw(), values[i].size()*sizeof(mvalue), MPI_BYTE, neighbors_list[i], 10 * global_id + 2, mpi_comm, &requests[2 * num_neighbors + i]);

            if (halo_row_ids.size() != 0)
            {
                MPI_Isend(halo_row_ids[i].raw(), halo_row_ids[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * global_id + 3, mpi_comm, &requests[6 * num_neighbors + i]);
            }
        }

        if (halo_row_ids.size() != 0)
        {
            MPI_Waitall(8 * num_neighbors, &requests[0], MPI_STATUSES_IGNORE);    //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
        }
        else
        {
            MPI_Waitall(6 * num_neighbors, &requests[0], MPI_STATUSES_IGNORE);    //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
        }

        row_offsets.swap(local_row_offsets);
        col_indices.swap(local_col_indices);
        values.swap(local_values);
        halo_row_ids.swap(local_row_ids);
#else
        FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
    }
}

template <class T_Config>
void CommsMPIDirect<T_Config>::exchange_matrix_halo(Matrix_Array &halo_rows, DistributedManager_Array &halo_btl, const Matrix<TConfig> &m)
{
    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("MPI Comms module no implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
#ifdef AMGX_WITH_MPI
        int total = 0;
        int neighbors = CommsMPIHostBufferStream<T_Config>::get_neighbors();
        MPI_Comm mpi_comm = CommsMPIHostBufferStream<T_Config>::get_mpi_comm();
        std::vector<MPI_Request> &requests = CommsMPIHostBufferStream<T_Config>::get_requests();
        MPI_Comm_size( mpi_comm, &total );
        int bsize = m.get_block_size();
        int rings = m.manager->B2L_rings[0].size() - 1;
        int diag = m.hasProps(DIAG);
        std::vector<Matrix<TConfig>> local_copy(halo_rows.size());
        std::vector<DistributedManager<TConfig>> local_copy_manager(halo_rows.size());
        {
            // there shouldn't be any uncompleted requests, because we don't want to rewrite them
            int completed;
            MPI_Testall(requests.size(), &requests[0], &completed, MPI_STATUSES_IGNORE);

            if (!completed)
            {
                MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
            }
        }
        std::vector<INDEX_TYPE> metadata(neighbors * (rings + 1 + 5)); //ring offsets (rings+1), num_rows, num_nz, base_index, index_range

        for (int i = 0; i < neighbors; i++)
        {
            for (int j = 0; j <= rings; j++) { metadata[i * (rings + 1 + 5) + j] = halo_btl[i].B2L_rings[0][j]; }

            metadata[i * (rings + 1 + 5) + rings + 1] = halo_rows[i].get_num_rows();
            metadata[i * (rings + 1 + 5) + rings + 2] = halo_rows[i].get_num_nz();
            metadata[i * (rings + 1 + 5) + rings + 3] = halo_btl[i].base_index();
            metadata[i * (rings + 1 + 5) + rings + 4] = halo_btl[i].index_range();
            metadata[i * (rings + 1 + 5) + rings + 5] = halo_btl[i].L2H_maps[0].size();
            MPI_Isend(&metadata[i * (rings + 1 + 5)], rings + 6, MPI_INT, m.manager->neighbors[i], 0, mpi_comm, &requests[i]);
        }

        std::vector<INDEX_TYPE> metadata_recv(rings + 1 + 5);

        for (int i = 0; i < neighbors; i++)
        {
            MPI_Recv(&metadata_recv[0], rings + 6, MPI_INT, m.manager->neighbors[i], 0, mpi_comm, MPI_STATUSES_IGNORE);
            local_copy[i].addProps(CSR);

            if (diag) { local_copy[i].addProps(DIAG); }

            local_copy[i].resize(metadata_recv[rings + 1], metadata_recv[rings + 1], metadata_recv[rings + 2], m.get_block_dimy(), m.get_block_dimx(), 1);
            local_copy_manager[i].set_base_index(metadata_recv[rings + 3]);
            local_copy_manager[i].set_index_range(metadata_recv[rings + 4]);
            local_copy_manager[i].B2L_rings.resize(1);
            local_copy_manager[i].B2L_rings[0].resize(rings + 1);
            local_copy_manager[i].B2L_maps.resize(1);
            local_copy_manager[i].B2L_maps[0].resize(local_copy[i].get_num_rows());
            local_copy_manager[i].L2H_maps.resize(1);
            local_copy_manager[i].L2H_maps[0].resize(metadata_recv[rings + 5]);

            for (int j = 0; j <= rings; j++) { local_copy_manager[i].B2L_rings[0][j] = metadata_recv[j]; }
        }

        MPI_Waitall(neighbors, &requests[0], MPI_STATUSES_IGNORE); //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
        typedef typename T_Config::MatPrec mvalue;

        for (int i = 0; i < neighbors; i++)
        {
            MPI_Irecv(local_copy[i].row_offsets.raw(), local_copy[i].row_offsets.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 0, mpi_comm, &requests[5 * neighbors + 5 * i]);
            MPI_Irecv(local_copy[i].col_indices.raw(), local_copy[i].col_indices.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 1, mpi_comm, &requests[5 * neighbors + 5 * i + 1]);
            MPI_Irecv(local_copy_manager[i].B2L_maps[0].raw(), local_copy_manager[i].B2L_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 2, mpi_comm, &requests[5 * neighbors + 5 * i + 2]);
            MPI_Irecv(local_copy_manager[i].L2H_maps[0].raw(), local_copy_manager[i].L2H_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 3, mpi_comm, &requests[5 * neighbors + 5 * i + 3]);
            MPI_Irecv(local_copy[i].values.raw(), local_copy[i].values.size()*sizeof(mvalue), MPI_BYTE, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 4, mpi_comm, &requests[5 * neighbors + 5 * i + 4]);
        }

        for (int i = 0; i < neighbors; i++)
        {
            MPI_Isend(halo_rows[i].row_offsets.raw(), halo_rows[i].row_offsets.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 0, mpi_comm, &requests[5 * i]);
            MPI_Isend(halo_rows[i].col_indices.raw(), halo_rows[i].col_indices.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 1, mpi_comm, &requests[5 * i + 1]);
            MPI_Isend(halo_btl[i].B2L_maps[0].raw(), halo_btl[i].B2L_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 2, mpi_comm, &requests[5 * i + 2]);
            MPI_Isend(halo_btl[i].L2H_maps[0].raw(), halo_btl[i].L2H_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 3, mpi_comm, &requests[5 * i + 3]);
            MPI_Isend(halo_rows[i].values.raw(), halo_rows[i].values.size()*sizeof(mvalue), MPI_BYTE, m.manager->neighbors[i], 10 * m.manager->global_id() + 4, mpi_comm, &requests[5 * i + 4]);
        }

        MPI_Waitall(2 * 5 * neighbors, &requests[0], MPI_STATUSES_IGNORE); //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function

        halo_rows.swap(local_copy);
        halo_btl.swap(local_copy_manager);
#else
        FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
    }
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class CommsMPIDirect<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
