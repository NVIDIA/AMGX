// SPDX-FileCopyrightText: 2019 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <error.h>

namespace amgx {
/** Transports parameters for matrix_upload_distributed() call. */
class MatrixDistribution 
{
public:
    enum class PartitionInformation {
        None,
        PartitionVec,
        PartitionOffsets
    };
private:
    int m_allocated_halo_depth;
    int m_num_import_rings;
    PartitionInformation m_partition_information;
    bool m_has_32bit_col_indices;
    const void *m_partition_data;
public:
    MatrixDistribution() :
        m_allocated_halo_depth(1),
        m_num_import_rings(1),
        m_partition_information(PartitionInformation::None),
        m_has_32bit_col_indices(false),
        m_partition_data(nullptr) {};
    void setNumImportRings(int num_import_rings) { m_num_import_rings = num_import_rings; }
    int getNumImportRings() const { return m_num_import_rings; }

    void setAllocatedHaloDepth(int allocated_halo_depth) { m_allocated_halo_depth = allocated_halo_depth; }
    int getAlllocatedHaloDepth() const { return m_allocated_halo_depth; }

    void set32BitColIndices(bool use32bit) { m_has_32bit_col_indices = use32bit; }
    int get32BitColIndices() const { return m_has_32bit_col_indices; }

    PartitionInformation getPartitionInformationStyle() const { return m_partition_information; }
    void setPartitionVec(const int* partition_vector) 
    {
        // Setting a "NULL" partition vector is valid, as the  upload routine will generate one in that case
        m_partition_information = PartitionInformation::PartitionVec;
        m_partition_data = partition_vector;
    }
    void setPartitionOffsets(const void* partition_offsets)
    {
        if (partition_offsets != nullptr) {
            m_partition_information = PartitionInformation::PartitionOffsets;
            m_partition_data = partition_offsets;
        }
        else {
            FatalError("partition_offsets cannot be NULL", AMGX_ERR_BAD_PARAMETERS);
        }
    }
    const void* getPartitionData() const { return m_partition_data; }
};

} // namespace amgx