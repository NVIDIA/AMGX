/* Copyright (c) 2013-2019, NVIDIA CORPORATION. All rights reserved.
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

namespace amgx {
/** Transports parameters for matrix_upload_distributed() call. */
class MatrixDistribution 
{
private:
    int m_allocated_halo_depth;
    int m_num_import_rings;
    bool m_has_full_partition_vec;
    bool m_has_32bit_col_indices;
    const int *m_partition_vector;
public:
    MatrixDistribution() :
        m_allocated_halo_depth(1),
        m_num_import_rings(1),
        m_has_full_partition_vec(false),
        m_has_32bit_col_indices(false),
        m_partition_vector(nullptr) {};
    void setNumImportRings(int num_import_rings) { m_num_import_rings = num_import_rings; }
    int getNumImportRings() const { return m_num_import_rings; }

    void setAllocatedHaloDepth(int allocated_halo_depth) { m_allocated_halo_depth = allocated_halo_depth; }
    int getAlllocatedHaloDepth() const { return m_allocated_halo_depth; }

    void set32BitColIndices(bool use32bit) { m_has_32bit_col_indices = use32bit; }
    int get32BitColIndices() const { return m_has_32bit_col_indices; }

    void setExplicitPartitionVec(const int* partition_vector) 
    {
        if (partition_vector != nullptr) {
            m_has_full_partition_vec = true;
            m_partition_vector = partition_vector;
        }
    }
    const int* getPartitionVec() const { return m_partition_vector; }
};

} // namespace amgx