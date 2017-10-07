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

#include <matrix_io.h>

namespace amgx
{

template <class T>
AMGX_ERROR free_maps_one_ring(T num_neighbors, T *neighbors, T *btl_sizes, T **btl_maps, T *lth_sizes, T **lth_maps);

template<class T_Config>
struct DistributedRead
{
    typedef T_Config TConfig;
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    typedef Vector<i64vec_value_type_h> I64Vector_h;
    static AMGX_ERROR distributedRead(const char *fnamec, Matrix<TConfig> &A, Vector<TConfig> &b, Vector<TConfig> &x, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec,  unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN)
    {
        FatalError("distributedRead not supported for device matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct DistributedRead<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    static void genRowPartitionsEqual(int partitions, IVector_h &partSize, int n_rows, IVector_h &partitionVec);

    static void consolidatePartitions(IVector_h &partSize, IVector_h &partitionVec, int partitions);
    static void readRowPartitions(const char *fnamec, int num_partitions, IVector_h &partSize, IVector_h &partitionVec);

    static void genMapRowPartitions(int rank, const IVector_h &partSize, IVector_h &partitionVec, IVector_h  &partRowVec);

    static void remapReadColumns(Matrix<TConfig_h> &A, IVector_h &colMapVec);

    static AMGX_ERROR distributedRead(const char *fnamec, Matrix<TConfig_h> &A, Vector<TConfig_h> &b, Vector<TConfig_h> &x, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN);
    static AMGX_ERROR distributedRead(const char *fnamec, Matrix<TConfig_h> &A, Vector<TConfig_h> &b, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props = io_config::RHS | io_config::MTX);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct DistributedRead<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef Vector<i64vec_value_type_h> I64Vector_h;
        static AMGX_ERROR distributedRead(const char *fnamec, Matrix<TConfig_d> &A, Vector<TConfig_d> &b, Vector<TConfig_d> &x, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN);
        static AMGX_ERROR distributedRead(const char *fnamec, Matrix<TConfig_d> &A, Vector<TConfig_d> &b, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props = io_config::RHS | io_config::MTX);
    private:
        static AMGX_ERROR distributedReadDeviceInit(const char *fnamec, Matrix<TConfig_h> &Ah_part, Matrix<TConfig_d> &A, Vector<TConfig_h> &bh_part, Vector<TConfig_h> &xh_part, I64Vector_h &part_offsets_h, int allocated_halo_depth, int part, int partitions, IVector_h &partSize, IVector_h &partitionVec, unsigned int props);
};

} // end namespace amgx
