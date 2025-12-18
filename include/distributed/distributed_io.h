// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
