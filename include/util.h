// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <misc.h>
#include <error.h>
#include <basic_types.h>
#include <thrust/reduce.h>
#include <matrix.h>
#include <limits>

#include <amgx_types/io.h>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

namespace amgx
{

/****************************************************
 * Debugging tools
 ***************************************************/
template<class TConfig>
void writeVector(const char *fname, const Vector<TConfig> &v)
{
    std::ofstream fout;
    fout.open(fname, std::ofstream::out | std::ofstream::app);
    fout << v.size() << std::endl;
    typename Vector<TConfig>::value_type val;
    fout.precision(14);
    fout.width(16);

    for (int i = 0; i < v.size(); i++)
    {
        val = v[i];
        fout << val << std::endl;
    }

    fout << std::endl;
    fout.close();
}

template<class TConfig>
void writeVector(const char *fname, const char *label, const Vector<TConfig> &v, bool full_precision = false)
{
    typedef std::numeric_limits< double > dblprec;
    std::ofstream fout;
    fout.open(fname, std::ofstream::out | std::ofstream::app);
    fout << label << std::endl << "size : " << v.size() << std::endl;

    if (full_precision)
    {
        fout.precision(dblprec::digits10);
        fout.width(dblprec::digits10);
    }
    else
    {
        fout.precision(14);
        fout.width(16);
    }

    typename Vector<TConfig>::value_type val;

    for (int i = 0; i < v.size(); i++)
    {
        val = v[i];
        fout << std::scientific << val << std::endl;
    }

    fout << std::endl;
    fout.close();
}

template<class TConfig>
void printVector(const char *label, const Vector<TConfig> &v)
{
    std::stringstream ss;
    ss << label << ": " << std::endl;

    for (int i = 0; i < v.size(); i++)
    {
        ss << std::setprecision(4) << std::setw(8) << v[i] << std::endl;
    }

    ss << std::endl;
    amgx_output(ss.str().c_str(), ss.str().length());
}

template<class TConfig>
void printVectorToFile(const char *f, const Vector<TConfig> &v)
{
    cudaDeviceSynchronize();
    cudaCheckError();
    int rank = 0;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    char filename[1024];
    FILE *fid = NULL;
#ifdef _WIN32
    _snprintf_s(filename, 1024, 1024, "%s_r%d.mtx", f, rank);
#else
    snprintf(filename, 1024, "%s_r%d.mtx", f, rank);
#endif
    fid = fopen(filename, "w");
    std::stringstream ss;

    for (int i = 0; i < v.size(); i++)
    {
        ss /*<< std::setprecision(4) << std::setw(8)*/ << v[i] << std::endl;
    }

    fprintf(fid, "%s", ss.str().c_str());
    cudaDeviceSynchronize();
    cudaGetLastError();
    fclose(fid);
}

template<class TConfig>
void printDenseNRows(const Matrix<TConfig> &A, int num_rows)
{
    typedef typename TConfig::MatPrec ValueTypeA;
    int num_cols = -1;
    num_cols = amgx::thrust::reduce(A.col_indices.begin(), A.col_indices.end(), num_cols, amgx::thrust::maximum<int>()) + 1;
    cudaCheckError();

    for (int i = 0; i < num_rows; i++)
    {
        std::map<int64_t, ValueTypeA> entries;

        for (int jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            int local_col = A.col_indices[jj];
            entries.insert(std::make_pair((int64_t) local_col, A.values[jj]));
        }

        typedef typename std::map<int64_t, ValueTypeA>::iterator map_it;

        for (map_it iter = entries.begin(); iter != entries.end(); iter++)
        {
            printf("%10d %10ld %10.7f \n", i, iter->first, iter->second);
        }
    }

    printf("\n");
}


template<class TConfig, class IVector>
void printDenseSerial(const Matrix<TConfig> &A, int num_local_rows, int64_t num_cols_global, IVector &local_to_global, int64_t base_index, int index_range)
{
    typedef typename TConfig::MatPrec ValueTypeA;

    for (int i = 0; i < num_local_rows; i++)
    {
        std::map<int64_t, ValueTypeA> entries;

        for (int jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            int local_col = A.col_indices[jj];
            int64_t global_col;

            if (local_col < index_range) // interior
            {
                global_col = base_index + local_col;
            }
            else
            {
                global_col = local_to_global[local_col - index_range];
            }

            entries.insert(std::make_pair((int64_t) global_col, A.values[jj]));
        }

        typedef typename std::map<int64_t, ValueTypeA>::iterator map_it;

        for (map_it iter = entries.begin(); iter != entries.end(); iter++)
        {
            printf("%10ld %10ld %10.7f \n", base_index + i, iter->first, iter->second);
        }
    }

    printf("\n");
}



template<class TConfig>
void printDense(const Matrix<TConfig> &A)
{
    typedef typename TConfig::MatPrec ValueTypeA;
    int num_cols = -1;
    num_cols = amgx::thrust::reduce(A.col_indices.begin(), A.col_indices.end(), num_cols, amgx::thrust::maximum<int>()) + 1;
    cudaCheckError();

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        std::map<int64_t, ValueTypeA> entries;

        for (int jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            int local_col = A.col_indices[jj];
            entries.insert(std::make_pair((int64_t) local_col, A.values[jj]));
        }

        typedef typename std::map<int64_t, ValueTypeA>::iterator map_it;

        for (map_it iter = entries.begin(); iter != entries.end(); iter++)
        {
            printf("%10d %10ld %10.7f \n", i, iter->first, iter->second);
        }
    }

    printf("\n");
}


template<class TConfig>
struct printMatrix {};

template<class Matrix>
inline
void printMatrixStats(const Matrix &A)
{
    amgx_printf("Matrix properties: A.num_rows = %d, A.num_cols = %d, A.num_nz = %d, A.block_size = %d, A.hasProps(DIAG) = %d, A.hasProps(CSR) = %d, A.hasProps(COO) = %d\n", A.get_num_rows(), A.get_num_cols(), A.get_num_nz(), A.get_block_dimx(), A.hasProps(DIAG) ? 1 : 0, A.hasProps(CSR) ? 1 : 0, A.hasProps(COO) ? 1 : 0 );
}

template<class Vector>
inline
void printVectorStats(const Vector &v)
{
    amgx_printf("Vector properties: v.size() = %d, v.block_size = %d, v.block_dimx = %d, v.block_dimy = %d\n", v.size(), v.get_block_size(), v.get_block_dimx(), v.get_block_dimy());
}

// Method to print matrix
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct printMatrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
    static inline void print(const Matrix_h &A, const char *fname)
    {
        FILE *fid = fopen(fname, "w");
        fprintf(fid, "Numrows: %d, NNZ: %d, block_dimy: %d, block_dimx: %d\n", A.get_num_rows(), A.get_num_nz(), A.get_block_dimy(), A.get_block_dimx());
        unsigned int diag_count = 0;
        unsigned int offdiag_count = 0;

        for (int i = 0; i < A.get_num_rows(); ++i)
        {
            // Print diagonal coefficients
            if (A.hasProps(DIAG))
            {
                fprintf(fid, "%9d %9d\n", i, i);

                for (int ie = 0; ie < A.get_block_dimy(); ++ie)
                {
                    fprintf(fid, "   ");

                    for (int je = 0; je < A.get_block_dimx(); ++je, ++diag_count)
                    {
                        fprintf(fid, " %23.16e", A.values[A.diag[diag_count]]);
                    }

                    fprintf(fid, "\n");
                }
            }

            // Print off-diagonal coefficients
            for (int ip = A.row_offsets[i]; ip < A.row_offsets[i + 1]; ++ip)
            {
                int j = A.col_indices[ip];
                fprintf(fid, "%9d %9d\n", i, j);

                for (int ie = 0; ie < A.get_block_dimy(); ++ie)
                {
                    fprintf(fid, "   ");

                    for (int je = 0; je < A.get_block_dimx(); ++je, ++offdiag_count)
                    {
                        fprintf(fid, " %23.16e", A.values[offdiag_count]);
                    }

                    fprintf(fid, "\n");
                }
            }
        }

        fclose(fid);
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct printMatrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
    typedef Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;

    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    static inline
    void print(const Matrix_d &A, const char *fname)
    {
        Matrix_h A_host = A;
        printMatrix<TConfig_h>::print(A_host, fname);
    }
};

} // namespace amgx
