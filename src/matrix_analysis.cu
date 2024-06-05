// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <matrix_analysis.h>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <set>
#include <algorithm>

#include <amgx_types/util.h>
#include <amgx_types/io.h>
#include <amgx_types/math.h>

namespace amgx
{

template <class T_Config>
void MatrixAnalysis<T_Config>::valueDistribution(double *minAbs, double *maxAbs)
{
    const int bx = A->get_block_dimx();
    const int by = A->get_block_dimy();
    const int nnz = A->get_num_nz();
    const int bs = bx * by;
    const int bs_alloc = bs + 1;
    const double magicDb = 1.23456789e300;
    std::vector<double> mind(bs_alloc, magicDb), maxd(bs_alloc, 0);
    std::vector<int> minc(bs_alloc, 0), maxc(bs_alloc, 0), nullc(bs_alloc, 0);
    //constPODVector<mat_value_type, index_type> Av= A->values.const_pod();

    if (TConfig::memSpace == AMGX_device)
    {
        FatalError("Device version not implemented", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else if (TConfig::memSpace == AMGX_host)
    {
        fprintf(fout, "Value distribution for matrix: A %8dx%8d, nnz %8d, block %2dx%2d\n", A->get_num_rows(), A->get_num_cols(), nnz, by, bx);

        for (int b = 0; b < nnz; b++)
        {
            for (int c = 0; c < bs; c++)
            {
                ValueTypeA temp = A->values[b * bs + c];
                PODTypeA atemp = types::util<ValueTypeA>::abs(temp);

                if (atemp == 0.0)
                {
                    nullc[c]++;
                }
                else
                {
                    if (atemp < mind[c]) { mind[c] = atemp; }

                    if (atemp > maxd[c]) { maxd[c] = atemp; }

                    if (atemp < minAbs[c]) { minc[c]++; /*fprintf(fout, "b %8d, c %2d: %10.3E\n", b, c, temp);*/ }

                    if (atemp > maxAbs[c]) { maxc[c]++; /*fprintf(fout, "b %8d, c %2d: %10.3E\n", b, c, temp);*/ }
                }
            }
        }

        for (int c = 0; c < bs; c++)
        {
            fprintf(fout, "c %2d: min %10.3E (#%8d < %10.3E), max %10.3E (#%8d > %10.3E), (#%8d == 0)\n", c, mind[c], minc[c], minAbs[c], maxd[c], maxc[c], maxAbs[c], nullc[c]);

            if (mind[c] < mind[bs]) { mind[bs] = mind[c]; }

            if (maxd[c] > maxd[bs]) { maxd[bs] = maxd[c]; }

            minc[bs] += minc[c];
            maxc[bs] += maxc[c];
            nullc[bs] += nullc[c];
        }

        fprintf(fout, "all : min %10.3E (#%8d < %10s), max %10.3E (#%8d > %10s), (#%8d == 0)\n", mind[bs], minc[bs], "threshold", maxd[bs], maxc[bs], "threshold", nullc[bs]);
        fprintf(fout, "\n");
        fprintf(fout, "Ranges for block coefficients:\n");

        for (int y = 0; y < by; y++)
        {
            for (int x = 0; x < bx; x++)
            {
                int c = y * bx + x;

                if (mind[c] == magicDb && maxd[c] == 0.0)
                {
                    fprintf(fout, "[%22.0f]   ", 0.0);
                }
                else
                {
                    fprintf(fout, "[%10.3E, %10.3E]   ", mind[c], maxd[c]);
                }
            }

            fprintf(fout, "\n");
        }

        fprintf(fout, "\n");
    }
}

template <class T_Config>
void MatrixAnalysis<T_Config>::checkSymmetry(bool &structuralSymmetric, bool &symmetric, bool &verbose)
{
    // initially assume symmetric
    structuralSymmetric = true;
    symmetric = true;

    if (TConfig::memSpace == AMGX_device)
    {
        // choose epsilon
        double eps = 1e-12;

        if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecFloat)
        {
            eps = 1e-7;
        }

        const int bx = A->get_block_dimx();
        const int by = A->get_block_dimy();
        const int nnz = A->get_num_nz();

        amgx::thrust::device_vector<bool> symm_d(1, true);
        amgx::thrust::device_vector<bool> struct_symm_d(1, true);
        auto* symmetric_d = amgx::thrust::raw_pointer_cast(symm_d.data());
        auto* structurally_symmetric_d = amgx::thrust::raw_pointer_cast(struct_symm_d.data());
        auto* col_indices = A->col_indices.raw();
        auto* row_offsets = A->row_offsets.raw();
        auto* values = A->values.raw();

        amgx::thrust::for_each_n(thrust::device, amgx::thrust::counting_iterator<int>(0), A->get_num_rows(), [=] __device__ (int i)
        {
            for (int jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++)
            {
                // check structure exists
                int j = col_indices[jj];

                // ignore diagonal
                if (i == j) { continue; }

                // loop over row j, search for column i
                bool found_on_row = false;

                for (int kk = row_offsets[j]; kk < row_offsets[j + 1]; kk++)
                {
                    int k = col_indices[kk];

                    if (k == i)
                    {
                        found_on_row = true;

                        // check values
                        // check all elements
                        const int blocksize = bx * by;

                        for (int m = 0; m < bx * by; m++)
                        {
                            if (types::util<ValueTypeA>::abs(values[jj * blocksize + m] - values[kk * blocksize + m]) > eps)
                            {
                                symmetric_d[0] = false;
                            }
                        }

                        break;
                    }
                }

                // if we didn't find the element, non-symmetric
                if (!found_on_row)
                {
                    structurally_symmetric_d[0] = false;
                    symmetric_d[0] = false;
                }
            }
        });

        symmetric = symm_d[0];
        structuralSymmetric = struct_symm_d[0];
    }
    else if (TConfig::memSpace == AMGX_host)
    {
        // choose epsilon
        double eps;

        if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecDouble)
        {
            eps = 1e-12;
        }
        else if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecFloat)
        {
            eps = 1e-7;
        }
        else
        {
            eps = 1e-12;
        }

        const int bx = A->get_block_dimx();
        const int by = A->get_block_dimy();
        const int nnz = A->get_num_nz();

        if (verbose) { fprintf(fout, "Checking Symmetry of Matrix: A %8dx%8d, nnz %8d, block %2dx%2d\n", A->get_num_rows(), A->get_num_cols(), nnz, bx, by); }

        for (int i = 0; i < A->get_num_rows(); i++)
        {
            for (int jj = A->row_offsets[i]; jj < A->row_offsets[i + 1]; jj++)
            {
                // check structure exists
                int j = A->col_indices[jj];

                // ignore diagonal
                if (i == j) { continue; }

                // loop over row j, search for column i
                bool found_on_row = false;

                for (int kk = A->row_offsets[j]; kk < A->row_offsets[j + 1]; kk++)
                {
                    int k = A->col_indices[kk];

                    if (k == i)
                    {
                        found_on_row = true;
                        // check values
                        // check all elements
                        const int blocksize = bx * by;

                        for (int m = 0; m < bx * by; m++)
                        {
                            if (types::util<ValueTypeA>::abs(A->values[jj * blocksize + m] - A->values[kk * blocksize + m]) > eps)
                            {
                                symmetric = false;
                            }
                        }

                        break;
                    }
                }

                // if we didn't find the element, non-symmetric
                if (!found_on_row)
                {
                    structuralSymmetric = false;
                    symmetric = false;
                }
            }

            // if non structurally symmetric, cannot be symmetric
            if (!structuralSymmetric)
            {
                if (verbose) { fprintf(fout, "A: non-symmetric, non-structurally symmetric\n"); }

                return;
            }
        }
    } // end host path

    // structurally symmetric, at this point - print symmetric or not
    if (symmetric)
    {
        if (verbose) { fprintf(fout, "A: non-symmetric, structurally symmetric\n"); }
    }
    else
    {
        if (verbose) { fprintf(fout, "A: symmetric, structurally symmetric\n"); }
    }

    return;
}

template <class T_Config>
void MatrixAnalysis<T_Config>::checkDiagDominate()
{
    const int num_rows = A->get_num_rows();
    const int bx       = A->get_block_dimx();
    const int by       = A->get_block_dimy();
    const int nnz      = A->get_num_nz();
    const int bs = bx * by;
    //const int bs_alloc= bs+1;
    int k = 0;
    //typename T_Config::MatPrec *sum = new typename T_Config::MatPrec[bx];
    std::vector<PODTypeA> sum(bx);
    //std::vector<int> rowp(nnz, 0);
    double eps;

    if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecDouble)
    {
        eps = 1e-12;
    }
    else if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecFloat)
    {
        eps = 1e-7;
    }
    else
    {
        eps = 1e-12;
    }

    if (TConfig::memSpace == AMGX_device)
    {
        amgx::thrust::device_vector<int> k_d(1, 0);
        auto* k = amgx::thrust::raw_pointer_cast(k_d.data());
        auto* col_indices = A->col_indices.raw();
        auto* row_offsets = A->row_offsets.raw();
        auto* values = A->values.raw();

        fprintf(fout, "Check whether the sparse matrix is diagonal dominate on every row for matrix: A %8dx%8d, nnz %8d, block %2dx%2d\n", A->get_num_rows(), A->get_num_cols(), nnz, bx, by);

        amgx::thrust::for_each_n(thrust::device, amgx::thrust::counting_iterator<int>(0), A->get_num_rows(), [=] __device__ (int i)
        {
            PODTypeA sum[8];

            for (int m = 0; m < bx; m++) 
            { 
                sum[m] = 0.; 
            }

            for (int j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            {
                for (int m = 0; m < bx; m++)
                {
                    for (int n = 0; n < by; n++)
                    {
                        if ((col_indices[j] == i) && (m == n)) 
                        {
                            sum[m] += types::util<ValueTypeA>::abs(values[bs * j + m * by + n]);
                        }
                        else 
                        { 
                            sum[m] -= types::util<ValueTypeA>::abs(values[bs * j + m * by + n]);
                        }
                    }
                }
            }

            for (int m = 0; m < bx; m++)
            {
                if (sum[m] < -eps) 
                {  
                    atomicAdd(&k[0], 1); 
                }
            }
        });

        std::stringstream ss;
        ss << "Percentage of the diagonal-dominant rows is " << 100.0 * (num_rows * bx - k_d[0]) / num_rows << "%" << std::endl;
        amgx_output(ss.str().c_str(), ss.str().length());
    }
    else if (TConfig::memSpace == AMGX_host)
    {
        fprintf(fout, "Check whether the sparse matrix is diagonal dominate on every row for matrix: A %8dx%8d, nnz %8d, block %2dx%2d\n", A->get_num_rows(), A->get_num_cols(), nnz, bx, by);
        //std::cout<<"Check whether the sparse matrix is diagonal on every row for matrix: A: "<<num_rows<<"x"<<num_rows<<", nnz: "<<nnz<<", block: "<<bx<<"x"<<by<<std::endl;
        //for (int i=0;i < num_rows;i++)
        //  for (int j=A->row_offsets[i];j<A->row_offsets[i+1];j++)
        //      for (int m=0; m<by; m++)rowp[bs*j+m] = i;

        for (int i = 0; i < num_rows; i++)
        {
            for (int m = 0; m < bx; m++) { sum[m] = 0.; }

            for (int j = A->row_offsets[i]; j < A->row_offsets[i + 1]; j++)
            {
                for (int m = 0; m < bx; m++)
                    for (int n = 0; n < by; n++)
                        if ((A->col_indices[j] == i) && (m == n)) {sum[m] += types::util<ValueTypeA>::abs(A->values[bs * j + m * by + n]);}
                        else {sum[m] -= types::util<ValueTypeA>::abs(A->values[bs * j + m * by + n]);}
            }

            for (int m = 0; m < bx; m++)
                if (sum[m] < -eps) { ++k; }
        }

        fprintf(fout, "Percentage of the diagonal-dominant rows is  %8f %% \n", 100.0 * (num_rows * bx - k) / num_rows);
        std::cout << "Percentage of the diagonal-dominant rows is " << 100.0 * (num_rows * bx - k) / num_rows << "%" << std::endl;
    }
}

template <class AMatrix, bool complex>
struct Z_matrix_check;

template <class AMatrix>
struct Z_matrix_check<AMatrix, true>
{
    static bool check(const AMatrix *A, FILE *fout) { FatalError("Z matrix is not defined for complex field", AMGX_ERR_NOT_IMPLEMENTED);}
};


template <class AMatrix>
struct Z_matrix_check<AMatrix, false>
{
    static bool check(const AMatrix *A, FILE *fout)
    {
        bool isZmatrix = true;
        const int num_rows = A->get_num_rows();
        const int bx = A->get_block_dimx();
        const int by = A->get_block_dimy();
        const int nnz = A->get_num_nz();
        const int bs = bx * by;
        int *positive_diag = new int[bx];
        int *zero_diag = new int[bx];
        int *pos_off_diag = new int[bx];
        int pos_sum = 0;

        for (int m = 0; m < bx; m++)
        {
            positive_diag[m] = 0;
            zero_diag[m]     = 0;
            pos_off_diag[m]  = 0;
        }

        const typename AMatrix::index_type  *A_row_offsets_ptr = A->row_offsets.raw();
        const typename AMatrix::index_type  *A_column_indices_ptr = A->col_indices.raw();
        const typename AMatrix::value_type *A_values_ptr = A->values.raw();
        const typename AMatrix::index_type *A_dia_ptr = A->diag.raw();
        double eps;

        if (types::PODTypes<typename AMatrix::value_type>::vec_prec == AMGX_vecDouble)
        {
            eps = 1e-12;
        }
        else if (types::PODTypes<typename AMatrix::value_type>::vec_prec == AMGX_vecFloat)
        {
            eps = 1e-7;
        }
        else
        {
            eps = 1e-12;
        }

        fprintf(fout, "Check whether the sparse matrix is Z-matrix for matrix: A %8dx%8d, nnz %8d, block %2dx%2d\n", A->get_num_rows(), A->get_num_cols(), nnz, bx, by);
        //std::cout<<"Check whether off-diagonal pelement of the sparse matrix is negtive for matrix: A: "<<num_rows<<"x"<<num_rows<<", nnz: "<<nnz<<", block: "<<bx<<"x"<<by<<std::endl;

        for (int i = 0; i < num_rows; i++)
        {
            for (int j = A->row_offsets[i]; j < A->row_offsets[i + 1]; j++)
            {
                int k = A->col_indices[j];

                if (j == i)
                {
                    for (int m = 0; m < bx; m++)
                    {
                        if (A->values[j * bs + m * by + m] > eps) {positive_diag[m] ++;}
                        else if (A->values[j * bs + m * by + m] > -eps) {zero_diag[m]++;}

                        for (int n = 0; n < by; n ++)
                            if ((m != n) && (A->values[j * bs + m * by + n] > eps)) { pos_off_diag[m] ++; }
                    }
                }
                else
                {
                    for (int m = 0; m < bx; m++)
                    {
                        for (int n = 0; n < by; n ++)
                            if (A->values[j * bs + m * by + n] > eps) { pos_off_diag[m] ++; }
                    }
                }
            }
        }

        std::cout << std::endl << "Percentage of the positive diagonal element is ";

        for (int m = 0; m < bx; m++)
        {
            std::cout << "Block " << m << " is " << 100 * positive_diag[m] / num_rows << "% " << "\t";
        }

        std::cout << std::endl << "Number of the zero diagonal element is  ";

        for (int m = 0; m < bx; m++)
        {
            std::cout << "Block " << m << " is " << 100 * zero_diag[m] << "\t";
        }

        for (int m = 0; m < bx; m++)
        {
            pos_sum += pos_off_diag[m];
        }

        std::cout << std::endl << "Percentage of the positive off-diagonal diagonal element is " << 100.0 * pos_sum / (nnz - num_rows * bx) << "%" << std::endl;

        for (int m = 0; m < bx; m++)
        {
            if (positive_diag[m] < num_rows) { isZmatrix = false; }
        }

        if (pos_sum < (nnz - num_rows * bx)) { isZmatrix = false; }

        delete[] positive_diag;
        delete[] zero_diag;
        delete[] pos_off_diag;
        return isZmatrix;
    }
};

template <class T_Config>
bool MatrixAnalysis<T_Config>::check_Z_matrix()
{
    if (TConfig::memSpace == AMGX_device)
    {
        FatalError("Device version not implemented", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        if (A == NULL)
        {
            FatalError("MatrixAnalisys: A is not initialized", AMGX_ERR_BAD_PARAMETERS);
        }
        else
        {
            return Z_matrix_check<Matrix<TConfig>, types::util<ValueTypeA>::is_complex>::check(this->A, fout);
        }
    }
}

template <class T_Config>
void MatrixAnalysis<T_Config>::draw_matrix_connection()
{
    //int idx_i, idx_j;
    const int N = A->get_num_rows();
    const int bx = A->get_block_dimx();
    const int by = A->get_block_dimy();
    const int bs = bx * by;
    double eps;
    int dim = 3;

    if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecDouble)
    {
        eps = 1e-12;
    }
    else if (types::PODTypes<ValueTypeA>::vec_prec == AMGX_vecFloat)
    {
        eps = 1e-7;
    }
    else
    {
        eps = 1e-12;
    }

    //if ((idx_i > bx)||(idx_j > by)) FatalError("idx_i/idx_j is larger than the block size.", AMGX_ERR_BAD_PARAMETERS);
    if (geo_x == NULL) { FatalError("NO geometry input Should call load_geometry(Vector<TConfig>* geox, Vector<TConfig>* geoy, Vector<TConfig>* geoz ).", AMGX_ERR_BAD_PARAMETERS); }

    if (geo_z == NULL) { dim = 2; }

    int num_line = 0;
    std::vector< std::vector<int> > bd_idx(bx);

    for (int i = 0; i < bx; i++)
    {
        bd_idx[i].resize(N);
    }

    bool *is_diag = new bool[bx];

    for (int i = 0; i < N; i++)
    {
        for (int m = 0; m < bx; m++) { is_diag[m] = true; }

        for (int j = A->row_offsets[i]; j < A->row_offsets[i + 1]; j++)
        {
            int k = A->col_indices[j];

            if (i == k)
            {
                for (int m = 0; m < bx; m++)
                {
                    if ((types::util<ValueTypeA>::abs(A->values[j * bs + m * by + m]) > 1 + eps) || (types::util<ValueTypeA>::abs(A->values[j * bs + m * by + m]) < 1 - eps)) { is_diag[m] = false; }

                    for (int n = 0; n < by; n++)
                        if ((m != n) && (types::util<ValueTypeA>::abs(A->values[j * bs + m * by + n]) > eps)) { is_diag[m] = false; }
                }
            }
            else
            {
                for (int m = 0; m < bx; m++)
                    for (int n = 0; n < by; n++)
                        if ((types::util<ValueTypeA>::abs(A->values[j * bs + m * by + n]) > eps) ) { is_diag[m] = false; }
            }
        }

        for (int m = 0; m < bx; m++)
            if (is_diag[m]) { bd_idx[m][i] = 1;}
            else {bd_idx[m][i] = 0;}
    }

    std::cout << "number of point: " << (int) geo_x->size() << " row: " << b->size() << "size of diag: " << A->diag.size() << std::endl;

    for (int i = 0; i < N; i++)
    {
        for (int j = A->row_offsets[i]; j < A->row_offsets[i + 1]; j++)
        {
            int k = A->col_indices[j];

            if (i == k)
            {
                for (int m = 0; m < bx; m++)
                    if (bd_idx[m][k] == 1)
                    {
                        for (int n = 0; n < bx; n++)
                        {
                            if (m != n)
                            {
                                (*b)[k * bx + n] = (*b)[k * bx + n] - A->values[j * bs + n * by + m] * (*b)[i * bx + m]; //A->values[j*bs+n*by+m] = 0.0;
                            }
                        }
                    }
            }
            else
            {
                for (int m = 0; m < bx; m++)
                    if (bd_idx[m][k] == 1)
                    {
                        for (int n = 0; n < bx; n++)
                        {
                            (*b)[k * bx + n] = (*b)[k * bx + n] - A->values[j * bs + n * by + m] * (*b)[i * bx + m]; //A->values[j*bs+n*by+m] = 0.0;
                        }
                    }
            }
        }
    }

    for (int i = 0; i < A->get_num_rows(); i++)
    {
        for (int j = A->row_offsets[i]; j < A->row_offsets[i + 1]; j++)
        {
            int k = A->col_indices[j];

            if ((bd_idx[0][k] == 0) && (types::util<ValueTypeA>::abs(A->values[j * bs]) > eps)) { num_line++; }
        }
    }

    //Part 1: Header
    fprintf(fout, "# vtk DataFile Version 3.0\n");
    //Part 2: Title
    fprintf(fout, "show matrix connection by edges\n");
    //Part 3: Data Type ASCII/BINARY
    fprintf(fout, "ASCII\n\n");
    //Part 4: Geometry/Topology: STRUCTURED_POINTS/STRUCTURED_GRID/UNSTRUCURED_GRID/POLYDATA/RECTILINEAR_GRID/FIELD
    fprintf(fout, "DATASET POLYDATA\n");
    fprintf(fout, "POINTS %8d double\n", (int) geo_x->size());

    for (int i = 0; i < geo_x->size(); i++)
    {
        fprintf(fout, "%8g %8g ", (double)(*geo_x)[i], (double)(*geo_y)[i]);

        if (dim == 3) {fprintf(fout, " %8g \n", (double)(*geo_z)[i]);}
        else {fprintf(fout, " %8f \n", 0.0);}
    }

    std::cout << "number of points: " << (int) geo_x->size() << " row: " << A->get_num_rows() << std::endl;
    fprintf(fout, "LINES %8d %8d\n", num_line, 3 * num_line);

    for (int i = 0; i < A->get_num_rows(); i++)
    {
        for (int j = A->row_offsets[i]; j < A->row_offsets[i + 1]; j++)
        {
            int k = A->col_indices[j];

            //if ((A->values[j*bs+bs-1] > eps) || (A->values[j*bs+bs-1] < -eps)) fprintf(fout,"%8d %8d %8d\n", 2, i, k);
            if ((bd_idx[0][k] == 0) && (types::util<ValueTypeA>::abs(A->values[j * bs]) > eps))
            {
                fprintf(fout, "%8d %8d %8d\n", 2, i, k);
            }
        }
    }

    //Part 5: Dataset attributes.The number of data items n of each type must match the number of points or cells in the dataset. (If type is FIELD, point and cell data should be omitted.)
    fprintf(fout, "\n POINT_DATA %8d\n", N);

    for (int m = 0; m < bx; m++)
    {
        fprintf(fout, "SCALARS boundary_%d double\n LOOKUP_TABLE default\n", m);

        for (int i = 0; i < N; i++)
        {
            fprintf(fout, " %d \n", bd_idx[m][i]);
        }
    }

    delete[] is_diag;
}


template<class T_Config>
float MatrixAnalysis<T_Config>::aggregatesQuality(typename Matrix<T_Config>::IVector &aggregates, DevVectorFloat &edge_weights)
{
    typedef TemplateConfig<AMGX_host, T_Config::vecPrec, T_Config::matPrec, T_Config::indPrec> TConfig_h;
    Matrix<TConfig_h> Ah = *A;
    typename Matrix<TConfig_h>::IVector aggregates_h;
    aggregates_h.copy(aggregates);
    float score = 0.;

    for ( int i = 0; i < Ah.get_num_rows(); i++ )
    {
        for ( int j = Ah.row_offsets[i]; j < Ah.row_offsets[i + 1]; j++ )
        {
            int k = Ah.col_indices[j];

            if ( k != i && aggregates_h[k] == aggregates_h[i])
            {
                score += edge_weights[j];
            }
        }
    }

    return score;
}

template<class T_Config>
void MatrixAnalysis<T_Config>::aggregatesQuality2(const typename Matrix<T_Config>::IVector &aggregates, int num_aggregates, const Matrix<T_Config> &Aorig)
{
    typedef TemplateConfig<AMGX_host, T_Config::vecPrec, T_Config::matPrec, T_Config::indPrec> TConfig_h;
    typename Matrix<TConfig_h>::IVector aggs = aggregates;
    int lvl = A->template getParameter <int> ("level");
    Matrix<TConfig_h> Ah = *A;
    Matrix<TConfig_h> Ahorig = Aorig;
    std::vector<int> agg_cnt(aggs.size(), 0);
    // counters
    int max_nnz_per_row = -1, singletons_number = 0, empty_rows = 0, max_edges_in_aggregate = 0;
    double avg_nnz_per_row, max_nnz_variance;
    ValueTypeA avg_nnz_variance = types::util<ValueTypeA>::get_zero(),
               avg_nnz_sum = types::util<ValueTypeA>::get_zero();
    unsigned long long int nnz_num2;
    avg_nnz_per_row = (double)( Ah.get_num_nz() - (Ah.hasProps(DIAG) ? 0 : Ah.get_num_rows()) ) / Ah.get_num_rows();

    //avg_nnz_per_row = (double)( Ah.values.size()/Ah.get_block_size() - Ah.get_num_rows() ) / Ah.get_num_rows();
    for (int row = 0; row < Ah.get_num_rows(); row++)
    {
        int start_co = Ah.row_offsets[row];
        int end_co = Ah.row_offsets[row + 1];

        if (start_co == end_co)
        {
            empty_rows++;
        }

        max_nnz_per_row = std::max(end_co - start_co - (Ah.hasProps(DIAG) ? 0 : 1), max_nnz_per_row);
        ValueTypeA avg_nnz = types::util<ValueTypeA>::get_zero(), avg_nnz2 = types::util<ValueTypeA>::get_zero(), nnz_var = types::util<ValueTypeA>::get_zero();
        int nnz_cnt = 0;

        for (int co = start_co; co < end_co; co++)
        {
            int col = Ah.col_indices[co];

            if (col != row)
            {
                avg_nnz = avg_nnz + Ah.values[ co * Ah.get_block_size() ];
                avg_nnz2 = avg_nnz2 + (Ah.values[ co * Ah.get_block_size() ]) * (Ah.values[ co * Ah.get_block_size() ]);
                nnz_cnt ++;
            }
        }

        avg_nnz  = nnz_cnt > 0 ? (avg_nnz / (double)nnz_cnt) : types::util<ValueTypeA>::get_zero();
        avg_nnz2 = nnz_cnt > 0 ? (avg_nnz2 / (double)nnz_cnt) : types::util<ValueTypeA>::get_zero();
        avg_nnz_sum = avg_nnz_sum + avg_nnz; // avg_nnz normalized by the num of nnz
        nnz_var = avg_nnz2 - avg_nnz * avg_nnz;
        avg_nnz_variance = avg_nnz_variance + nnz_var;
        max_nnz_variance = max_nnz_variance < types::util<ValueTypeA>::abs(nnz_var) ? types::util<ValueTypeA>::abs(nnz_var) : max_nnz_variance;
        nnz_num2 += nnz_cnt * nnz_cnt;
    }

    avg_nnz_sum = avg_nnz_sum / (double)Ah.get_num_rows();
    avg_nnz_variance = avg_nnz_variance / (double)Ah.get_num_rows();
    double nnz_per_row_var = (double)nnz_num2 / Ah.get_num_rows() - avg_nnz_per_row * avg_nnz_per_row;

    for (unsigned int i = 0; i < aggs.size(); i++)
    {
        agg_cnt[aggs[i]]++;
    }

    singletons_number = (int)(std::count(agg_cnt.begin(), agg_cnt.end(), 1));
    max_edges_in_aggregate = *(std::max_element(agg_cnt.begin(), agg_cnt.end()));
    std::stringstream ss;
    agg_cnt.resize(aggs.size());
    std::fill(agg_cnt.begin(), agg_cnt.end(), 0);
    std::vector< std::vector<int> > agg_list(aggs.size());

    for (unsigned int agg = 0; agg < aggs.size();  agg++)
    {
        agg_list[aggs[agg]].push_back(agg);
    }

    for (unsigned int agg = 0; agg < aggs.size();  agg++)
        if (agg_list[agg].size() > 0)
        {
            std::vector<int> clusters(agg_list[agg].size());
            thrust_wrapper::sequence<AMGX_host>(clusters.begin(), clusters.end());

            for (int iter = 0; iter < clusters.size(); iter++)
            {
                //if (Ah.get_num_rows() < 20) printf("Processing aggregate ")
                for (int edge_id = (int)(agg_list[agg].size()) - 1; edge_id >= 0 ; edge_id--)
                {
                    int edge = agg_list[agg][edge_id];
                    int max_id = clusters[edge_id];

                    for (int ro = Ahorig.row_offsets[edge]; ro < Ahorig.row_offsets[edge + 1]; ro++)
                    {
                        std::vector<int>::iterator finding = std::find(agg_list[agg].begin(), agg_list[agg].end(), Ahorig.col_indices[ro]);

                        if (finding != agg_list[agg].end())
                        {
                            max_id = std::max(max_id, clusters[finding - agg_list[agg].begin()]);
                        }
                    }

                    clusters[edge_id] = max_id;
                }
            }

            std::set<int> clusters_counter (clusters.begin(), clusters.end());
            agg_cnt[agg] = clusters_counter.size();
        }

    cudaCheckError();
    int max_uncon = *(std::max_element(agg_cnt.begin(), agg_cnt.end())) - 1;
    int num_uncon = num_aggregates - (int)(std::count(agg_cnt.begin(), agg_cnt.end(), 1));

    if (lvl == 1)
    {
        ss << "\nMatrix aggregation and galerkin product information:\n";
        ss << std::setw(4) << "LVL" << std::setw(10) << "from rows" << std::setw(10) << "to rows" << std::setw(12) << "empty rows" << std::setw(20) << "max nodes in aggr" << std::setw(15) << "singletons#" \
           << std::setw(18) << "max nnz# per row" << std::setw(18) << "avg nnz# per row" << std::setw(18) << "nnz# per row var" << std::setw(24) << "agg# with uncon nodes" << std::setw(18) << "max# uncon nodes" << std::endl;
        ss << "   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    }

    ss << std::setw(4) << lvl << std::setw(10) << Ahorig.get_num_rows() << std::setw(10) << Ah.get_num_rows() << std::setw(12) << empty_rows << std::setw(20) << max_edges_in_aggregate << std::setw(15) << singletons_number \
       << std::setw(18) << max_nnz_per_row << std::setw(18) << avg_nnz_per_row << std::setw(18) << nnz_per_row_var << std::setw(24) << num_uncon << std::setw(18) << max_uncon << std::endl;
    amgx_printf("%s", ss.str().c_str());
    /*printf("Matrix with %d number of rows, level %d\nNumber of empty rows is %d.\nMaximum edges in the aggregate is %d.\nNumber of singletons is %d.\n",
      Ah.get_num_rows(), empty_rows, max_edges_in_aggregate, singletons_number);
    printf("Max nnz per row: %d\nAverage nnz per row: %f\nMax nnz variance: %f\nAverage nnz variance: %f\nAverage nnz sum: %f\n",
      max_nnz_per_row, avg_nnz_per_row, max_nnz_variance, avg_nnz_variance, avg_nnz_sum); */
    fflush(stdout);
    return;
}

template<class T_Config>
void MatrixAnalysis<T_Config>::visualizeAggregates(typename Matrix<TConfig>::IVector &aggregates)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef Vector< TConfig_h > Vector_h;
    Matrix<TConfig_h> Ah = *A;
    typename Matrix<TConfig_h>::IVector aggregates_h;
    aggregates_h.copy(aggregates);
    int dim = 0;

    if ( Ah.hasParameter("geo.x") && Ah.hasParameter("geo.y"))
    {
        if (Ah.hasParameter("geo.z"))
        {
            dim = 3;
        }
        else
        {
            dim = 2;
        }
    }
    else
    {
        std::cout << "Cannot visualize data, no geometry information attached" << std::endl;
        return;
    }

    PODVecHost *geo_x = Ah.template getParameterPtr< PODVecHost >("geo.x");
    PODVecHost *geo_y = Ah.template getParameterPtr< PODVecHost >("geo.y");
    PODVecHost *geo_z = Ah.template getParameterPtr< PODVecHost >("geo.z");
    // Count the number of lines to create
    int num_line = 0;

    for (int i = 0; i < Ah.get_num_rows(); i++)
    {
        for (int j = Ah.row_offsets[i]; j < Ah.row_offsets[i + 1]; j++)
        {
            int k = Ah.col_indices[j];

            if (k != i && aggregates_h[k] == aggregates_h[i] )
            {
                num_line++;
            }
        }
    }

    std::stringstream file_name;
    file_name << "aggregates_" << Ah.get_num_rows() << ".vtk";
    FILE *fout;
    fout = fopen(file_name.str().c_str(), "w");
    //Part 1: Header
    fprintf(fout, "# vtk DataFile Version 3.0\n");
    //Part 2: Title
    fprintf(fout, "show matrix connection by edges\n");
    //Part 3: Data Type ASCII/BINARY
    fprintf(fout, "ASCII\n\n");
    //Part 4: Geometry/Topology: STRUCTURED_POINTS/STRUCTURED_GRID/UNSTRUCURED_GRID/POLYDATA/RECTILINEAR_GRID/FIELD
    fprintf(fout, "DATASET POLYDATA\n");
    fprintf(fout, "POINTS %8d double\n", (int) geo_x->size());

    for (int i = 0; i < geo_x->size(); i++)
    {
        fprintf(fout, "%8g %8g ", (double)(*geo_x)[i], (double)(*geo_y)[i]);

        if (dim == 3) {fprintf(fout, " %8g \n", (double)(*geo_z)[i]);}
        else {fprintf(fout, " %8f \n", 0.0);}
    }

    std::cout << "number of points: " << (int) geo_x->size() << " row: " << A->get_num_rows() << std::endl;
    fprintf(fout, "LINES %8d %8d\n", num_line, 3 * num_line);

    for (int i = 0; i < Ah.get_num_rows(); i++)
    {
        for (int j = Ah.row_offsets[i]; j < Ah.row_offsets[i + 1]; j++)
        {
            int k = Ah.col_indices[j];

            if (k != i && aggregates_h[k] == aggregates_h[i])
            {
                fprintf(fout, "%8d %8d %8d\n", 2, i, k);
            }
        }
    }

    std::cout << "done writing file" << std::endl;
    fclose(fout);
    std::cout << "done closing file" << std::endl;
}








/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MatrixAnalysis<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}//end namespace amgx
