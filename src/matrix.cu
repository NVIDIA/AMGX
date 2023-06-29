/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <matrix.h>
#include <cutil.h>
#include <util.h>
#include <solvers/solver.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <vector_thrust_allocator.h>
#include <cusp/detail/format_utils.h>
#include <permute.h>
#include <multiply.h>
#include <amgx_types/util.h>
#include <algorithm>

template<typename T>
struct row_length : public amgx::thrust::unary_function<T, T>
{
    __host__ __device__ T operator()(const T &x) const
    {
        const T *next_ptr = &x;
        next_ptr++;
        return (*next_ptr) - x;
    }
};

namespace amgx
{

__global__ void computeRowOffsetsKernel(INDEX_TYPE num_rows, INDEX_TYPE num_nz, const INDEX_TYPE *row_indices, INDEX_TYPE *row_offsets )
{
    //one thread per non-zero
    int nz = blockIdx.x * blockDim.x + threadIdx.x;

    if (nz == 0)
    {
        row_offsets[0] = 0;
        row_offsets[num_rows] = num_nz;
    }

    while (nz < num_nz - 1)
    {
        int row = row_indices[nz];
        int next_row = row_indices[nz + 1];

        while (row < next_row) //this loop should probably always execute once, but I'm making it a loop just in case...
        {
            row_offsets[++row] = nz + 1;
        }

        nz += blockDim.x * gridDim.x;
    }
}

__global__ void computeRowIndicesKernel(INDEX_TYPE num_rows, const INDEX_TYPE *row_offsets, INDEX_TYPE *row_indices )
{
    //one warp per row //possible optimziation:  multiple warps per row
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / AMGX_WARP_SIZE;
    int warp_id = threadIdx.x % AMGX_WARP_SIZE;

    while (row < num_rows)
    {
        int start = row_offsets[row] + warp_id;
        int end = row_offsets[row + 1];

        for (int nz = start; nz < end; nz += AMGX_WARP_SIZE)
        {
            row_indices[nz] = row;
        }

        row += blockDim.x * gridDim.x / AMGX_WARP_SIZE;
    }
}

__global__ void computeDiagonalKernelCOO(INDEX_TYPE num_nz, INDEX_TYPE *row_indices, INDEX_TYPE *col_indices, INDEX_TYPE *diag, INDEX_TYPE *diag_end_offsets)
{
    //BLOCKY*BLOCKX threads per nz
    INDEX_TYPE nz = (blockIdx.x * blockDim.x + threadIdx.x);

    while (nz < num_nz)
    {
        INDEX_TYPE row = row_indices[nz];
        INDEX_TYPE col = col_indices[nz];

        if (row == col)
        {
            //copy block to diag
            diag[row] = nz;
            diag_end_offsets[row] = nz + 1;
        }

        nz += blockDim.x * gridDim.x;
    }
}

__global__ void computeDiagonalKernelCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE *diag, INDEX_TYPE null_index, INDEX_TYPE *diag_end_offsets)
{
    INDEX_TYPE row = (blockIdx.x * blockDim.x + threadIdx.x);

    while (row < num_rows)
    {
        int nz = row_offsets[row];
        int last_nz = row_offsets[row + 1];

        while (nz < last_nz)
        {
            int col = col_indices[nz];

            if (row == col)
            {
                diag[row] = nz;
                diag_end_offsets[row] = nz + 1;
                break;
            }

            nz++;
        }

        row += blockDim.x * gridDim.x;
    }
}

__global__ void computeColorOffsetsKernelCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, const INDEX_TYPE *row_colors, INDEX_TYPE *smaller_color_offsets, INDEX_TYPE *larger_color_offsets, INDEX_TYPE *diag)
{
    INDEX_TYPE row = (blockIdx.x * blockDim.x + threadIdx.x);

    while (row < num_rows)
    {
        int my_color = row_colors[row];
        int nz = row_offsets[row];
        int last_nz = row_offsets[row + 1];
        int location_small = -1;
        int location_large = -1;

        while (nz < last_nz)
        {
            int col = col_indices[nz];

            if (row_colors[col] >= my_color && location_small == -1)
            {
                location_small = nz;
            }

            if (row_colors[col] > my_color && location_large == -1)
            {
                location_large = nz;
                break;
            }

            nz++;
        }

        if (location_large == -1) { location_large = last_nz + 1; }

        if (location_small == -1) { location_small = last_nz + 1; }

        larger_color_offsets[row]  = location_large;
        smaller_color_offsets[row] = location_small;
        row += blockDim.x * gridDim.x;
    }
}

__global__ void computeDiagonalKernelDiagProp (INDEX_TYPE num_rows, INDEX_TYPE num_nz, INDEX_TYPE *diag, INDEX_TYPE *diag_end_offsets)
{
    INDEX_TYPE r = (blockIdx.x * blockDim.x + threadIdx.x);

    while (r < num_rows)
    {
        diag[r] = num_nz + r;
        diag_end_offsets[r] = num_nz + r + 1;
        r += blockDim.x * gridDim.x;
    }
}

template <class T>
__global__ void reorderElements (INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *permutation, T *data, T *temp, INDEX_TYPE max_row, INDEX_TYPE blockSize)
{
    INDEX_TYPE rowId = blockIdx.x;

    while (rowId < num_rows)
    {
        INDEX_TYPE rowStart = row_offsets[rowId];
        INDEX_TYPE rowLen = row_offsets[rowId + 1] - rowStart;

        int i = threadIdx.x;

        //copy and reorder into temp storage
        while (i < rowLen * blockSize)
        {
            temp[max_row * blockIdx.x + i] = data[(permutation[rowStart + i / blockSize]) * blockSize + i % blockSize];
            i += blockDim.x;
        }

        __syncthreads();
        //copy back
        i = threadIdx.x;

        while (i < rowLen * blockSize)
        {
            data[rowStart * blockSize + i] = temp[max_row * blockIdx.x + i];
            i += blockDim.x;
        }

        rowId += gridDim.x;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Vector<TConfig> &v_ = const_cast<Vector<TConfig>&>(v);
    multiply(*this, v_, res, view);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Vector<TConfig> &v_ = const_cast<Vector<TConfig>&>(v);
    multiply(*this, v_, res, view);
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::print(char *f, char *s, int srows, int erows, int trank)
{
    typedef typename TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>::MatPrec ValueType;
    int rank = 0;
    int level = 0;
    char filename[1024];
    FILE *fid = NULL;
    int printRowsStart, printRowsEnd;
    int i, j, ii, xdim, ydim, tnnz;
    ValueType a;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    //check target rank
    if (rank == trank)
    {
        //check whether to ouput to stdout or file
        if (f == NULL)
        {
            fid = stdout;
        }
        else
        {
            level = this->amg_level_index;
#ifdef _WIN32
            _snprintf_s(filename, 1024, 1024, "%s_l%d_r%d.mtx", f, level, rank);
#else
            snprintf(filename, 1024, "%s_l%d_r%d.mtx", f, level, rank);
#endif
            fid = fopen(filename, "w");
        }

        cudaDeviceSynchronize();
        cudaCheckError();
        printRowsStart = (srows < 0) ? 0                    : srows;
        printRowsEnd  = (erows < 0) ? this->get_num_rows() : erows;
        tnnz = this->get_num_nz() * this->get_block_size();

        //matrix might have separate diagonal (need to accoutn for it in nnz)
        if (this->hasProps(DIAG, this->props))
        {
            //matrix might be non-square so take std::min of # of rows and cols
            tnnz += std::min(this->get_num_rows(), this->get_num_cols());
        }

        auto trafI = [&](auto const &I, auto const &i) { return I *  this->get_block_dimy() + i + 1; };
        auto trafJ = [&](auto const &J, auto const &j) { return J *  this->get_block_dimx() + j + 1; };

        fprintf(fid, "%%%%MatrixMarket matrix coordinate real general\n");
        fprintf(fid, "%% %s\n", s);
        fprintf(fid, "%d %d %d\n", this->get_num_rows() * this->get_block_dimx(), this->get_num_cols() * this->get_block_dimy(), tnnz);

        for (i = printRowsStart; i < printRowsEnd; i++)
        {
            for (ydim = 0; ydim < this->get_block_dimy(); ydim++)
            {
                if (this->hasProps(DIAG, this->props))
                {
                    if (i < std::min(this->get_num_rows(), this->get_num_cols()))
                    {
                        for (xdim = 0; xdim < this->get_block_dimx(); xdim++)
                        {
                            a = this->values[this->diag[i] * this->get_block_dimx() * this->get_block_dimy() + this->get_block_dimy() * ydim + xdim];
                            fprintf(fid, "%d %d ", trafI(i, ydim), trafI(i, xdim));
                            types::util<value_type>::fprintf(fid, "%20.16f", a);
                            fprintf(fid, "\n");
                        }
                    }
                }

                for (ii = this->row_offsets[i]; ii < this->row_offsets[i + 1]; ii++)
                {
                    j = this->col_indices[ii];

                    for (xdim = 0; xdim < this->get_block_dimx(); xdim++)
                    {
                        a = this->values[ii * this->get_block_dimx() * this->get_block_dimy() + this->get_block_dimy() * ydim + xdim];
                        fprintf(fid, "%d %d ", trafI(i, ydim), trafJ(j, xdim));
                        types::util<value_type>::fprintf(fid, "%20.16f", a);
                        fprintf(fid, "\n");
                    }
                }
            }
        }

        cudaDeviceSynchronize();
        cudaGetLastError();

        if (fid != stdout)
        {
            fclose(fid);
        }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::printToFile(char *f, char *s, int srows, int erows)
{
    int rank = 0;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    //notice that print will be called with different (target) rank on different on different ranks/processes
    this->print(f, s, srows, erows, rank);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::print(char *f, char *s, int srows, int erows, int trank)
{
    typedef typename TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec>::MatPrec ValueType;
    int rank = 0;
    int level = 0;
    char filename[1024];
    FILE *fid = NULL;
    int printRowsStart, printRowsEnd;
    int i, j, ii, xdim, ydim, tnnz;
    ValueType a;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    //check target rank
    if (rank == trank)
    {
        //check whether to ouput to stdout or file
        if (f == NULL)
        {
            fid = stdout;
        }
        else
        {
            level = this->amg_level_index;
#ifdef _WIN32
            _snprintf_s(filename, 1024, 1024, "%s_r%d_l%d.mtx", f, rank, level);
#else
            snprintf(filename, 1024, "%s_r%d_l%d.mtx", f, rank, level);
#endif
            fid = fopen(filename, "w");
        }

        cudaDeviceSynchronize();
        cudaCheckError();
        printRowsStart = (srows < 0) ? 0                    : srows;
        printRowsEnd  = (erows < 0) ? this->get_num_rows() : erows;
        tnnz = this->get_num_nz() * this->get_block_size();

        //matrix might have separate diagonal (need to accoutn for it in nnz)
        if (this->hasProps(DIAG, this->props))
        {
            //matrix might be non-square so take min of # of rows and cols
            tnnz += std::min(this->get_num_rows(), this->get_num_cols()) * this->get_block_size();
        }

        fprintf(fid, "%%%%MatrixMarket matrix coordinate real general\n");
        fprintf(fid, "%% %s\n", s);
        fprintf(fid, "%d %d %d\n", this->get_num_rows() * this->get_block_dimx(), this->get_num_cols() * this->get_block_dimy(), tnnz);

        auto trafI = [&](auto const &I, auto const &i) { return I *  this->get_block_dimy() + i + 1; };
        auto trafJ = [&](auto const &J, auto const &j) { return J *  this->get_block_dimx() + j + 1; };

        for (i = printRowsStart; i < printRowsEnd; i++)
        {
            for (ydim = 0; ydim < this->get_block_dimy(); ydim++)
            {
                if (this->hasProps(DIAG, this->props))
                {
                    if (i < std::min(this->get_num_rows(), this->get_num_cols()))
                    {
                        for (xdim = 0; xdim < this->get_block_dimx(); xdim++)
                        {
                            a = this->values[this->diag[i] * this->get_block_dimx() * this->get_block_dimy() + this->get_block_dimy() * ydim + xdim];
                            fprintf(fid, "%d %d ", trafI(i, ydim), trafI(i, xdim));
                            types::util<value_type>::fprintf(fid, "%20.16f", a);
                            fprintf(fid, "\n");
                        }
                    }
                }

                for (ii = this->row_offsets[i]; ii < this->row_offsets[i + 1]; ii++)
                {
                    j = this->col_indices[ii];

                    for (xdim = 0; xdim < this->get_block_dimx(); xdim++)
                    {
                        a = this->values[ii * this->get_block_dimx() * this->get_block_dimy() + this->get_block_dimy() * ydim + xdim];
                        fprintf(fid, "%d %d ", i + 1, j + 1);
                        types::util<value_type>::fprintf(fid, "%20.16f", a);
                        fprintf(fid, "\n");
                    }
                }
            }
        }

        cudaDeviceSynchronize();
        cudaGetLastError();

        if (fid != stdout)
        {
            fclose(fid);
        }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::printToFile(char *f, char *s, int srows, int erows)
{
    int rank = 0;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    //notice that print will be called with different (target) rank on different on different ranks/processes
    this->print(f, s, srows, erows, rank);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > ::convert( const Matrix<TConfig> &mat, unsigned int new_props, int block_dimy, int block_dimx )
{
    if ( !mat.is_initialized() )
    {
        FatalError("Trying to convert from the uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    this->set_initialized(0);
    this->addProps(new_props);
    index_type block_size = block_dimx * block_dimy;
    index_type new_num_rows = ( mat.get_num_rows() * mat.block_dimy + block_dimy - 1 ) / block_dimy;
    index_type new_num_cols = ( mat.get_num_cols() * mat.block_dimx + block_dimx - 1 ) / block_dimx;
    MVector new_values;
    IVector new_col_indices;
    index_type new_num_nnz = 0;
    IVector new_row_indices;
    IVector new_row_offsets;
    MVector new_dia_values;
    new_dia_values.resize( new_num_rows * block_size, types::util<value_type>::get_zero() );

    if ( mat.hasProps(COO) )
    {
        std::map< std::pair<index_type, index_type>, index_type> ind;

        for ( index_type i = 0; i < mat.get_num_nz(); i++ )
            for ( index_type loc_row = 0; loc_row < mat.block_dimy; loc_row++ )
                for ( index_type loc_col = 0; loc_col < mat.block_dimx; loc_col++ )
                {
                    index_type in_row = mat.row_indices[i] * mat.block_dimy + loc_row;
                    index_type in_col = mat.col_indices[i] * mat.block_dimx + loc_col;
                    value_type in_val = mat.values[i * mat.block_size + loc_row * mat.block_dimx + loc_col];

                    if ( types::util<value_type>::is_zero(in_val) ) { continue; } // skip zero entries

                    index_type out_br = in_row / block_dimy;
                    index_type out_bc = in_col / block_dimx;
                    index_type out_lr = in_row % block_dimy;
                    index_type out_lc = in_col % block_dimx;

                    if ( ind.find( std::pair<index_type, index_type>( out_br, out_bc ) ) == ind.end() )
                    {
                        // allocate a new block
                        ind.insert( std::pair< std::pair<index_type, index_type>, index_type>( std::pair<index_type, index_type>( out_br, out_bc ), new_num_nnz ) );

                        if (out_br != out_bc || !this->hasProps(DIAG))
                        {
                            new_row_indices.push_back( out_br );
                            new_col_indices.push_back( out_bc );
                            new_num_nnz++;

                            for ( int b = 0; b < block_size; b++ )
                            {
                                new_values.push_back( types::util<value_type>::get_zero() );
                            }
                        }
                    }

                    if ( out_br != out_bc || !this->hasProps(DIAG) )
                    {
                        new_values[ ind[std::pair<index_type, index_type>( out_br, out_bc )] * block_size + out_lr * block_dimx + out_lc ] = in_val;
                    }
                    else
                    {
                        new_dia_values[ out_br * block_size + out_lr * block_dimx + out_lc ] = in_val;
                    }
                }
    } // ( mat.hasProps(COO) )
    else if ( mat.hasProps(CSR))
    {
        new_num_nnz = 0;
        //MVector new_dia_values;
        //new_dia_values.resize( new_num_rows*block_size, 0.0 );
        new_row_offsets.resize( new_num_rows + 1 );

        // process each output block row
        for ( index_type i = 0; i < new_num_rows; i++ )
        {
            new_row_offsets[i] = new_num_nnz;
            // count non zero column blocks
            IVector non_zero_blocks( new_num_cols );

            for ( index_type j = i * block_dimy; j < (i + 1) * block_dimy && j < mat.get_num_rows() * mat.block_dimy; j++ )
            {
                // input row block / local position
                index_type in_br = j / mat.block_dimy;
                index_type in_lr = j % mat.block_dimy;

                // loop through block columns
                for ( index_type r = mat.row_offsets[in_br]; r < mat.row_offsets[in_br + 1] + ( mat.hasProps(DIAG) ); r++ )
                {
                    index_type in_bc = ( r == mat.row_offsets[in_br + 1] ) ? in_br : mat.col_indices[r];

                    // loop through local columns
                    for ( index_type in_lc = 0; in_lc < mat.block_dimx; in_lc++ )
                    {
                        index_type in_col = in_bc * mat.block_dimx + in_lc;
                        index_type out_bc = in_col / block_dimx;
                        // fetch input entry value
                        value_type val = ( r == mat.row_offsets[in_br + 1] ) ?
                                         mat.values[mat.diag[in_br] * mat.block_size + in_lr * mat.block_dimx + in_lc] :
                                         mat.values[r * mat.block_size + in_lr * mat.block_dimx + in_lc];

                        if ( types::util<value_type>::is_zero(val) ) { continue; }  // skip zero entries

                        // mark non_zero column
                        non_zero_blocks[out_bc] = 1;
                    }
                }
            }

            // populate non zero column blocks
            for ( int bc = 0; bc < new_num_cols; bc++ )
                if ( non_zero_blocks[bc] != 0 )
                {
                    if ( i != bc || !this->hasProps(DIAG) )  // off-diagonal for DIAG
                    {
                        non_zero_blocks[bc] = new_num_nnz++;
                        new_col_indices.push_back( bc );

                        for ( int b = 0; b < block_size; b++ )
                        {
                            new_values.push_back( types::util<value_type>::get_zero() );
                        }
                    }
                }

            // fill non zero values
            for ( index_type j = i * block_dimy; j < (i + 1) * block_dimy && j < mat.get_num_rows() * mat.block_dimy; j++ )
            {
                // output row block/local position
                index_type out_br = j / block_dimy;
                index_type out_lr = j % block_dimy;
                // input row block/local position
                index_type in_br = j / mat.block_dimy;
                index_type in_lr = j % mat.block_dimy;

                // loop through block columns
                for ( index_type r = mat.row_offsets[in_br]; r < mat.row_offsets[in_br + 1] + ( mat.hasProps(DIAG) ); r++ )
                {
                    index_type in_bc = ( r == mat.row_offsets[in_br + 1] ) ? in_br : mat.col_indices[r];

                    // loop through local columns
                    for ( index_type in_lc = 0; in_lc < mat.block_dimx; in_lc++ )
                    {
                        index_type in_col = in_bc * mat.block_dimx + in_lc;
                        index_type out_bc = in_col / block_dimx;
                        index_type out_lc = in_col % block_dimx;
                        // fetch input entry value
                        value_type val = ( r == mat.row_offsets[in_br + 1] ) ?
                                         mat.values[mat.diag[in_br] * mat.block_size + in_lr * mat.block_dimx + in_lc] :
                                         mat.values[r * mat.block_size + in_lr * mat.block_dimx + in_lc];

                        if ( types::util<value_type>::is_zero(val) ) { continue; }  // skip zero entries

                        // write to new matrix
                        if ( out_br != out_bc || !this->hasProps(DIAG) )
                        {
                            new_values[ non_zero_blocks[out_bc] * block_size + out_lr * block_dimx + out_lc] = val;
                        }
                        else
                        {
                            new_dia_values[ out_br * block_size + out_lr * block_dimx + out_lc ] = val;
                        }
                    }
                }
            }
        } // for( i < new_num_rows )

        // fill extra diagonal for the last block
        int extra_start = ( mat.get_num_rows() * mat.block_dimy ) % block_dimy;

        if ( extra_start > 0 )
        {
            for ( int r = extra_start; r < block_dimy; r++ )
                if ( this->hasProps(DIAG) )
                {
                    new_dia_values[ (new_num_rows - 1) * block_size + r * block_dimx + r] = types::util<value_type>::get_one();
                }
                else
                {
                    new_values[ (new_num_nnz - 1) * block_size + r * block_dimx + r] = types::util<value_type>::get_one();
                }
        }

        new_row_offsets[new_num_rows] = new_num_nnz;
    } // ( mat.hasProps(CSR) )
    else
    {
        FatalError("Input matrix for conversion doesn't have COO or CSR format", AMGX_ERR_CONFIGURATION);
    }

    if ( this->hasProps(DIAG) )
    {
        new_values.insert(new_values.end(), new_dia_values.begin(), new_dia_values.end());
    }
    else
        for ( int b = 0; b < block_size; b++ )
        {
            new_values.push_back( types::util<value_type>::get_zero() );
        }

    this->resize( new_num_rows, new_num_cols, new_num_nnz, block_dimy, block_dimx );
    this->values.copy( new_values );
    this->values.set_block_dimx(block_dimx);
    this->values.set_block_dimy(block_dimy);
    this->col_indices.copy( new_col_indices );

    if ( mat.hasProps(COO) )
    {
        this->row_indices.copy( new_row_indices );
        this->props |= COO;
    }

    if ( mat.hasProps(CSR) )
    {
        this->row_offsets.copy( new_row_offsets );
        this->props |= CSR;
    }

    if (this->hasProps(COO) && this->row_indices.size() == 0)
    {
        this->row_indices.resize(new_num_nnz);
        computeRowIndices();
    }

    if (this->hasProps(CSR) && this->row_offsets.size() == 0)
    {
        this->row_offsets.resize(new_num_rows + 1);
        computeRowOffsets();
    }

    computeDiagonal();
    this->set_initialized(1);
}

template <class T_Config> class MatrixBase;

template<class T_Config>
AMGX_ERROR
MatrixBase<T_Config>::resize(index_type num_rows, index_type num_cols, index_type num_nz, int skipDiaCompute)
{
    if (this->is_initialized())
    {
        FatalError("Debug throw: resizing already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
    }

    {
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_nz = num_nz;

        if ( hasProps(DIAG) )
        {
            values.resize(num_nz * block_size + num_rows * block_size);
        }
        else
        {
            values.resize((num_nz + 1)*block_size);
            //thrust_wrapper::fill<T_Config::memSpace>(values.begin() + num_nz*block_size, values.end(), static_cast<value_type>(0.0));
        }

        diag.resize(num_rows);
        m_diag_end_offsets.resize(num_rows);
        col_indices.resize(num_nz);

        //if( props == NONE ) {props = CSR; props |= DIAG;}
        if ( hasProps(COO) ) { row_indices.resize(num_nz); }

        if ( hasProps(CSR) ) { row_offsets.resize(num_rows + 1); }

        m_seq_offsets.resize(num_rows + 1);
        thrust_wrapper::sequence<T_Config::memSpace>(m_seq_offsets.begin(), m_seq_offsets.end());
        cudaCheckError();

        if (!skipDiaCompute )
        {
            computeDiagonal();
        }
    }

    return AMGX_OK;
}





template<class T_Config>
void
MatrixBase<T_Config>::setupMatrix(Solver<T_Config> *outer_solver, AMG_Config &cfg, bool reuse_matrix_structure)
{
    // Check that matrix is initialized
    if (!this->is_initialized())
    {
        FatalError("Trying to setup from the uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    this->set_initialized(0);
    m_separation_interior = cfg.getParameter<ViewType>("separation_interior", "default");
    m_separation_exterior = cfg.getParameter<ViewType>("separation_exterior", "default");

    if (m_separation_interior > m_separation_exterior) { FatalError("Interior separation cannot be wider than the exterior separation", AMGX_ERR_CONFIGURATION); }

    // If latency hiding is disabled, the interior is overwritten
    if(!isLatencyHidingEnabled(cfg))
    {
        m_separation_interior = m_separation_exterior;
    }

    bool is_coloring_needed  = outer_solver->isColoringNeeded();

    if (!reuse_matrix_structure)
    {
        // Color the matrix since the structure has changed
        if ( is_coloring_needed )
        {
            // Get the scope of the solver that needs coloring
            std::string cfg_scope_for_coloring;
            outer_solver->getColoringScope(cfg_scope_for_coloring);
            this->colorMatrix(cfg, cfg_scope_for_coloring);
        }
    }

    // Set the matrix block format
    BlockFormat block_format = cfg.getParameter<BlockFormat>( "block_format", "default" );

    if ( this->getBlockFormat() != block_format )
    {
        this->setBlockFormat( block_format );
    }

    // Permute the values and the column indices (if necessary)
    bool reorder_cols_by_color = outer_solver->getReorderColsByColorDesired();
    bool insert_diagonal  = outer_solver->getInsertDiagonalDesired();

    if (reorder_cols_by_color)
    {
        if (reuse_matrix_structure) // Only permute the values
        {
            this->permuteValues();
        }
        else // Permute the values and the columns
        {
            this->reorderColumnsByColor(insert_diagonal);
            this->permuteValues();
        }
    }

    this->set_initialized(1);
    m_is_matrix_setup = true;
}

template<class T_Config>
bool MatrixBase<T_Config>::isLatencyHidingEnabled(AMG_Config &cfg)
{
    const int min_rows_latency_hiding =
        cfg.getParameter<int>("min_rows_latency_hiding", "default");

    // Test all partitions to check if they all fall below the threshold
    if (!is_matrix_singleGPU() && min_rows_latency_hiding >= 0)
    {
        const auto& nrows_per_part = manager->getNumRowsPerPart();

        // Look at all partitions to check whether the number of rows falls
        // below the user defined minimum
        for(auto& nrpp : nrows_per_part)
        {
            // If any partitions still have a large enough set of rows, 
            // continue latency hiding
            if(nrpp >= min_rows_latency_hiding)
            {
                return true;
            }
        }
    }

    return false;
}

template<class T_Config>
void
MatrixBase<T_Config>::reorderColumnsByColor(bool insert_diagonal)
{
    // If columns already reordered, don't reorder again
    if (this->m_cols_reordered_by_color)
    {
        return;
    }

    // Check if matrix is colored
    if (!this->hasProps(COLORING))
    {
        FatalError("Matrix must be colored in order to be reordered by colors. Try setting coloring_level=1 in the configuration file", AMGX_ERR_CONFIGURATION);
    }

    set_initialized(0);
    // Compute the row indices
    addProps(COO);
    set_allow_recompute_diag(false);
    index_type num_non_zeros = num_nz;

    if (hasProps(DIAG) && insert_diagonal)   // Diagonal stored separetely
    {
        num_non_zeros += num_rows;
    }

    // Append the diagonal if stored separately
    // The new matrix will have inside diagonal
    if (hasProps(DIAG) && insert_diagonal)
    {
        amgx::thrust::counting_iterator<int> first(0);
        amgx::thrust::counting_iterator<int> last = first + num_rows;
        // Create new row_indices with appended diagonal
        IVector new_row_indices(num_non_zeros);
        IVector new_col_indices(num_non_zeros);
        amgx::thrust::copy(row_indices.begin(), row_indices.end(), new_row_indices.begin());
        amgx::thrust::copy(first, last, new_row_indices.begin() + num_nz);
        // Create new col_indices with appended diagonal
        amgx::thrust::copy(col_indices.begin(), col_indices.end(), new_col_indices.begin());
        amgx::thrust::copy(first, last, new_col_indices.begin() + num_nz);
        row_indices.swap(new_row_indices);
        col_indices.swap(new_col_indices);
        new_row_indices.clear();
        new_row_indices.shrink_to_fit();
        new_col_indices.clear();
        new_col_indices.shrink_to_fit();
    }

    // Compute the color of every column
    IVector element_colors(num_non_zeros);
    amgx::thrust::copy(amgx::thrust::make_permutation_iterator(this->getMatrixColoring().getRowColors().begin(), col_indices.begin()),
                 amgx::thrust::make_permutation_iterator(this->getMatrixColoring().getRowColors().begin(), col_indices.end()),
                 element_colors.begin());
    // Compute the permutation vector by sorting by rows and columns
    m_values_permutation_vector.resize(num_non_zeros);
    thrust_wrapper::sequence<T_Config::memSpace>(m_values_permutation_vector.begin(), m_values_permutation_vector.end());
    cusp::detail::sort_by_row_and_column(row_indices, element_colors, m_values_permutation_vector);
    cudaCheckError();
    element_colors.clear();
    element_colors.shrink_to_fit();
    // Compute the new column indices sorted by color
    IVector new_column_indices(num_non_zeros);
    amgx::thrust::copy(amgx::thrust::make_permutation_iterator(col_indices.begin(), m_values_permutation_vector.begin()),
                 amgx::thrust::make_permutation_iterator(col_indices.begin(), m_values_permutation_vector.end()),
                 new_column_indices.begin());
    col_indices.swap(new_column_indices);
    new_column_indices.clear();
    new_column_indices.shrink_to_fit();

    if (hasProps(DIAG) && insert_diagonal)
    {
        // Change the number of nonzeros
        set_num_nz(num_non_zeros);
        values.resize( (num_non_zeros + 1)*this->get_block_size());
        this->m_is_permutation_inplace = false;
    }
    else
    {
        this->m_is_permutation_inplace = true;
    }

    if (hasProps(DIAG) && insert_diagonal)
    {
        delProps(DIAG);
        // Force recomputation of row offsets
        delProps(CSR);
    }

    // Compute row offsets if input matrix only had COO format or if diagonal was inserted
    addProps(CSR);
    // Recompute the diagonal
    set_allow_recompute_diag(true);
    computeDiagonal();
    // Compute the color offsets
    m_smaller_color_offsets.resize(this->get_num_rows());
    m_larger_color_offsets.resize(this->get_num_rows());
    computeColorOffsets();
    this->m_cols_reordered_by_color = true;
    set_initialized(1);
}

template<class T_Config>
void
MatrixBase<T_Config>::sortByRowAndColumn()
{
    this->set_initialized(0);
    // Add row_indices array
    this->addProps(COO);
    this->set_allow_recompute_diag(false);

    if (this->get_block_dimx() != 1 || this->get_block_dimy() != 1)
    {
        FatalError("sortByRowAndColumn only works for scalar matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    size_t N = this->row_indices.size();
    IVector permutation(N);
    thrust_wrapper::sequence<T_Config::memSpace>(permutation.begin(), permutation.end());
    cudaCheckError();
    // compute permutation and sort by (I,J)
    {
        IVector temp(this->col_indices);
        amgx::thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
        temp = this->row_indices;
        amgx::thrust::gather(permutation.begin(), permutation.end(), temp.begin(), this->row_indices.begin());
        amgx::thrust::stable_sort_by_key(this->row_indices.begin(), this->row_indices.end(), permutation.begin());
        temp = this->col_indices;
        amgx::thrust::gather(permutation.begin(), permutation.end(), temp.begin(), this->col_indices.begin());
    }
    cudaCheckError();
    // use permutation to reorder the values
    {
        MVector temp(this->values);
        amgx::thrust::gather(permutation.begin(), permutation.end(), temp.begin(), this->values.begin());
    }
    cudaCheckError();
    this->set_allow_recompute_diag(true);
    this->addProps(CSR);
    // remove row indices array
    this->delProps(COO);
    this->computeDiagonal();
    this->set_initialized(1);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDiagonal()
{
    ViewType oldView  = this->currentView();

    if (this->m_initialized) { this->setView(ALL); }

    if (this->allow_recompute_diag)
    {
        index_type num_rows = this->get_num_rows();
        index_type num_nz = this->get_num_nz();
        IVector &row_offsets = this->row_offsets;
        IVector &row_indices = this->row_indices;
        IVector &col_indices = this->col_indices;
        IVector &diag = this->diag;

        if (this->diag.size() != this->get_num_rows()) { this->diag.resize(this->get_num_rows()); }

        if (this->m_diag_end_offsets.size() != this->get_num_rows()) { this->m_diag_end_offsets.resize(this->get_num_rows()); }

        if ( this->hasProps(DIAG) )
        {
            int first = num_nz;

            for (int r = 0; r < num_rows; r++)
            {
                diag[r] = first++;
            }
        }
        else
        {
            index_type null_index = this->get_num_nz();

            if ( this->hasProps(CSR) )
            {
                for (int r = 0; r < num_rows; r++)
                {
                    int start = row_offsets[r];
                    int end = row_offsets[r + 1];
                    diag[r] = null_index;

                    for (int j = start; j < end; j++)
                    {
                        if (col_indices[j] == r)
                        {
                            diag[r] = j;
                            break;
                        }
                    }
                }
            }
            else if (this->hasProps(COO) )
            {
                for (int i = 0; i < num_rows; i++)
                {
                    diag[i] = null_index;
                }

                for (int j = 0; j < num_nz; j++)
                {
                    int r = row_indices[j];

                    if (r == col_indices[j])
                    {
                        diag[r] = j;
                    }
                }
            }
        }

        for (int r = 0; r < num_rows; r++)
        {
            this->m_diag_end_offsets[r] = diag[r] + 1;
        }
    }

    this->setView(oldView);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDiagonal()
{
    ViewType oldView;
    index_type null_index = 0; //this->get_num_nz();

    if (!this->allow_recompute_diag || !(this->get_num_rows() > 0))
    {
        return;
    }

    if (this->m_initialized)
    {
        oldView = this->currentView();
        this->setView(ALL);
    }

    if (this->diag.size() < this->get_num_rows())
    {
        this->diag.resize(this->get_num_rows());
    }

    if (this->m_diag_end_offsets.size() < this->get_num_rows())
    {
        this->m_diag_end_offsets.resize(this->get_num_rows());
    }

    if (this->hasProps(DIAG))
    {
        int num_blocks = std::min(4096, (this->get_num_rows() + 511) / 512);
        computeDiagonalKernelDiagProp <<< num_blocks, 512, 0, amgx::thrust::global_thread_handle::get_stream()>>>(this->get_num_rows(), this->get_num_nz(), this->diag.raw(), this->m_diag_end_offsets.raw());
    }
    else if (this->hasProps(COO))
    {
        int num_blocks = std::min(4096, (this->get_num_nz() + 511) / 512);
        computeDiagonalKernelCOO <<< num_blocks, 512>>>(this->get_num_nz(), this->row_indices.raw(), this->col_indices.raw(), this->diag.raw(), this->m_diag_end_offsets.raw());
    }
    else
    {
        int num_blocks = std::min(4096, (this->get_num_rows() + 511) / 512);
        computeDiagonalKernelCSR <<< num_blocks, 512>>>(this->get_num_rows(), this->row_offsets.raw(), this->col_indices.raw(), this->diag.raw(), null_index, this->m_diag_end_offsets.raw());
    }

    cudaCheckError();

    if (this->m_initialized)
    {
        this->setView(oldView);
    }

#ifdef DEBUG

    if (this->diag_copy.size() == 0)
    {
        this->diag_copy = this->diag;
    }
    else
    {
        if ((this->diag.size() != this->diag_copy.size()) || (this->diag.size() == 1)) { return; }

        IVector_h h_diag = this->diag;
        IVector_h h_diag_copy = this->diag_copy;
        bool equal = true;

        for (unsigned int i = 0; i < this->diag.size(); ++i)
        {
            if (h_diag[i] != h_diag_copy[i])
            {
                equal = false;
                break;
            }
        }

        if (equal)
        {
            FatalError("ComputeDiagonal was called, but diagonal hasn't changed", AMGX_ERR_UNKNOWN);
        }
    }

#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void
Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::permuteValues()
{
    if (this->m_cols_reordered_by_color && this->m_is_permutation_inplace)
    {
        reorderValuesInPlace();
    }
    else if (this->m_cols_reordered_by_color && !this->m_is_permutation_inplace )
    {
        // might use a lot of memory
        MVector temp_values;
        temp_values.resize(this->values.size());
        temp_values.set_block_dimx(this->values.get_block_dimx());
        temp_values.set_block_dimy(this->values.get_block_dimy());
        amgx::unpermuteVector(this->values, temp_values, this->m_values_permutation_vector, (this->get_num_nz()) * (this->get_block_size()));
        this->values.swap(temp_values);
        temp_values.clear();
        temp_values.shrink_to_fit();
    }
    else
    {
        FatalError("Invalid reordering level in permuteValues", AMGX_ERR_CONFIGURATION);
    }
}


void computeRowOffsetsDevice(int num_blocks, INDEX_TYPE num_rows, INDEX_TYPE num_nz, const INDEX_TYPE *row_indices, INDEX_TYPE *row_offsets, INDEX_TYPE block_size )
{
    computeRowOffsetsKernel <<< num_blocks, 512>>>(num_rows, num_nz, row_indices, row_offsets);
    cudaCheckError();
}

extern void computeRowIndicesDevice(int num_blocks, INDEX_TYPE num_rows, const INDEX_TYPE *row_offsets, INDEX_TYPE *row_indices, INDEX_TYPE block_size )
{
    computeRowIndicesKernel <<< num_blocks, 512>>>(num_rows, row_offsets, row_indices);
    cudaCheckError();
}

void computeColorOffsetsDeviceCSR(int num_blocks, INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, const INDEX_TYPE *row_colors, INDEX_TYPE *smaller_color_offsets, INDEX_TYPE *larger_color_offsets, INDEX_TYPE block_size, INDEX_TYPE *diag )
{
    computeColorOffsetsKernelCSR <<< num_blocks, 512>>>(num_rows, row_offsets, col_indices, row_colors, smaller_color_offsets, larger_color_offsets, diag);
    cudaCheckError();
}

template <typename T>
void reorderElementsDeviceCSR(INDEX_TYPE num_rows,
                              INDEX_TYPE *row_offsets,
                              INDEX_TYPE *permutation,
                              INDEX_TYPE *col_indices,
                              T *values,
                              INDEX_TYPE block_size)
{
    amgx::thrust::device_ptr<INDEX_TYPE> dev_ptr = amgx::thrust::device_pointer_cast(row_offsets);
    INDEX_TYPE max_row_length = std::max(1, thrust_wrapper::transform_reduce<AMGX_device>(dev_ptr, dev_ptr + num_rows, row_length<INDEX_TYPE>(), 0, amgx::thrust::maximum<INDEX_TYPE>()));
    //TODO: optimise this in terms of storage
    INDEX_TYPE storage_space = 100 * 1024 * 1024 * sizeof(T) / sizeof(cuDoubleComplex); // because we allocate as for cuComplex
    INDEX_TYPE blocks = 1500 < storage_space / (max_row_length * block_size * sizeof(T)) ? 1500 : storage_space / (max_row_length * block_size * sizeof(T));
    blocks = blocks < num_rows ? blocks : num_rows;
    INDEX_TYPE aligned_space = ((max_row_length * block_size * sizeof(T) / 128 + 1) * 128) / sizeof(T); //pad to 128 bytes
    Vector<amgx::TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt> > tempstorage(blocks * aligned_space);
    reorderElements <<< blocks, 256>>>(num_rows, row_offsets, permutation, values, (T *)tempstorage.raw(), aligned_space, block_size);
    cudaCheckError();
}

// explicitly instantiate reorderElementsDeviceCSR, since we call it from header and it's not a part of some class
template void reorderElementsDeviceCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *permutation, INDEX_TYPE *col_indices, float *values, INDEX_TYPE block_size);
template void reorderElementsDeviceCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *permutation, INDEX_TYPE *col_indices, double *values, INDEX_TYPE block_size);
template void reorderElementsDeviceCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *permutation, INDEX_TYPE *col_indices, cuComplex *values, INDEX_TYPE block_size);
template void reorderElementsDeviceCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *permutation, INDEX_TYPE *col_indices, cuDoubleComplex *values, INDEX_TYPE block_size);


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MatrixBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Matrix<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE





}//end namespace amgx
