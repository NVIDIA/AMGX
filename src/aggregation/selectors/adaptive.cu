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

#include <aggregation/selectors/adaptive.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <basic_types.h>
#include <matrix_analysis.h>
#include <transpose.h>

#include <async_event.h>

#include <thrust/count.h> //count
#include <thrust/sort.h> //sort
#include <thrust/binary_search.h> //lower_bound
#include <thrust/unique.h> //unique
#include <thrust/host_vector.h>
#include <cusp/detail/format_utils.h> //offsets_to_indices
#include <determinism_checker.h>
#include <curand.h>

namespace amgx
{
namespace aggregation
{
namespace adaptive
{

template <typename IndexType, typename ValueType>
__global__
void checkDiagonalDominance( const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values, const IndexType *diag, IndexType *aggregates, int num_owned, int bsize, double diag_dom )
{
    //this time tid = i
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int bsize_sq = bsize * bsize;

    while ( i < num_owned )
    {
        bool dd = false;

        for ( int row = 0; row < bsize; row++ )
        {
            ValueType sum = 0.0;
            int iimax = row_offsets[i + 1];

            for ( int ii = row_offsets[i]; ii < iimax; ii++ )
            {
                //dont add the diagonal
                if ( col_indices[ii] == i )
                {
                    for ( int col = 0; col < bsize; col++ )
                    {
                        sum += fabs( values[ii * bsize_sq + bsize * row + col] );

                        if ( col == row )
                        {
                            continue;
                        }
                    }
                }
                else
                {
                    for ( int col = 0; col < bsize; col++ )
                    {
                        sum += fabs( values[ii * bsize_sq + bsize * row + col] );
                    }
                }
            }

            //check for DD condition and sort out
            dd = sum <= diag_dom * fabs( values[diag[i] * bsize_sq + row * bsize + row] );

            if ( !dd )
            {
                break;
            }
        }

        if ( dd )
        {
            aggregates[i] = num_owned;
        }

        i += gridDim.x * blockDim.x;
    }
}


template <typename IndexType, typename ValueType>
__global__
void assignNodeToBin( const ValueType *x, IndexType *aggregates, ValueType min, ValueType max, IndexType numRows, IndexType numAggregates )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    double range = (max - min) / numAggregates;

    while ( tid < numRows )
    {
        aggregates[tid] = floor( (x[tid] - min) / range );
        tid += gridDim.x * blockDim.x;
    }
}


template <typename ValueType, typename IndexType>
__global__
void rescaleVector( ValueType *x, IndexType numRows )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < numRows )
    {
        x[tid] = 2 * x[tid] - 1;
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------
//    Methods
// ----------------

// Constructor
template<class T_Config>
AdaptiveSelectorBase<T_Config>::AdaptiveSelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
    smoother = SolverFactory<T_Config>::allocate( cfg, cfg_scope, "smoother" );
}
/*
template <typename ValueType>
void curandGenerateUniformWrapper( curandGenerator_t generator, ValueType *outputPtr, size_t num );
template<> void curandGenerateUniformWrapper<float>( curandGenerator_t generator, float *outputPtr, size_t num )
{
    curandGenerateUniform( generator, outputPtr, num);
}
template<> void curandGenerateUniformWrapper<double>( curandGenerator_t generator, double *outputPtr, size_t num )
{
    curandGenerateUniformDouble( generator, outputPtr, num);
}
*/

template<class T_Config>
void AdaptiveSelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &numAggregates)
{
    FatalError("Adaptive selector is still in development", AMGX_ERR_NOT_IMPLEMENTED);
    /*
        if (A.get_block_dimx() == A.get_block_dimy())
        {
            IndexType numRows = A.get_num_rows();


            // This is how adaptive aggregation works:
            // 1. init random x vector (random values would preferably be between -1 and 1) and 0 rhs
            // 2. apply smoother a certain number of steps
            // 3. solve "binning problem":
            //      1. find min and max of x
            //      2. create N*coarseningrate bins with linear range
            //      3. assign nodes to bin, this is the final aggregates vector

            // allocate
            VVector& x = *Allocator<VVector>::allocate( numRows );
            VVector& rhs = *Allocator<VVector>::allocate( numRows );

            // initialize rhs
            thrust::fill( rhs.begin(), rhs.end(), 0.0 );

            //compute random numbers
            curandGenerator_t generator;
            curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
            curandGenerateUniformWrapper(generator, x.raw(), numRows);

            cudaStream_t str = 0;
            const int threads_per_block = 256;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (numRows-1)/threads_per_block + 1 );
            rescaleVector<<<num_blocks, threads_per_block, 0, str>>>( x.raw(), numRows );


            // smooth
            smoother->setup( A, false );
            smoother->set_max_iters( 15 ); //TODO: make this a parameter
            smoother->solve( x, rhs, false );

            //free rhs
            Allocator<VVector>::free( &rhs, numRows );

            //find min, max
            ValueType min = thrust::reduce( x.begin(), x.end(), 0.0, thrust::minimum<ValueType>() );
            ValueType max = thrust::reduce( x.begin(), x.end(), 0.0, thrust::maximum<ValueType>() );

            //bin
            aggregates.resize( numRows );
            numAggregates = numRows / 4;
            assignNodeToBin<<<num_blocks, threads_per_block, 0, str>>>( x.raw(), aggregates.raw(), min, max, numRows, numAggregates );

            //sync
            cudaStreamSynchronize( str );
            cudaCheckError();

            //free
            Allocator<VVector>::free( &x, numRows );

            //fill "gaps" of empty bins
            this->renumberAndCountAggregates(aggregates, aggregates_global, numRows, numAggregates);
        }
        else
            FatalError("Unsupported block size for Adaptive Aggregation", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        */
}

// -------------------------
//    Explict instantiations
// -------------------------
#define AMGX_CASE_LINE(CASE) template class AdaptiveSelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class AdaptiveSelector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
}
