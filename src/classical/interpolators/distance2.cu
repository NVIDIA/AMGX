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

#include <classical/interpolators/distance2.h>
#include <classical/interpolators/common.h>
#include <amgx_timer.h>
#include <basic_types.h>
#include <types.h>
#include <cutil.h>
#include <util.h>
#include <sm_utils.inl>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <sort.h>
#include <assert.h>
#include <misc.h>
#include <sstream>
#include <csr_multiply.h>
#include <hash_workspace.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <device_properties.h>
#include <thrust_wrapper.h>

namespace amgx
{

using std::set;
using std::vector;

/*
 * remove all duplicates from a vector
 */
template <typename T>
void removeDuplicates(vector<T> &vec)
{
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

/*
 * find the value at a particular column of a row if it exists
 */
template <typename IVector, typename VVector>
typename VVector::value_type findValue(const IVector &columns,
                                       const VVector &values,
                                       const int row_begin, const int row_end, const int column_needed)
{
    typename VVector::value_type v = 0;

    // NAIVE
    for (int k = row_begin; k < row_end; k++)
    {
        int kcol = columns[k];

        if (kcol == column_needed)
        {
            v = values[k];
        }
    }

    return v;
}


template <typename IndexType>
void __global__  createCfMapGlobal(const IndexType *cf_map, int64_t *cf_map_global, const int64_t my_part_offset, const IndexType num_owned_fine_pts)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_owned_fine_pts ; tidx += blockDim.x * gridDim.x)
    {
        // Only renumber the interior points
        if (cf_map[tidx] >= 0 && tidx < num_owned_fine_pts)
        {
            // Simply shift
            cf_map_global[tidx] = (int64_t) cf_map[tidx] /* my_local_id */ + my_part_offset;
        }
        else
        {
            cf_map_global[tidx] = -1;
        }
    }
}

template <int coop, typename index_type>
__global__ void fill_P_global_col_indices_kernel(index_type *row_offsets, index_type *C_hat_start, index_type *C_hat_end, index_type *C_hat, int64_t *cf_map_global, int64_t *global_col_indices, index_type num_rows)
{
    int coopIdx = threadIdx.x % coop;

    for (int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop; row < num_rows ; row += blockDim.x * gridDim.x / coop)
    {
        int64_t coarse_fine_id = cf_map_global[row];
        int j = row_offsets[row]  + coopIdx;

        // coarse row
        if (coarse_fine_id >= 0)
        {
            if (coopIdx == 0)
            {
                global_col_indices[j] = coarse_fine_id;
            }

            continue;
        }

        // strong fine row
        if (coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // fine row
        for (int i = C_hat_start[row] + coopIdx; i < C_hat_end[row]; i += coop, j += coop)
        {
            index_type fine_col = C_hat[i];
            int64_t global_col = cf_map_global[fine_col];
            global_col_indices[j] = global_col;
        }
    }
}


/*
 * calculate \tilde{a}_{ii} from eqn. 4.11 in [4]
 */
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance2_Interpolator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::calculateD( const Matrix_h &A, IVector &cf_map, BVector &s_con,
        vector<int> *C_hat, VVector &diag, VVector &D,
        set<int> *weak_lists, VVector &innerSum,
        IVector &innerSumOffsets)
{
    // only do the non-weak part for now
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        int outerOffset = innerSumOffsets[i], innerOffset = 0;
        ValueType sum = 0;

        // loop over all non-zeros
        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
        {
            int kcol = A.col_indices[k];

            if (cf_map[kcol] == FINE && s_con[k]) // strongly connected fine, ignore diagonal
            {
                int sgn = diag[kcol] < 0 ? -1 : 1;
                // find a_ki
                ValueType temp = findValue(A.col_indices, A.values, A.row_offsets[kcol], A.row_offsets[kcol + 1], i);
                ValueType a_ki = (sgn * temp < 0) * temp;
                sum += a_ki * innerSum[outerOffset + innerOffset];
                innerOffset++;
            }
        }

        // now calculate the weak part
        set<int> difference;

        for (set<int>::iterator it = weak_lists[i].begin(); it != weak_lists[i].end(); ++it)
        {
            difference.insert(*it);
        }

        for (int l = 0; l < C_hat[i].size(); l++)
        {
            difference.erase(C_hat[i][l]);
        }

        ValueType weak = 0;

        for (set<int>::iterator it = difference.begin(); it != difference.end(); ++it)
        {
            weak += findValue(A.col_indices, A.values, A.row_offsets[i], A.row_offsets[i + 1], *it);
        }

        D[i] = diag[i] + weak + sum;
    }
}

/*
 * Generate the lower sum from eqns. 4.10 and 4.11 of [4]
 * strongly resembles a sparse dot product
 */
template <typename IVector, typename VVector>
typename VVector::value_type sparseSum(const IVector &columns,
                                       const VVector &values,
                                       const int row_begin, const int row_end, const std::vector<int> &C_hat_row, const int sgn)
{
    int i1 = row_begin, i2 = 0;

    // check lengths are non-zero
    if (row_begin >= row_end || C_hat_row.size() == 0) { return 0; }

    typename VVector::value_type sum = 0;

    // if column exists in both C_hat_row and row of matrix, add matrix value
    while (true)
    {
        int col1 = columns[i1];
        int col2 = C_hat_row[i2];

        if (col1 == col2) // match
        {
            if (sgn * values[i1] < 0)
            {
                sum += values[i1];
            }

            i1++;
            i2++;

            if (i1 >= row_end || i2 >= C_hat_row.size()) { break; }
        }
        else if (col1 > col2)
        {
            // instead of linear increase, binary search
            /*      int min = i2++, max = C_hat_row.size();
                  while(min <= max)
                  {
                    int mid = (min+max)/2;
                    col2 = C_hat_row[mid];
                    if (col1 > col2)
                      min = mid+1;
                    else if (col1 < col2)
                      max = mid-1;
                    else
                    {
                      i2 = mid;
                      break;
                    }
                  }*/
            i2++;

            if (i2 >= C_hat_row.size()) { break; }
        }
        else
        {
            i1++;

            if (i1 >= row_end) { break; }
        }
    }

    return sum;
}

/*
 * Generate the repeatedly reused inner sum from eqns 4.10, 4.11 in [4]
 * \sum_{k\in F^s_i} a_{ik}\frac{1}{\sum_{l\in\hat{C}_i\cup\{i\}} \bar{a}_{kl}}
 */
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance2_Interpolator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generateInnerSum(Matrix_h &A, const IntVector &cf_map,
        const BVector &s_con,
        vector<int> *C_hat, VVector &diag,
        VVector &innerSum,
        IntVector &innerSumOffsets)
{
    // calculate necessary storage for innerSum
    int connections = 0;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            int jcol = A.col_indices[j];

            if (cf_map[jcol] == FINE && s_con[j] && jcol != i)
            {
                connections++;
            }
        }
    }

    innerSum.resize(connections);
    innerSumOffsets.resize(A.get_num_rows() + 1);
    innerSumOffsets[0] = 0;
    int innerOffset = 0;

    // generate the values
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
        {
            int kcol = A.col_indices[k];

            if (cf_map[kcol] == FINE && s_con[k] && kcol != i)
            {
                ValueType bottom_sum = 0;
                ValueType a_ik = A.values[k];
                int sgn = diag[kcol] < 0 ? -1 : 1;
                // generate the bottom sum - looks similar to sparse dot product
                bottom_sum = sparseSum(A.col_indices, A.values, A.row_offsets[kcol], A.row_offsets[kcol + 1],
                                       C_hat[i], sgn);
                // add \union i
                ValueType iTerm = findValue(A.col_indices, A.values, A.row_offsets[kcol], A.row_offsets[kcol + 1], i);
                bottom_sum += (sgn * iTerm < 0) * iTerm;

                if (bottom_sum == 0 && cf_map[i] == FINE)
                {
                    char buf[255];
                    sprintf(buf, "DIVISION BY ZERO: row: %d, col: %d\n", i, kcol);
                    amgx_output(buf, strlen(buf));
                }

                innerSum[innerOffset] = a_ik / bottom_sum;
                innerOffset++;
            }
        }

        innerSumOffsets[i + 1] = innerOffset;
    }
}

/*************************************************************************
* create the interpolation matrix P
************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance2_Interpolator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(Matrix_h &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix_h &P,
        void *amg)
{
    if (!A.is_matrix_singleGPU())
    {
        FatalError("Distributed classical AMG not implemented on host\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix_h::value_type ValueType;
    // implementation begins here
    set<int> *coarse_lists = new set<int>[A.get_num_rows()];
    set<int> *weak_lists = new set<int>[A.get_num_rows()];
    vector<int> *C_hat = new vector<int>[A.get_num_rows()]; // hold full \hat{C} set for each row
    set<int> *C_hat_set = new set<int>[A.get_num_rows()];
    set<int>::iterator set_it;
    VVector diag(A.get_num_rows(), 0);

    // grab the diagonal
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            if (A.col_indices[j] == i)
            {
                diag[i] = A.values[j];
            }
        }
    }

    // generate coarse sets
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        // loop over non-zero elements
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            int jcol = A.col_indices[j];

            if (jcol == i) { continue; } // skip the diagonal

            // Coarse and strongly connected
            if (cf_map[jcol] >= 0 && s_con[j])
            {
                coarse_lists[i].insert(jcol);
            }
            else if (!s_con[j] && jcol != i && cf_map[jcol] != STRONG_FINE)
            {
                weak_lists[i].insert(jcol);
            }
        }
    }

    int numWTotal = 0;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        numWTotal += (int) weak_lists[i].size();
    }

    typedef typename set<int>::iterator iSetIter;

    // generate \hat{C} for each row, i
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        C_hat[i].clear();

        if (cf_map[i] == FINE)
        {
            // copy coarse_lists[i] to C_hat_set[i]
            for (iSetIter it = coarse_lists[i].begin(); it != coarse_lists[i].end(); ++it)
            {
                C_hat_set[i].insert(*it);
            }

            // loop over non-zeros
            for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
            {
                int jcol = A.col_indices[j];

                if (jcol == i) { continue; } // don't want to take into account myself

                // if this is a strong-fine connection, union of the sets
                if (cf_map[jcol] == FINE && s_con[j])
                {
                    for (iSetIter it = coarse_lists[jcol].begin(); it != coarse_lists[jcol].end(); ++it)
                    {
                        C_hat_set[i].insert(*it);
                    }
                }
            }

            removeDuplicates(C_hat[i]);
        }

        // copy C_hat_set[i] into C_hat[i]
        for (iSetIter it = C_hat_set[i].begin(); it != C_hat_set[i].end(); ++it)
        {
            C_hat[i].push_back(*it);
        }
    }

    // check that \hat{C_i} for fine points does not contain i
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        if (cf_map[i] == FINE)
        {
            vector<int>::iterator it = std::find(C_hat[i].begin(), C_hat[i].end(), i);
            {
                assert(it == C_hat[i].end());
            }
        }
    }

    // generate the inner sum for all i, k
    VVector innerSum;
    IntVector innerSumOffsets;
    generateInnerSum(A, cf_map, s_con, C_hat, diag, innerSum, innerSumOffsets);
    // calculate the ~a_ii term
    VVector D(A.get_num_rows(), 0);
    calculateD(A, cf_map, s_con, C_hat, diag, D, weak_lists, innerSum, innerSumOffsets);
    // delete all storage we have no more need for
    delete [] coarse_lists;
    delete [] weak_lists;
    delete [] C_hat_set;
    // now we have \hat{C} and the diagonals, we can start calculating the weights
    // remap cf_map (to be like in distance1)
    int coarsePoints = 0;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        // if coarse
        if (cf_map[i] >= 0) { coarsePoints++; }
    }

    // count the non-zeros - 1 per row if coarse, len(C_hat[i]) otherwise
    IntVector nonZeroOffsets(A.get_num_rows() + 1);
    IntVector nonZerosPerRow(A.get_num_rows());
    int nonZeros = 0;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        if (cf_map[i] >= 0)
        {
            nonZerosPerRow[i] = 1;
            nonZeros++;
        }
        else if (cf_map[i] == STRONG_FINE)
        {
            nonZerosPerRow[i] = 0;
        }
        else
        {
            nonZerosPerRow[i] = (int) C_hat[i].size();
            nonZeros += (int) C_hat[i].size();
        }
    }

    // get the offsets with an exclusive scan
    thrust_wrapper::exclusive_scan(nonZerosPerRow.begin(), nonZerosPerRow.end(), nonZeroOffsets.begin());
    cudaCheckError();
    nonZeroOffsets[A.get_num_rows()] = nonZeros;
    // resize P
    P.addProps(CSR);
    P.resize(A.get_num_rows(), coarsePoints, nonZeros, 1);
    // copy nonzero offsets to the P matrix
    amgx::thrust::copy(nonZeroOffsets.begin(), nonZeroOffsets.end(), P.row_offsets.begin());
    cudaCheckError();
    // main loop
    int outerOffset = 0, innerOffset;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        int nonZeroOffset = nonZeroOffsets[i];
        int localNonZeros = 0;
        outerOffset = innerSumOffsets[i];

        if (cf_map[i] >= 0)
        {
            // will be setting here & continuing
            P.values[nonZeroOffset + localNonZeros] = 1.0;
            P.col_indices[nonZeroOffset + localNonZeros] = cf_map[i];
            continue;
        }
        else if (cf_map[i] != STRONG_FINE) // fine row
        {
            //
            for (int j = 0; j < C_hat[i].size(); j++)
            {
                innerOffset = 0; // set to 'row' offset
                int jcol = C_hat[i][j]; // j can potentially not exist on row i

                if (jcol == i) // skip diagonal
                {
                    FatalError("Error - Diagonal should never be in C_hat[i]", AMGX_ERR_BAD_PARAMETERS);
                }

                ValueType value = 0;
                // loop over i and find a_ij - if not found, doesn't exist, keep as 0
                ValueType a_ij = 0;
                a_ij = findValue(A.col_indices, A.values, A.row_offsets[i], A.row_offsets[i + 1], jcol);

                // now loop over and find strong-fine connections
                for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
                {
                    int kcol = A.col_indices[k];

                    if (cf_map[kcol] == FINE && s_con[k]) // Strong-Fine edge, ignore diagonal
                    {
                        if (kcol == i)
                        {
                            FatalError("Error - point should never to strongly connected to itself", AMGX_ERR_BAD_PARAMETERS);
                        }

                        // point we care about
                        int sgn = diag[kcol] < 0 ? -1 : 1;
                        ValueType a_kj = 0;
                        // get \bar{a_kj}
                        ValueType temp = findValue(A.col_indices, A.values, A.row_offsets[kcol], A.row_offsets[kcol + 1], jcol);
                        a_kj = (sgn * temp < 0) * temp;
                        // see if we need to add \bar{a_ki} in
                        vector<int>::iterator it = std::find(C_hat[i].begin(), C_hat[i].end(), i);

                        if (it != C_hat[i].end())
                        {
                            FatalError("Error - point i should never be in\\hat{C_i}", AMGX_ERR_BAD_PARAMETERS);
                        }
                        else
                        {
                            // we now have everything we need - add to sum
                            // not valid in non-symmetric matrices
                            value += innerSum[outerOffset + innerOffset] * a_kj;
                            innerOffset++;
                        }
                    } // strong fine connection end
                } // k loop (find strong fine connections) end

                ValueType diagonal = (D[i]);
                // now set column & value for P
                P.col_indices[nonZeroOffset + localNonZeros] = cf_map[jcol];
                P.values[nonZeroOffset + localNonZeros] = -1.0 / (diagonal) * (a_ij + value);
                localNonZeros++;
            } // end j loop
        } // fine connection loop

        outerOffset += innerOffset;
    } // end i loop

    delete [] C_hat;
} // end distance2 interpolator


///////////////////////////////////////////////////////////////////////////////

// DEVICE CODE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace distance2_sm35
{

#include <sm_utils.inl>
#include <hash_containers_sm35.inl> // Included inside the namespace to solve name colisions.

__device__ __forceinline__ int get_work( int *queue, int warp_id )
{
    int offset = -1;

    if ( utils::lane_id() == 0 )
    {
        offset = atomicAdd( queue, 1 );
    }

    return utils::shfl( offset, 0 );
}

} // namespace distance2_sm35

namespace distance2_sm70
{

#include <sm_utils.inl>
#include <hash_containers_sm70.inl> // Included inside the namespace to solve name colisions.

__device__ __forceinline__ int get_work( int *queue, int warp_id )
{
    int offset = -1;

    if ( utils::lane_id() == 0 )
    {
        offset = atomicAdd( queue, 1 );
    }

    return utils::shfl( offset, 0 );
}

} // namespace distance2_sm70

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace distance2
{

#include <sm_utils.inl>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS )
    {
        // Skip coarse and strong fine rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // Load A row IDs.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // The number of elements.
        int count(0);

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            count += __popc( utils::ballot( is_coarse_strongly_connected ) );
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            __syncthreads();

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0, num_rows = __popc(vote) ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [b_col_it, b_col_end).
                int b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                // Load the range of B.
                int b_col_it  = A_rows[b_row_id + 0];
                int b_col_end = A_rows[b_row_id + 1];

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != b_row_id;
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_off_diagonal && s_con[b_col_it] && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    count += __popc( utils::ballot( is_coarse_strongly_connected ) );
                }
            }
        }

        // Store the number of columns in each row.
        if ( lane_id == 0 )
        {
            C_hat_offsets[a_row_id] = count;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    s_b_row_ids[threadIdx.x] = 0;
    __syncthreads();

    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS )
    {
        // Skip coarse rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // Load A row IDs.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // The number of elements.
        int count(0);

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            count += __popc( utils::ballot( is_coarse_strongly_connected ) );
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            utils::syncwarp();

            // For each warp, we have up to 32 rows of B to proceed.

            int num_rows = __popc(vote);
            for ( int k = 0; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [b_col_it, b_col_end).
                int b_row_id = -1;

                if ( is_active_k )
                {
                    b_row_id = s_b_row_ids[warp_id * WARP_SIZE + local_k];
                }

                // Load the range of B.
                int b_col_it = 0, b_col_end = 0;

                if ( is_active_k )
                {
                    b_col_it  = A_rows[b_row_id + 0];
                    b_col_end = A_rows[b_row_id + 1];
                }

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != b_row_id;
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_off_diagonal && s_con[b_col_it] && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    count += __popc( utils::ballot( is_coarse_strongly_connected ) );
                }
            }
        }

        // Store the number of columns in each row.
        if ( lane_id == 0 )
        {
            C_hat_offsets[a_row_id] = count;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const int *__restrict cf_map,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_end,
                      int *__restrict C_hat,
                      int *__restrict C_hat_pos,
                      int gmem_size,
                      int *g_keys,
                      int *wk_work_queue,
                      int *wk_status )
{

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Shared memory to vote.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 700
    distance2_sm70::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#else
    distance2_sm35::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 700
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm70::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm35::get_work( wk_work_queue, warp_id ) )
#endif
    {
        // Skip coarse rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        __syncthreads();
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );
        __syncthreads();

        // _iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( is_active )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = is_active && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            int item = -1;

            if ( is_coarse_strongly_connected )
            {
                item = a_col_id;
            }

            set.insert( item, wk_status );
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            utils::syncwarp();

            int num_rows = __popc( vote );

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                int uniform_b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                // Load the range of the row of B.
                int b_col_it  = A_rows[uniform_b_row_id + 0];
                int b_col_end = A_rows[uniform_b_row_id + 1];

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != uniform_b_row_id;
                    // Is it a strongly connected node.
                    is_strongly_connected = is_off_diagonal && s_con[b_col_it];
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_strongly_connected && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    int b_item = -1;

                    if ( is_coarse_strongly_connected )
                    {
                        b_item = b_col_id;
                    }

                    set.insert( b_item, wk_status );
                }
            }
        }

        int c_col_it = C_hat_start[a_row_id];
        int count = set.store_with_positions( &C_hat[c_col_it], &C_hat_pos[c_col_it] );

        if ( lane_id == 0 )
        {
            C_hat_end[a_row_id] = c_col_it + count;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const int *__restrict cf_map,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_end,
                      int *__restrict C_hat,
                      int *__restrict C_hat_pos,
                      int gmem_size,
                      int *g_keys,
                      int *wk_work_queue,
                      int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // Shared memory to vote.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];

    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 700
    distance2_sm70::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#else
    distance2_sm35::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 700
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm70::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm35::get_work( wk_work_queue, warp_id ) )
#endif
    {
        // Skip coarse rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // Clear the set.
        set.clear();

        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );

        // _iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( is_active )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = is_active && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            int item = -1;

            if ( is_coarse_strongly_connected )
            {
                item = a_col_id;
            }

            set.insert( item, wk_status );
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            utils::syncwarp();

            int num_rows = __popc( vote );

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                int uniform_b_row_id = -1;

                if ( is_active_k )
                {
                    uniform_b_row_id = s_b_row_ids[warp_id * WARP_SIZE + local_k];
                }

                // Load the range of the row of B.
                int b_col_it = 0, b_col_end = 0;

                if ( is_active_k )
                {
                    b_col_it  = A_rows[uniform_b_row_id + 0];
                    b_col_end = A_rows[uniform_b_row_id + 1];
                }

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != uniform_b_row_id;
                    // Is it a strongly connected node.
                    is_strongly_connected = is_off_diagonal && s_con[b_col_it];
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_strongly_connected && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    int b_item = -1;

                    if ( is_coarse_strongly_connected )
                    {
                        b_item = b_col_id;
                    }

                    set.insert( b_item, wk_status );
                }
            }
        }

        int c_col_it = C_hat_start[a_row_id];
        int count = set.store_with_positions( &C_hat[c_col_it], &C_hat_pos[c_col_it] );

        if ( lane_id == 0 )
        {
            C_hat_end[a_row_id] = c_col_it + count;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_inner_sum_kernel( const int A_num_rows,
                          const int *__restrict A_rows,
                          const int *__restrict A_cols,
                          const Value_type *__restrict A_vals,
                          const int *__restrict cf_map,
                          const bool *__restrict s_con,
                          const int *__restrict C_hat,
                          const int *__restrict C_hat_pos,
                          const int *__restrict C_hat_start,
                          const int *__restrict C_hat_end,
                          const Value_type *__restrict diag,
                          const int *__restrict inner_sum_offsets,
                          Value_type *inner_sum,
                          const int gmem_size,
                          int *g_keys,
                          int *wk_work_queue )
{

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // A shared location where threads propose a value.
    __shared__ volatile Value_type s_a_values[CTA_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 700
    distance2_sm70::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#else
    distance2_sm35::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 700
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm70::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm35::get_work( wk_work_queue, warp_id ) )
#endif
    {
        // Skip coarse rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // Clear the set.
        set.clear();
        // Rebuild C_hat.
        int c_hat_it  = C_hat_start[a_row_id];
        int c_hat_end = C_hat_end  [a_row_id];
        set.load( c_hat_end - c_hat_it, &C_hat[c_hat_it], &C_hat_pos[c_hat_it] );
        // The offset in the inner sum table.
        int inner_sum_offset = inner_sum_offsets[a_row_id];
        // And share the value of the diagonal.
        bool sign_diag = false;

        if ( lane_id == 0 )
        {
            sign_diag = sign( diag[a_row_id] );
        }

        sign_diag = utils::shfl( sign_diag, 0 );

        // Load A row IDs.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // _columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_off_diagonal && s_con[a_col_it] && cf_map[a_col_id] == FINE;
            // Read the associated value from A.
            Value_type a_value(0);

            if ( is_fine_strongly_connected )
            {
                a_value = A_vals[a_col_it];
            }

            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
                s_a_values [warp_id * WARP_SIZE + dest] = a_value;
            }

            utils::syncwarp();

            int num_rows = __popc( vote );
            // First n_rows threads reload the correct value.
            a_value = s_a_values[warp_id * WARP_SIZE + lane_id];

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Each thread keeps its own sum.
                Value_type bottom_sum(0);
                // Threads in the warp proceeds columns of B in the range [b_col_it, b_col_end).
                int b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                // TODO: make sure we have better memory accesses.
                int b_col_it  = A_rows[b_row_id  ];
                int b_col_end = A_rows[b_row_id + 1];

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any( b_col_it < b_col_end ) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it needed to compute the bottom sum.
                    bool is_needed = set.contains( b_col_id ) || b_col_id == a_row_id;
                    // Read the associated value from A.
                    Value_type b_value(0);

                    if ( is_needed )
                    {
                        b_value = A_vals[b_col_it];
                    }

                    // Update the sum if needed.
                    if ( sign_diag != sign( b_value ) )
                    {
                        bottom_sum += b_value;
                    }
                }

                // Reduce the row to a single value.
#pragma unroll
                for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
                {
                    int hi = __double2hiint(bottom_sum);
                    hi = utils::shfl_xor( hi, offset );
                    int lo = __double2loint(bottom_sum);
                    lo = utils::shfl_xor( lo, offset );
                    bottom_sum += __hiloint2double(hi, lo);
                }

                bottom_sum = utils::shfl( bottom_sum, 0 );

                if ( lane_id == k && bottom_sum != Value_type(0) )
                {
                    a_value /= bottom_sum;
                }
            }

            // Write the results.
            if ( lane_id < num_rows )
            {
                inner_sum[inner_sum_offset + lane_id] = a_value;
            }

            inner_sum_offset += num_rows;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_inner_sum_kernel( const int A_num_rows,
                          const int *__restrict A_rows,
                          const int *__restrict A_cols,
                          const Value_type *__restrict A_vals,
                          const int *__restrict cf_map,
                          const bool *__restrict s_con,
                          const int *__restrict C_hat,
                          const int *__restrict C_hat_pos,
                          const int *__restrict C_hat_start,
                          const int *__restrict C_hat_end,
                          const Value_type *__restrict diag,
                          const int *__restrict inner_sum_offsets,
                          Value_type *inner_sum,
                          const int gmem_size,
                          int *g_keys,
                          int *wk_work_queue )
{

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // A shared location where threads propose a value.
    __shared__ volatile Value_type s_a_values[CTA_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 700
    distance2_sm70::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#else
    distance2_sm35::Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 700
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm70::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm35::get_work( wk_work_queue, warp_id ) )
#endif
    {
        // Skip coarse rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // Clear the set.
        set.clear();
        // Rebuild C_hat.
        int c_hat_it  = C_hat_start[a_row_id];
        int c_hat_end = C_hat_end  [a_row_id];
        set.load( c_hat_end - c_hat_it, &C_hat[c_hat_it], &C_hat_pos[c_hat_it] );
        // The offset in the inner sum table.
        int inner_sum_offset = inner_sum_offsets[a_row_id];
        // And share the value of the diagonal.
        bool sign_diag = false;

        if ( lane_id == 0 )
        {
            sign_diag = sign( diag[a_row_id] );
        }

        sign_diag = utils::shfl( sign_diag, 0 );
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // _columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_off_diagonal && s_con[a_col_it] && cf_map[a_col_id] == FINE;
            // Read the associated value from A.
            Value_type a_value(0);

            if ( is_fine_strongly_connected )
            {
                a_value = A_vals[a_col_it];
            }

            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
                s_a_values [warp_id * WARP_SIZE + dest] = a_value;
            }

            int num_rows = __popc( vote );
            // First n_rows threads reload the correct value.
            a_value = s_a_values[warp_id * WARP_SIZE + lane_id];

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                // Each thread keeps its own sum.
                Value_type bottom_sum(0);
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [b_col_it, b_col_end).
                int b_row_id = -1;

                if ( is_active_k )
                {
                    b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k + lane_id_div_num_threads];
                }

                // TODO: make sure we have better memory accesses.
                int b_col_it = 0, b_col_end = 0;

                if ( is_active_k )
                {
                    b_col_it  = A_rows[b_row_id  ];
                    b_col_end = A_rows[b_row_id + 1];
                }

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any( b_col_it < b_col_end ) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it needed to compute the bottom sum.
                    bool is_needed = set.contains( b_col_id ) || b_col_id == a_row_id;
                    // Read the associated value from A.
                    Value_type b_value(0);

                    if ( is_needed )
                    {
                        b_value = A_vals[b_col_it];
                    }

                    // Update the sum if needed.
                    if ( sign_diag != sign( b_value ) )
                    {
                        bottom_sum += b_value;
                    }
                }

                // Reduce the row to a single value.
#pragma unroll
                for ( int offset = NUM_THREADS_PER_ROW / 2 ; offset > 0 ; offset >>= 1 )
                {
                    int hi = __double2hiint(bottom_sum);
                    hi = utils::shfl_down( hi, offset, NUM_THREADS_PER_ROW );
                    int lo = __double2loint(bottom_sum);
                    lo = utils::shfl_down( lo, offset, NUM_THREADS_PER_ROW );
                    bottom_sum += __hiloint2double(hi, lo);
                }

                bottom_sum = utils::shfl( bottom_sum, lane_id_mod_num_threads * NUM_THREADS_PER_ROW );

                if ( lane_id >= k && lane_id < k + NUM_THREADS_PER_ROW && bottom_sum != Value_type(0) )
                {
                    a_value /= bottom_sum;
                }
            }

            // Write the results.
            if ( lane_id < num_rows )
            {
                inner_sum[inner_sum_offset + lane_id] = a_value;
            }

            inner_sum_offset += num_rows;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_interp_weight_kernel( const int A_num_rows,
                              const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const Value_type *__restrict A_vals,
                              const int *__restrict cf_map,
                              const bool *__restrict s_con,
                              const int *__restrict C_hat,
                              const int *__restrict C_hat_pos,
                              const int *__restrict C_hat_start,
                              const int *__restrict C_hat_end,
                              const Value_type *__restrict diag,
                              const int *__restrict inner_sum_offsets,
                              const Value_type *__restrict inner_sum,
                              const int *__restrict P_rows,
                              int *P_cols,
                              Value_type *P_vals,
                              const int gmem_size,
                              int *g_keys,
                              Value_type *g_vals,
                              int *wk_work_queue )
{
    
    const int NUM_WARPS = CTA_SIZE / 32;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // A shared location where threads propose a value.
    __shared__ volatile Value_type s_aki[NUM_WARPS];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 700
    distance2_sm70::Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );
#else
    distance2_sm35::Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 700
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm70::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = distance2_sm35::get_work( wk_work_queue, warp_id ) )
#endif
    {
        int coarse_fine_id = cf_map[a_row_id];

        // Skip coarse rows.
        if ( coarse_fine_id >= 0 )
        {
            if ( lane_id == 0 )
            {
                int p_row_it = P_rows[a_row_id];
                P_cols[p_row_it] = coarse_fine_id;
                P_vals[p_row_it] = Value_type( 1 );
            }

            continue;
        }

        if ( coarse_fine_id == STRONG_FINE )
        {
            continue;
        }

        // Clear the table.
        map.clear();
        // Rebuild C_hat.
        int c_hat_it  = C_hat_start[a_row_id];
        int c_hat_end = C_hat_end  [a_row_id];
        map.load( c_hat_end - c_hat_it, &C_hat[c_hat_it], &C_hat_pos[c_hat_it] );
        // Load A row IDs.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // The offset in the inner sum table.
        int inner_sum_offset = inner_sum_offsets[a_row_id];
        // Weak value.
        Value_type weak(0), sum(0);

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id(-1);
            Value_type a_value(0);

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
                a_value  = A_vals[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it a strongly-connected column.
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it a weakly connected node.
            bool is_weakly_connected = is_off_diagonal && !is_strongly_connected;
            // Is isolated
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Update C_hat values. If the value is not in C_hat it will be skipped (true parameter).
            bool is_in_c_hat = map.update( a_col_id, a_value );

            // Update the weak value.
            if ( is_weakly_connected && !is_in_c_hat && !is_strong_fine)
            {
                weak += a_value;
            }

            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && cf_map[a_col_id] == FINE;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            utils::syncwarp();

            int num_rows = __popc( vote );
            // We pre-load inner sums.
            sum = Value_type(0);

            if ( lane_id < num_rows )
            {
                sum = inner_sum[inner_sum_offset + lane_id];
            }

            inner_sum_offset += num_rows;

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [b_col_it, bCol_end).
                int b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                // TODO: make sure we have better memory accesses. b_colBegin rather than bCol_it because we iterate twice.
                int b_col_begin = A_rows[b_row_id  ];
                int b_col_end   = A_rows[b_row_id + 1];
                // Diagonal element.
                Value_type b_diag = diag[b_row_id];

                // Set s_aki to 0 because a_row_id may not appear in a row.
                if ( lane_id == 0 )
                {
                    s_aki[warp_id] = Value_type(0);
                }

                utils::syncwarp();

                // Load the kth inner sum.
                Value_type uniform_val = utils::shfl( sum, k );

                // _iterate over the range of columns of B.
                for ( int b_col_it = b_col_begin + lane_id ; utils::any( b_col_it < b_col_end ) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    int b_col_id(-1);
                    Value_type b_value(0);

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                        b_value  = A_vals[b_col_it];
                    }

                    // Update bValue if needed.
                    if ( sign( b_diag ) == sign( b_value ) )
                    {
                        b_value = Value_type(0);
                    }

                    // See if we have found a_row_id in the row. Only one thread can evaluate to true.
                    if ( a_row_id == b_col_id )
                    {
                        s_aki[warp_id] = b_value;
                    }

                    // Update C_hat values. If the value is not in C_hat it will be skipped (true parameter).
                    map.update( b_col_id, b_value * uniform_val );
                }

                utils::syncwarp();

                // The thread k updates the sum.
                if ( lane_id == 0 )
                {
                    weak += s_aki[warp_id] * uniform_val;
                }
            }
        }

        // We're done with that row of A. We compute D.
#pragma unroll

        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            weak += utils::shfl_xor( weak, mask );
        }

        if ( lane_id == 0 )
        {
            weak += diag[a_row_id];
            sum   = Value_type(-1) / weak;
        }

        sum = utils::shfl( sum, 0 );

        int p_col_tmp = -1;

        if ( lane_id < 2 )
        {
            p_col_tmp = P_rows[a_row_id + lane_id];
        }

        int p_col_it  = utils::shfl( p_col_tmp, 0 );
        int p_col_end = utils::shfl( p_col_tmp, 1 );

        map.store_map_keys_scale_values( p_col_end - p_col_it, cf_map, &P_cols[p_col_it], sum, &P_vals[p_col_it] );
    }
}


} // namespace distance2

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename IndexType, int kCtaSize >
__global__
void
calculateInnerSumStorageKernel( const IndexType *A_rows,
                                const IndexType *A_cols,
                                const int A_num_rows,
                                const int *cf_map,
                                const bool *s_con,
                                int *innerSum_offsets )
{
    const int nWarps = kCtaSize / 32;
    // The coordinates of the thread inside the CTA/warp.
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Loop over rows of A.
    for ( int aRowId = blockIdx.x * nWarps + warpId ; aRowId < A_num_rows ; aRowId += gridDim.x * nWarps )
    {
        // Skip coarse rows.
        int coarse_fine_id = cf_map[aRowId];

        if ( coarse_fine_id >= 0 || coarse_fine_id == STRONG_FINE)
        {
            continue;
        }

        // The sum for each thread.
        int sum = 0;
        // Load A row IDs.
        int aColIt  = A_rows[aRowId  ];
        int aColEnd = A_rows[aRowId + 1];

        // Iterate over the columns of A.
        for ( aColIt += laneId ; utils::any( aColIt < aColEnd ) ; aColIt += 32 )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int aColId = aColIt < aColEnd ? A_cols[aColIt] : -1;
            // Is it an off-diagonal element.
            bool is_off_diagonal = aColIt < aColEnd && aColId != aRowId;
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_off_diagonal && s_con[aColIt] && cf_map[aColId] == FINE;
            // Update the sum.
            sum += __popc( utils::ballot( is_fine_strongly_connected ) );
        }

        // Update the value.
        if ( laneId == 0 )
        {
            innerSum_offsets[aRowId] = sum;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * grab the diagonal of the matrix
 */
template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernel(const IndexType *offsets, const IndexType *column_indices,
                       const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        for (int j = offset; j < offset + numj; j++)
        {
            int jcol = column_indices[j];

            if (tIdx == jcol)
            {
                diagonal[tIdx] = values[j];
            }
        }
    }
}

/*************************************************************************
 * Implementing Extended+i algorithm from \S 4.5 of:
 * "Distance-two interpolation for parallel algebraic multigrid"
 * Reference [4] on wiki
 ************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
Distance2_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::Distance2_Interpolator(AMG_Config &cfg, const std::string &cfg_scope) : Base(cfg, cfg_scope)
{}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
Distance2_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::~Distance2_Interpolator()
{}

enum { WARP_SIZE = 32, SMEM_SIZE = 128 };

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance2_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(Matrix_d &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix_d &P,
        void *amg_ptr )
{
    const int blockSize = 256;
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // get raw pointers to data
    const IndexType *Aoffsets = A.row_offsets.raw();
    const IndexType *Acolumn_indices = A.col_indices.raw();
    const ValueType *Avalues = A.values.raw();
    const IndexType Anum_rows = (int) A.get_num_rows();
    typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;

    cudaDeviceProp props = getDeviceProperties();
    int grid_size = (props.major >= 7) ? 1024 : 128;
    int gmem_size = (props.major >= 7) ? 512 : 2048;

    Hash_Workspace<TConfig_d, int> exp_wk(true, grid_size, 8, gmem_size);

    IntVector C_hat_start( A.get_num_rows() + 1, 0 ), C_hat_end( A.get_num_rows() + 1, 0 );
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int work_offset = grid_size * NUM_WARPS;
        cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
        int avg_nz_per_row = (A.get_num_rows() == 0) ? 0 : A.get_num_nz() / A.get_num_rows();

        if ( avg_nz_per_row < 16 )
        {
            distance2::estimate_c_hat_size_kernel< 8, CTA_SIZE, WARP_SIZE> <<< 2048, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat_start.raw());
        }
        else
        {
            distance2::estimate_c_hat_size_kernel<CTA_SIZE, WARP_SIZE> <<< 2048, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat_start.raw());
        }

        cudaCheckError();
    }
    // Compute row offsets.
    thrust_wrapper::exclusive_scan( C_hat_start.begin( ), C_hat_start.end( ), C_hat_start.begin( ) );
    cudaCheckError();
    // Allocate memory to store columns/values.
    int nVals = C_hat_start[C_hat_start.size() - 1];
    IntVector C_hat( nVals ), C_hat_pos( nVals );
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = (A.get_num_rows() == 0) ? 0 : A.get_num_nz() / A.get_num_rows();
        int attempt = 0;

        for ( bool done = false ; !done && attempt < 10 ; ++attempt )
        {
            // Double the amount of GMEM (if needed).
            if ( attempt > 0 )
            {
                exp_wk.expand();
            }

            // Reset the status. TODO: Launch async copies.
            int status = 0;
            cudaMemcpy( exp_wk.get_status(), &status, sizeof(int), cudaMemcpyHostToDevice );
            // Compute the set C_hat.
            int work_offset = grid_size * NUM_WARPS;
            cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );

            // Run the computation.
            if ( avg_nz_per_row < 16 )
            {
                distance2::compute_c_hat_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                    A.get_num_rows(),
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    cf_map.raw(),
                    s_con.raw(),
                    C_hat_start.raw(),
                    C_hat_end.raw(),
                    C_hat.raw(),
                    C_hat_pos.raw(),
                    exp_wk.get_gmem_size(),
                    exp_wk.get_keys(),
                    exp_wk.get_work_queue(),
                    exp_wk.get_status() );
            }
            else
            {
                distance2::compute_c_hat_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                    A.get_num_rows(),
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    cf_map.raw(),
                    s_con.raw(),
                    C_hat_start.raw(),
                    C_hat_end.raw(),
                    C_hat.raw(),
                    C_hat_pos.raw(),
                    exp_wk.get_gmem_size(),
                    exp_wk.get_keys(),
                    exp_wk.get_work_queue(),
                    exp_wk.get_status() );
            }

            cudaCheckError();
            // Read the result from count_non_zeroes.
            cudaMemcpy( &status, exp_wk.get_status(), sizeof(int), cudaMemcpyDeviceToHost );
            done = status == 0;
        }
    }
    // get pointers to data
    int *C_hat_ptr = C_hat.raw();
    int *C_hat_start_ptr = C_hat_start.raw();
    int *C_hat_end_ptr = C_hat_end.raw();
    // modify Coarse/Fine map array to have the column index of the new interpolation matrix
    // for evry coarse point
    int coarsePoints;
    DistributedArranger<TConfig_d> *prep;

    if (A.is_matrix_distributed())
    {
        prep = new DistributedArranger<TConfig_d>;
        int num_owned_fine_pts = A.get_num_rows();
        int num_owned_coarse_pts = thrust_wrapper::count_if(cf_map.begin(), cf_map.begin() + num_owned_fine_pts, is_non_neg());
        cudaCheckError();
        int num_halo_coarse_pts = thrust_wrapper::count_if(cf_map.begin() + num_owned_fine_pts, cf_map.end(), is_non_neg());
        cudaCheckError();
        coarsePoints = num_owned_coarse_pts + num_halo_coarse_pts;
        // Using the number of owned_coarse_pts (number of rows of Ac), initialize P manager
        // This will set the part_offsets array, index_range, halo_offsets[0] value
        prep->initialize_manager(A, P, num_owned_coarse_pts);
    }
    else
    {
        coarsePoints = (int) thrust_wrapper::count_if(cf_map.begin(), cf_map.end(), is_non_neg());
        cudaCheckError();
    }

    // mark as 1 if a coarse point
    int size_one_ring;

    if (!A.is_matrix_singleGPU())
    {
        int offset;
        A.getOffsetAndSizeForView(FULL, &offset, &size_one_ring);
    }
    else
    {
        size_one_ring = A.get_num_rows();
    }

    int numBlocksOneRing = min( 4096, (int) (size_one_ring + blockSize - 1) / blockSize );
    int numBlocks = min( 4096, (int) (A.get_num_rows() + blockSize - 1) / blockSize );
    // count the number of non-zeros in the interpolation matrix
    IntVector nonZeroOffsets(A.get_num_rows() + 1);
    IntVector nonZerosPerRow(A.get_num_rows());
    int *nonZerosPerRow_ptr = nonZerosPerRow.raw();

    if (A.get_num_rows() > 0)
    {
        nonZerosPerRowKernel <<< numBlocks, blockSize>>>(Anum_rows, cf_map.raw(), C_hat_start_ptr, C_hat_end_ptr,
                nonZerosPerRow_ptr);
        cudaCheckError();
    }

    // get total with a reduction
    int nonZeros = thrust_wrapper::reduce(nonZerosPerRow.begin(), nonZerosPerRow.end());
    cudaCheckError();
    // get the offsets with an exclusive scan
    thrust_wrapper::exclusive_scan(nonZerosPerRow.begin(), nonZerosPerRow.end(), nonZeroOffsets.begin());
    cudaCheckError();
    nonZeroOffsets[A.get_num_rows()] = nonZeros;
    // resize P
    P.resize(0, 0, 0, 1);
    P.addProps(CSR);
    P.resize(A.get_num_rows(), coarsePoints, nonZeros, 1);
    IndexType *Poffsets = P.row_offsets.raw();
    IndexType *Pcolumn_indices = P.col_indices.raw();
    ValueType *Pvalues = P.values.raw();
    // copy nonzero offsets to the P matrix
    amgx::thrust::copy(nonZeroOffsets.begin(), nonZeroOffsets.end(), P.row_offsets.begin());
    cudaCheckError();
    // grab the diagonal terms
    VVector diag(size_one_ring);
    ValueType *diag_ptr = diag.raw();

    if (A.get_num_rows() > 0)
    {
        find_diag_kernel_indexed_dia <<< numBlocksOneRing, blockSize>>>(
            size_one_ring,
            A.diag.raw(),
            A.values.raw(),
            diag.raw());
    }

    cudaCheckError();
    IntVector innerSumOffset( A.get_num_rows() + 1, 0 );

    if (A.get_num_rows() > 0)
    {
        const int kCtaSize = 256;
        const int nWarps = kCtaSize / 32;
        const int grid_size = std::min( 4096, ( Anum_rows + nWarps - 1 ) / nWarps );
        calculateInnerSumStorageKernel<IndexType, kCtaSize> <<< grid_size, kCtaSize>>>(
            Aoffsets,
            Acolumn_indices,
            Anum_rows,
            cf_map.raw(),
            s_con.raw(),
            innerSumOffset.raw() );
        cudaCheckError();
    }

    // get the offsets
    thrust_wrapper::exclusive_scan( innerSumOffset.begin(), innerSumOffset.end(), innerSumOffset.begin() );
    cudaCheckError();
    // assign memory & get pointer
    int numInnerSum = innerSumOffset[A.get_num_rows()];
    VVector innerSum( numInnerSum, 0 );
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = (A.get_num_rows() == 0) ? 0 : A.get_num_nz() / A.get_num_rows();
        // Compute the set C_hat.
        int work_offset = grid_size * NUM_WARPS;
        cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
        // Run the computation.
        typedef typename MatPrecisionMap<t_matPrec>::Type Value_type;
        {
            distance2::compute_inner_sum_kernel<Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat.raw(),
                C_hat_pos.raw(),
                C_hat_start.raw(),
                C_hat_end.raw(),
                diag.raw(),
                innerSumOffset.raw(),
                innerSum.raw(),
                exp_wk.get_gmem_size(),
                exp_wk.get_keys(),
                exp_wk.get_work_queue());
        }
        cudaCheckError();
    }
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = (A.get_num_rows() == 0) ? 0 : A.get_num_nz() / A.get_num_rows();
        // Compute the set C_hat.
        int work_offset = grid_size * NUM_WARPS;
        cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
        // Run the computation.
        typedef typename MatPrecisionMap<t_matPrec>::Type Value_type;
        distance2::compute_interp_weight_kernel<Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
            A.get_num_rows(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            A.values.raw(),
            cf_map.raw(),
            s_con.raw(),
            C_hat.raw(),
            C_hat_pos.raw(),
            C_hat_start.raw(),
            C_hat_end.raw(),
            diag.raw(),
            innerSumOffset.raw(),
            innerSum.raw(),
            P.row_offsets.raw(),
            P.col_indices.raw(),
            P.values.raw(),
            exp_wk.get_gmem_size(),
            exp_wk.get_keys(),
            exp_wk.get_vals(),
            exp_wk.get_work_queue());
        cudaCheckError();
    }

    if (A.is_matrix_distributed())
    {
        //TODO: cf_map_global could be smaller (only contain halo nodes)
        I64Vector_d cf_map_global(cf_map.size());
        int num_owned_fine_pts = P.get_num_rows();
        int my_rank = A.manager->global_id();
        const int cta_size = 128;
        const int grid_size = std::min( 4096, (num_owned_fine_pts + cta_size - 1) / cta_size);

        // Write the global coarse index of all the owned fine points (-1 if not a coarse point)
        if (num_owned_fine_pts > 0)
        {
            createCfMapGlobal <<< grid_size, cta_size>>>(cf_map.raw(), cf_map_global.raw(), P.manager->part_offsets_h[my_rank], num_owned_fine_pts);
            cudaCheckError();
        }

        // Exchange the cf_map_global so that we know the coarse global id of halo nodes
        cf_map_global.dirtybit = 1;
        A.manager->exchange_halo_2ring(cf_map_global, cf_map_global.tag);
        I64Vector_d P_col_indices_global(P.col_indices.size());

        if (num_owned_fine_pts > 0)
        {
            fill_P_global_col_indices_kernel<16, int> <<< grid_size, cta_size>>>(P.row_offsets.raw(), C_hat_start.raw(), C_hat_end.raw(), C_hat.raw(), cf_map_global.raw(), P_col_indices_global.raw(), num_owned_fine_pts);
            cudaCheckError();
        }

        // Now that we know the global col indices of P, we can find:
        // the list of neighbors, create the B2L_maps for P, create local numbering for halo nodes, create local_to_global_map array, etc...
        prep->initialize_manager_from_global_col_indices(P, P_col_indices_global);
        prep->createRowsLists(P, true);
        delete prep;
    }
}

template< class T_Config>
void Distance2_InterpolatorBase<T_Config>::generateInterpolationMatrix(Matrix<T_Config> &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix<T_Config> &P,
        void *amg)
{
    P.set_initialized(0);
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    // If multi-GPU do a 2-ring exchange of cf_map since this is a distance two interpolator
    if (A.get_block_size() == 1)
    {
        generateInterpolationMatrix_1x1(A, cf_map, s_con, scratch, P, amg);
    }
    else
    {
        FatalError("Unsupported dimensions for distance2 interpolator", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
    P.set_initialized(1);
}

#define AMGX_CASE_LINE(CASE) template class Distance2_InterpolatorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Distance2_Interpolator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
