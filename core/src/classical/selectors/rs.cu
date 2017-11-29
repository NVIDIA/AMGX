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

#include <classical/selectors/rs.h>
#include <classical/interpolators/common.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <set>

#include<thrust/count.h>

namespace amgx
{

namespace classical
{
/*************************************************************************
 * marks the strongest connected (and indepent) points as coarse
 ************************************************************************/

typedef std::pair< int, int> pair_type; // give it a more meaningful name

struct compare
{
    bool operator()(const pair_type &a, const pair_type &b) const
    {
        if (a.first < b.first) { return true; }
        else if ( (a.first == b.first) && (b.second < a.second) ) { return true; }
        else { return false; }
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void RS_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::markCoarseFinePoints_1x1(Matrix_h &A,
                           FVector &weights,
                           const BVector &s_con,
                           IVector &cf_map,
                           IVector &scratch,
                           int cf_map_init)
{
    // Create matrix S_T (for each vertex a list of
    int num_rows = A.get_num_rows();
    int num_nz = A.col_indices.size();
    IVector ST_row_offsets(num_rows + 1);
    IVector ST_col_indices(num_nz);

    for (int i = 0; i <= num_rows; i++)
    {
        ST_row_offsets[i] = 0;
    }

    for (int i = 0; i < num_nz; i++)
    {
        int col = A.col_indices[i];

        if (s_con[i] && col < num_rows)
        {
            ST_row_offsets[col + 1]++;
        }
    }

    for (int i = 0; i < num_rows; i++)
    {
        ST_row_offsets[i + 1] += ST_row_offsets[i];
    }

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            int col = A.col_indices[j];

            if (s_con[j] && col < num_rows)
            {
                ST_col_indices[ST_row_offsets[col]] = i;
                ST_row_offsets[col]++;
            }
        }
    }

    for (int i = num_rows; i > 0; i--)
    {
        ST_row_offsets[i] = ST_row_offsets[i - 1];
    }

    ST_row_offsets[0] = 0;
    // Compute the measure, which is row length of S_T
    IVector i_weights(num_rows);

    for (int i = 0; i < num_rows; i++)
    {
        i_weights[i] = ST_row_offsets[i + 1] - ST_row_offsets[i];
    }

    // Initialize the map
    int num_left = 0;

    for (int j = 0; j < num_rows; j++)
    {
        int isolated = true;

        for (int k = A.row_offsets[j]; k < A.row_offsets[j + 1]; k++)
        {
            // Only consider interior connections
            int col = A.col_indices[k];

            if (col < num_rows && s_con[k] )
            {
                isolated = false;
                break;
            }
        }

        if (isolated)
        {
            if (cf_map_init == 3) { cf_map[j] = COARSE; }
            else { cf_map[j] = STRONG_FINE; }

            i_weights[j] = 0;
        }
        else
        {
            cf_map[j] = UNASSIGNED;
            num_left++;
        }
    }

    std::set<pair_type, compare> nodes_list;

    for (int j = 0; j < num_rows; j++)
    {
        int weight = i_weights[j];

        if (cf_map[j] != STRONG_FINE)
        {
            if (weight > 0)
            {
                nodes_list.insert(pair_type(weight, j));
            }
            else
            {
                cf_map[j] = FINE;

                for (int k = A.row_offsets[j]; k < A.row_offsets[j + 1]; k++)
                {
                    int neighbor = A.col_indices[k];

                    if (s_con[k] && neighbor < num_rows)
                    {
                        if (cf_map[neighbor] != STRONG_FINE)
                        {
                            if (neighbor < j)
                            {
                                int new_weight = i_weights[neighbor];

                                if (new_weight > 0)
                                {
                                    nodes_list.erase(pair_type(new_weight, neighbor));
                                }

                                new_weight = ++(i_weights[neighbor]);
                                nodes_list.insert(pair_type(new_weight, neighbor));
                            }
                            else
                            {
                                int new_weight = ++(i_weights[neighbor]);
                            }
                        }
                    } // if s_con
                }

                --num_left;
            }
        }
    }

    // First pass of Ruge Steuben coarsening

    while (num_left > 0)
    {
        int index = nodes_list.rbegin()->second;
        cf_map[index] = COARSE;
        int weight = i_weights[index];
        i_weights[index] = 0;
        --num_left;
        nodes_list.erase(pair_type(weight, index));

        for (int j = ST_row_offsets[index]; j < ST_row_offsets[index + 1]; j++)
        {
            int neighbor = ST_col_indices[j];

            if (cf_map[neighbor] == UNASSIGNED)
            {
                cf_map[neighbor] = FINE;
                int weight = i_weights[neighbor];
                nodes_list.erase(pair_type(weight, neighbor));
                --num_left;

                for (int k = A.row_offsets[neighbor]; k < A.row_offsets[neighbor + 1]; k++)
                {
                    int d2_neighbor = A.col_indices[k];

                    if (s_con[k] && d2_neighbor < num_rows)
                    {
                        if (cf_map[d2_neighbor] == UNASSIGNED)
                        {
                            int weight = i_weights[d2_neighbor];
                            nodes_list.erase(pair_type(weight, d2_neighbor));
                            int new_weight = ++(i_weights[d2_neighbor]);
                            //printf("d2_neighbor = %d, new_weight = %d \n",d2_neighbor,new_weight);
                            nodes_list.insert(pair_type(new_weight, d2_neighbor));
                        }
                    }
                }
            }
        }

        for (int j = A.row_offsets[index]; j < A.row_offsets[index + 1]; j++)
        {
            int neighbor = A.col_indices[j];

            if (s_con[j] && neighbor < num_rows)
            {
                if (cf_map[neighbor] == UNASSIGNED)
                {
                    int weight = i_weights[neighbor];
                    nodes_list.erase(pair_type(weight, neighbor));
                    i_weights[neighbor] = --weight;

                    if (weight > 0)
                    {
                        nodes_list.insert(pair_type(weight, neighbor));
                    }
                    else
                    {
                        cf_map[neighbor] = FINE;
                        --num_left;

                        for (int k = A.row_offsets[neighbor]; k < A.row_offsets[neighbor + 1]; k++)
                        {
                            int d2_neighbor = A.col_indices[k];

                            if (s_con[k] && d2_neighbor < num_rows)
                            {
                                if (cf_map[d2_neighbor] == UNASSIGNED)
                                {
                                    int new_weight = i_weights[d2_neighbor];
                                    nodes_list.erase(pair_type(new_weight, d2_neighbor));
                                    new_weight = ++(i_weights[d2_neighbor]);
                                    nodes_list.insert(std::pair<const int, int>(new_weight, d2_neighbor));
                                }
                            }
                        }
                    }
                }
            } // if s_con
        }
    }
}

/*************************************************************************
 * Implementing the RS algorith
 ************************************************************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void RS_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::markCoarseFinePoints_1x1(Matrix_d &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    FatalError("Ruge Steuben coarsening not implemented on GPU (it's a sequential algorithm)", AMGX_ERR_NOT_IMPLEMENTED);
}

template <class T_Config>
void RS_SelectorBase< T_Config>::markCoarseFinePoints(Matrix< T_Config> &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    if (A.get_block_size() == 1)
    {
        markCoarseFinePoints_1x1(A, weights, s_con, cf_map, scratch, cf_map_init);
    }
    else
    {
        FatalError("Unsupported block size RS selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class RS_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class RS_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
