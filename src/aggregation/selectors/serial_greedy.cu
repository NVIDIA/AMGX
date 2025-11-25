// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// ---------------------------------------------------------
//  Dummy selector (permits comparison between host and device codes
// --------------------------------------------------------

#include <aggregation/selectors/serial_greedy.h>
#include <cutil.h>
#include <types.h>
#include <basic_types.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif

#ifdef _WIN32
#pragma warning (pop)
#endif

namespace amgx
{
namespace aggregation
{

// Constructor
template<class T_Config>
SerialGreedySelector<T_Config>::SerialGreedySelector(AMG_Config &cfg, const std::string &cfg_scope)
{
    aggregate_size = cfg.AMG_Config::template getParameter<int>("aggregate_size", cfg_scope);
    edge_weight_component = cfg.AMG_Config::template getParameter<int>("aggregation_edge_weight_component", cfg_scope);
}

//  setAggregates for block_dia_csr_matrix_h format
template <class T_Config>
void SerialGreedySelector<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &numAggregates)
{
    if ( A.get_block_dimx() != A.get_block_dimy() )
    {
        FatalError("Unsupported: Blockdim x must equal Blockdim y.", AMGX_ERR_BAD_PARAMETERS);
    }

    IndexType numRows = A.get_num_rows();
    IndexType nnz = A.get_num_nz();
    int bsize = A.get_block_dimx();
    int bsize_sq = bsize * bsize;
    int ewc = bsize * this->edge_weight_component + this->edge_weight_component;
    IndexType nnzj = nnz;

    if ( A.hasProps( DIAG ) )
    {
        nnz += numRows;
    }

    IndexType *ia, *ja, *diag, *agg;
    ValueType *data;
    typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;

    if ( typeid( MemorySpace ) == typeid( device_memory ) )
    {
        //copy data to host
        ia = new IndexType[numRows + 1];
        ja = new IndexType[nnzj];
        data = new ValueType[nnz * bsize_sq];
        diag = new IndexType[numRows];
        cudaMemcpy( ia, A.row_offsets.raw(), (numRows + 1)*sizeof(IndexType), cudaMemcpyDeviceToHost );
        cudaCheckError();
        cudaMemcpy( ja, A.col_indices.raw(), nnzj * sizeof(IndexType), cudaMemcpyDeviceToHost );
        cudaCheckError();
        cudaMemcpy( data, A.values.raw(), nnz * bsize_sq * sizeof(ValueType ), cudaMemcpyDeviceToHost );
        cudaCheckError();
        cudaMemcpy( diag, A.diag.raw(), numRows * sizeof(IndexType), cudaMemcpyDeviceToHost );
        cudaCheckError();
        //create aggregates
        agg = new IndexType[numRows];
    }
    else
    {
        //reference A
        ia = A.row_offsets.raw();
        ja = A.col_indices.raw();
        data = A.values.raw();
        diag = A.diag.raw();
        //and aggregates
        aggregates.resize(numRows);
        agg = aggregates.raw();
    }

    //init stack and put node with minimum degree on top of it
    IndexType *stack = new IndexType[numRows];
    IndexType top = -1;
    IndexType minDegree = numRows;
    IndexType maxDegree = 0;

    for (IndexType i = 0; i < numRows; i++)
    {
        agg[i] = -1;

        if ( ia[i + 1] - ia[i] < minDegree )
        {
            //mark old top as uninitialized
            if ( top != -1 )
            {
                stack[top] = -2;
            }

            //put i on top instead
            top = i;
            stack[i] = -1;
            //update min degree
            minDegree = ia[i + 1] - ia[i];
        }
        else
        {
            stack[i] = -2;
        }

        if ( ia[i + 1] - ia[i] > maxDegree )
        {
            maxDegree = ia[i + 1] - ia[i];
        }
    }

    //holds the nodes of the aggregate that is currently being formed
    IndexType *curNodes = new IndexType[this->aggregate_size];
    numAggregates = 0;

    while ( top != -1 )
    {
        //pop top node from stack
        IndexType seed = top;
        top = stack[top];

        if ( agg[seed] != -1 )
        {
            continue;
        }

        //form aggregate
        agg[seed] = numAggregates;
        curNodes[0] = seed;

        for (int size = 1; size < this->aggregate_size; size++)
        {
            ValueType maxWeight = 0.0;
            IndexType newNode = -1;

            for (IndexType curSize = 0; curSize < size; curSize++)
            {
                IndexType node = curNodes[curSize];

                for (IndexType ii = ia[node]; ii < ia[node + 1]; ii++)
                {
                    IndexType j = ja[ii];

                    if ( agg[j] != -1)
                    {
                        continue;
                    }

                    IndexType jj;
                    ValueType weight = 0.0;

                    for (jj = ia[j]; jj < ia[j + 1]; jj++)
                    {
                        IndexType k = ja[jj];

                        if ( agg[k] == numAggregates )
                        {
                            weight += -data[jj * bsize_sq + ewc] / data[diag[k] * bsize_sq + ewc];    //weight formula is not symmetric
                        }
                    }

                    if ( weight > maxWeight )
                    {
                        //add old maximum to stack
                        if ( newNode != -1 && stack[newNode] == -2 ) //only add to stack once
                        {
                            stack[newNode] = top;
                            top = newNode;
                        }

                        maxWeight = weight;
                        newNode = j;
                    }
                    else if ( stack[j] == -2 )
                    {
                        //no max: add to stack
                        stack[j] = top;
                        top = j;
                    }
                }
            }

            //no neighbor left to aggregate, then leave aggregate as it is
            if ( newNode == -1 )
            {
                if ( size < 3 && size <= this->aggregate_size / 2)
                {
                    //aggregate too small: destroy and merge later
                    for (int curSize = 0; curSize < size; curSize++)
                    {
                        agg[curNodes[curSize]] = -1;
                    }

                    numAggregates--;
                }

                break;
            }

            agg[newNode] = numAggregates;
            curNodes[size] = newNode;
        }

        numAggregates++;
    }

    IndexType *neighborAgg = new IndexType[maxDegree];
    ValueType *neighborStrength = new ValueType[maxDegree];
    int singletons = 0;

    for (IndexType i = 0; i < numRows; i++)
    {
        if ( agg[i] == -1 )
        {
            singletons++;
            //compute strength of connections to neighbor aggregates
            IndexType numNeighbors = 0;

            for (IndexType ii = ia[i]; ii < ia[i + 1]; ii++)
            {
                IndexType j = ja[ii];

                if ( agg[j] == -1)
                {
                    continue;
                }

                ValueType weight = -0.5 * (data[ii * bsize_sq + ewc] / data[diag[i] * bsize_sq + ewc] + data[ii * bsize_sq + ewc] / data[diag[j] * bsize_sq + ewc]); //weight formula assumes symmetric matrix
                neighborAgg[numNeighbors] = agg[j];
                neighborStrength[numNeighbors] = weight;
                numNeighbors++;
            }

            //reduce neighbor array to sum
            for (int jj = 1; jj < 0; jj++)
            {
                for (int kk = 0; kk < jj; kk++)
                {
                    if ( neighborAgg[kk] == neighborAgg[jj] )
                    {
                        neighborStrength[kk] += neighborStrength[jj];
                        neighborStrength[jj] = 0;
                        break;
                    }
                }
            }

            //find strongest neighbor
            IndexType newAgg = -1;
            ValueType maxStrength = 0.0;

            for (int jj = 0; jj < numNeighbors; jj++)
            {
                if ( neighborStrength[jj] > maxStrength )
                {
                    maxStrength = neighborStrength[jj];
                    newAgg = neighborAgg[jj];
                }
            }

            //...and join
            if ( newAgg != -1 )
            {
                agg[i] = newAgg;
            }
            else
            {
                agg[i] = numAggregates;
                numAggregates++;
            }
        }
    }

    //have to copy back and delete all pointers
    if ( typeid( MemorySpace ) == typeid( device_memory ) )
    {
        aggregates.resize( numRows );
        cudaError_t cuda_rc = cudaMemcpy( aggregates.raw(), agg, numRows * sizeof(IndexType), cudaMemcpyHostToDevice );
        if (cuda_rc != cudaSuccess)
        {
            FatalError("cudaMemcpy aggregates H2D failed in serial_greedy", AMGX_ERR_CUDA_FAILURE);
        }
        delete[] agg;
        delete[] ia;
        delete[] ja;
        delete[] data;
        delete[] diag;
    }

    delete[] curNodes;
    delete[] stack;
    delete[] neighborAgg;
    delete[] neighborStrength;
    //this->assertAggregates( aggregates, numAggregates );
}
// ---------------------------
//  Explict instantiations
// ---------------------------
#define AMGX_CASE_LINE(CASE) template class SerialGreedySelector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
