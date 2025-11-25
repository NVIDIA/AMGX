// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/locally_downwind.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <queue>
#include <iostream>

namespace amgx
{
namespace locally_downwind_kernels
{
// ---------------------------
// Kernels
// ---------------------------

// Kernel to color the rows of the matrix, using min-max approach
template <typename IndexType>
__global__
void colorRowsKernel(IndexType *row_colors, const int num_colors, const int num_rows)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        row_colors[i] = i % num_colors;
    }
}


//is this edge and outgoing edge?
template <typename IndexType, typename ValueType>
__host__ __device__
bool outgoing_edge( const IndexType *ia, const IndexType *ja, const ValueType *aa, IndexType i, IndexType ii, int blocksize )
{
    IndexType j = ja[ii];
    ValueType weight = 0.0;

    for (IndexType iii = ii * blocksize; iii < (ii + 1)*blocksize; iii++)
    {
        weight += (aa[iii] * aa[iii]);
    }

    for (IndexType jj = ia[j]; jj < ia[j + 1]; jj++)
    {
        if ( ja[jj] == i )
        {
            ValueType counter_weight = 0.0;

            for (IndexType jjj = jj * blocksize; jjj < (jj + 1)*blocksize; jjj++)
            {
                counter_weight += (aa[jjj] * aa[jjj]);
            }

            return (weight >= counter_weight);
        }
    }

    return true;
}

template <typename IndexType>
struct Buffer
{
    IndexType *buffer;
    int buffersize;
    int loptr;
    int hiptr;
    int offset;

    __device__ int size();
    __device__ IndexType pop();
    __device__ void push( IndexType node );
};

template <typename IndexType>
__device__
IndexType Buffer<IndexType>::pop()
{
    int pos = loptr % buffersize;
    loptr++;
    return buffer[offset + pos];
}

template <typename IndexType>
__device__
void Buffer<IndexType>::push(IndexType element )
{
    buffer[offset + hiptr] = element;
    hiptr = (hiptr + 1) % buffersize;
}

template <typename IndexType>
__device__
int Buffer<IndexType>::size()
{
    return (hiptr - loptr) % buffersize;
}


template <typename IndexType, typename ValueType>
__global__
void traverse( const IndexType *ia, const IndexType *ja, const ValueType *aa, const IndexType *ria, const IndexType *rja, const IndexType *agg, IndexType *color, IndexType numAggregates, int buffersize, int blocksize )
{
    extern __shared__ IndexType tr_smem[];
    Buffer<IndexType> ring;
    ring.buffer = &tr_smem[0];
    ring.buffersize = buffersize;
    ring.loptr = 0;
    ring.hiptr = 0;
    ring.offset = threadIdx.x * buffersize;
    int aggregate = threadIdx.x + blockDim.x * blockIdx.x;

    if ( aggregate < numAggregates )
    {
        bool nodesLeft = true;

        for (int allowed_incoming_edges = 0; nodesLeft; allowed_incoming_edges++)
        {
            nodesLeft = false;

            for (IndexType ii = ria[aggregate]; ii < ria[aggregate + 1]; ii++)
            {
                //find possible root node for modified BFS coloring
                IndexType node = rja[ii];
                int found_edges = 0;

                //not colored yet
                if ( color[node] == -1 )
                {
                    for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                    {
                        if ( agg[ja[jj]] == aggregate && !outgoing_edge( ia, ja, aa, node, jj, blocksize ) )
                        {
                            found_edges++;
                        }
                    }

                    nodesLeft = true;
                }
                else
                {
                    found_edges = allowed_incoming_edges + 1;
                }

                //start modified BFS
                if ( found_edges <= allowed_incoming_edges )
                {
                    color[node] = 0; //TODO: make the node look around for already set neighbors first
                    ring.push(node);

                    while ( ring.size() > 0 )
                    {
                        //this node is already colored.
                        node = ring.pop();
                        int myInitialColor = color[node];
                        //traverse all neighbors to determin minimum own color
                        int myColor = myInitialColor;

                        for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                        {
                            IndexType j = ja[jj];

                            if ( color[j] == myColor && j != node)
                            {
                                //try next color
                                myColor++;
                                jj = ia[node];
                            }
                        }

                        //repair own color
                        //color[node] = myColor;

                        //traverse all children
                        for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                        {
                            IndexType child = ja[jj];

                            if ( agg[child] != aggregate || //only set colors in own aggregate
                                    !outgoing_edge( ia, ja, aa, node, jj, blocksize )) //only process outgoing edges
                            {
                                continue;
                            }

                            //unset -> set and push
                            if ( color[child] == -1)
                            {
                                color[child] = myColor + 1;
                                ring.push( child );
                            }

                            if ( color[child] >= myInitialColor )
                            {
                                color[child] = myColor + 1; //max( myColor+1, color[child] );
                                ring.push( child );
                            }

                            /*
                            //same color: reset and reset children, no push
                            if( color[child] == myColor )
                            {
                                color[child] = myColor+1;
                                for(IndexType kk = ia[child]; kk < ia[child+1]; kk++)
                                {
                                    //same restrictions for setting colors apply here:
                                    //stay in aggregate, only set outgoing edges, only set uncolored or same level
                                    IndexType grandchild = ja[kk];
                                    if( agg[grandchild] != aggregate || !outgoing_edge( ia, ja, aa, child, kk, blocksize ) )
                                        continue;

                                    if( color[grandchild] == -1 )
                                    {
                                        color[grandchild] = myColor+2;
                                        ring.push( grandchild );
                                    }
                                    if( color[grandchild] == myColor+1 )
                                        color[grandchild] = myColor+2;
                                }
                            }
                            */
                        }
                    }
                }
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__
void repair( const IndexType *ia, const IndexType *ja, const ValueType *aa, const IndexType *ria, const IndexType *rja, const IndexType *agg, IndexType *color, IndexType numAggregates, int buffersize, int blocksize )
{
    extern __shared__ IndexType tr_smem[];
    Buffer<IndexType> ring;
    ring.buffer = &tr_smem[0];
    ring.buffersize = buffersize;
    ring.loptr = 0;
    ring.hiptr = 0;
    ring.offset = threadIdx.x * buffersize;
    int aggregate = threadIdx.x + blockDim.x * blockIdx.x;

    if ( aggregate < numAggregates )
    {
        bool nodesLeft = true;

        for (int allowed_incoming_edges = 0; nodesLeft; allowed_incoming_edges++)
        {
            nodesLeft = false;

            for (IndexType ii = ria[aggregate]; ii < ria[aggregate + 1]; ii++)
            {
                //find possible root node for modified BFS coloring
                IndexType node = rja[ii];
                int found_edges = 0;

                //not colored yet
                if ( color[node] == -1 )
                {
                    for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                    {
                        if ( agg[ja[jj]] == aggregate && !outgoing_edge( ia, ja, aa, node, jj, blocksize ) )
                        {
                            found_edges++;
                        }
                    }

                    nodesLeft = true;
                }
                else
                {
                    found_edges = allowed_incoming_edges + 1;
                }

                //start modified BFS
                if ( found_edges <= allowed_incoming_edges )
                {
                    color[node] = 0; //TODO: make the node look around for already set neighbors first
                    ring.push(node);

                    while ( ring.size() > 0 )
                    {
                        //this node is already colored.
                        node = ring.pop();
                        int myInitialColor = color[node];
                        //traverse all neighbors to determin minimum own color
                        int myColor = myInitialColor;

                        for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                        {
                            IndexType j = ja[jj];

                            if ( color[j] == myColor && j != node)
                            {
                                //try next color
                                myColor++;
                                jj = ia[node];
                            }
                        }

                        //traverse all children
                        for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                        {
                            IndexType child = ja[jj];

                            if ( agg[child] != aggregate || //only set colors in own aggregate
                                    !outgoing_edge( ia, ja, aa, node, jj, blocksize )) //only process outgoing edges
                            {
                                continue;
                            }

                            //unset -> set and push
                            if ( color[child] == -1)
                            {
                                color[child] = myColor + 1;
                                ring.push( child );
                            }

                            if ( color[child] >= myInitialColor )
                            {
                                color[child] = myColor + 1; //max( myColor+1, color[child] );
                                ring.push( child );
                            }

                            /*
                            //same color: reset and reset children, no push
                            if( color[child] == myColor )
                            {
                                color[child] = myColor+1;
                                for(IndexType kk = ia[child]; kk < ia[child+1]; kk++)
                                {
                                    //same restrictions for setting colors apply here:
                                    //stay in aggregate, only set outgoing edges, only set uncolored or same level
                                    IndexType grandchild = ja[kk];
                                    if( agg[grandchild] != aggregate || !outgoing_edge( ia, ja, aa, child, kk, blocksize ) )
                                        continue;

                                    if( color[grandchild] == -1 )
                                    {
                                        color[grandchild] = myColor+2;
                                        ring.push( grandchild );
                                    }
                                    if( color[grandchild] == myColor+1 )
                                        color[grandchild] = myColor+2;
                                }
                            }
                            */
                        }
                    }
                }
            }
        }
    }
}



}//locally_downwind_kernels namepsace
// ---------------------------
// Methods
// ---------------------------

template<class T_Config>
LocallyDownwindColoringBase<T_Config>::LocallyDownwindColoringBase(AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    this->m_num_colors = cfg.AMG_Config::template getParameter<int>("num_colors", cfg_scope);
}

template<class TConfig>
void LocallyDownwindColoringBase<TConfig>::colorMatrix(Matrix<TConfig> &A)
{
    //wait for colorMatrixUsingAggregates to be called
    std::cout << "coloring denied" << std::endl;
    return;
}

template<class TConfig>
void LocallyDownwindColoringBase<TConfig>::colorMatrixUsingAggregates(Matrix<TConfig> &A, IVector &R_row_offsets, IVector &R_col_indices, IVector &aggregates )
{
#define CPU_VERSION
#ifdef CPU_VERSION
    IndexType numRows = A.get_num_rows();
    IndexType nnz = A.get_num_nz();
    int blockdim = A.get_block_dimx() * A.get_block_dimy();
    //allocate memory on host
    IndexType *ia = new IndexType[numRows + 1];
    IndexType *ja = new IndexType[A.col_indices.size()];
    ValueType *aa = new ValueType[nnz * A.get_block_dimx()*A.get_block_dimy()];
    IndexType *ria = new IndexType[R_row_offsets.size()];
    IndexType *rja = new IndexType[R_col_indices.size()];
    IndexType *agg = new IndexType[aggregates.size()];
    IndexType *color = new IndexType[numRows];
    //copy data from device to host
    cudaMemcpy(ia, A.row_offsets.raw(), (numRows + 1)*sizeof(IndexType), cudaMemcpyDeviceToHost );
    cudaCheckError();
    cudaMemcpy(ja, A.col_indices.raw(), A.col_indices.size()*sizeof(IndexType), cudaMemcpyDeviceToHost );
    cudaCheckError();
    cudaMemcpy(aa, A.values.raw(), nnz * blockdim * sizeof(ValueType), cudaMemcpyDeviceToHost );
    cudaCheckError();
    cudaMemcpy(ria, R_row_offsets.raw(), R_row_offsets.size()*sizeof(IndexType), cudaMemcpyDeviceToHost );
    cudaCheckError();
    cudaMemcpy(rja, R_col_indices.raw(), R_col_indices.size()*sizeof(IndexType), cudaMemcpyDeviceToHost );
    cudaCheckError();
    cudaMemcpy(agg, aggregates.raw(), aggregates.size()*sizeof(IndexType), cudaMemcpyDeviceToHost );
    cudaCheckError();

    for (IndexType i = 0; i < numRows; i++)
    {
        color[i] = -1;
    }

    //color aggregate by aggregate
    for (IndexType aggregate = 0; aggregate < R_row_offsets.size() - 1; aggregate++)
    {
        std::queue<IndexType> q;
        bool nodesLeft = true;

        while ( nodesLeft )
        {
            //find uncolored node to start with (i.e. with minimum in degree)
            IndexType min_in_degree = numRows;
            IndexType next_node = -1;
            nodesLeft = false;

            for (IndexType node_index = ria[aggregate]; node_index < ria[aggregate + 1]; node_index++)
            {
                IndexType node = rja[node_index];
                int in_degree = 0;

                if ( color[node] == -1 )
                {
                    nodesLeft = true;

                    for (IndexType ii = ia[node]; ii < ia[node + 1]; ii++)
                    {
                        if ( agg[ja[ii]] == aggregate && !locally_downwind_kernels::outgoing_edge( ia, ja, aa, node, ii, blockdim ) )
                        {
                            in_degree++;
                        }
                    }

                    if ( in_degree < min_in_degree )
                    {
                        min_in_degree = in_degree;
                        next_node = node;
                    }
                }
            }

            if (!nodesLeft)
            {
                break;
            }

            //start modified BFS
            color[next_node] = 0;
            q.push( next_node );

            while ( q.size() > 0 )
            {
                IndexType node = q.front();
                q.pop();
                int myInitialColor = color[node];
                int myColor = myInitialColor;

                //find valid color for this node
                for (IndexType ii = ia[node]; ii < ia[node + 1]; ii++)
                {
                    if ( color[ja[ii]] == myColor && ja[ii] != node && (agg[ja[ii]] != aggregate || !locally_downwind_kernels::outgoing_edge( ia, ja, aa, node, ii, blockdim )) )
                    {
                        myColor++;
                        ii = ia[node] - 1;
                    }
                }

                //set color
                color[node] = myColor;

                //update children
                for (IndexType jj = ia[node]; jj < ia[node + 1]; jj++)
                {
                    IndexType child = ja[jj];

                    if ( agg[child] != aggregate || //only set colors in own aggregate
                            !locally_downwind_kernels::outgoing_edge( ia, ja, aa, node, jj, blockdim ) ||//only process outgoing edges
                            child == node)  // don't mess with yourself
                    {
                        continue;
                    }

                    //unset -> set and push
                    if ( color[child] >= myInitialColor )
                    {
                        color[child] = max( myColor + 1, color[child] );
                        q.push( child );
                    }

                    if ( color[child] == -1)
                    {
                        color[child] = myColor + 1;
                        q.push( child );
                    }
                }
            }
        }
    }

    //copy back results
    this->m_row_colors.resize(A.row_offsets.size() - 1, 0);
    cudaMemcpy( this->m_row_colors.raw(), color, numRows * sizeof(IndexType), cudaMemcpyHostToDevice );
    cudaCheckError();
    //free all the others
    delete [] ia;
    delete [] ja;
    delete [] aa;
    delete [] agg;
    delete [] ria;
    delete [] rja;
    delete [] color;
#else
    std::cout << "coloring with aggregate information" << std::endl;
    ViewType oldView = A.currentView();
    this->m_row_colors.resize(A.row_offsets.size() - 1, 0);

    if    (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    if (this->m_coloring_level == 0)
    {
        FatalError("Calling coloring scheme but coloring level==0", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else if (this->m_coloring_level == 1)
    {
        IndexType numRows = A.get_num_rows();
        IndexType numAggregates = R_row_offsets.size() - 1;
        int blocksize = A.get_block_dimx() * A.get_block_dimy();
        int max_aggregate_size = 100;
        const int threads_per_block = 64;
        const int num_blocks = (numAggregates - 1) / threads_per_block + 1;
        const int smem_size = max_aggregate_size * threads_per_block * sizeof(IndexType);
        this->m_row_colors.resize( numRows );
        thrust_wrapper::fill( this->m_row_colors.begin(), this->m_row_colors.end(), -1 );
        cudaCheckError();
        std::cout << "start coloring kernel" << std::endl;
        locally_downwind_kernels::traverse <<< num_blocks, threads_per_block, smem_size>>>( A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                R_row_offsets.raw(),
                R_col_indices.raw(),
                aggregates.raw(),
                this->m_row_colors.raw(),
                numAggregates,
                max_aggregate_size,
                blocksize);
                cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();
        std::cout << "uncolored nodes: " << amgx::thrust::count( this->m_row_colors.begin(), this->m_row_colors.end(), -1 ) << std::endl;
    }
    else
    {
        FatalError("Locally Downwind coloring algorithm can only do one ring coloring", AMGX_ERR_NOT_IMPLEMENTED);
    }

    A.setView(oldView);
#endif
}

// Block version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void LocallyDownwindColoring<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_d &A)
{
    FatalError("This method is no longer needed", AMGX_ERR_NOT_IMPLEMENTED);
    /*
    // One thread per row
    const int num_rows = A.get_num_rows();
    IndexType *row_colors_ptr = this->m_row_colors.raw();

    const int threads_per_block = 64;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (int) (num_rows-1)/threads_per_block + 1);

    locally_downwind_kernels::colorRowsKernel<IndexType> <<<num_blocks,threads_per_block>>>(row_colors_ptr, this->m_num_colors, num_rows);
    cudaCheckError();
    */
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void LocallyDownwindColoring<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_h &A)
{
    FatalError("Haven't implemented locally downwind coloring for host", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

#define AMGX_CASE_LINE(CASE) template class LocallyDownwindColoringBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class LocallyDownwindColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx

