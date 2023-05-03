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

#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/serial_greedy_bfs.h>
#include <cusp/format.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <matrix_coloring/bfs.h>
#include <matrix_coloring/coloring_utils.h>
#include <set>

#include <algorithm>

// Pseudo-random number generator
namespace amgx
{

static __host__ __device__ __forceinline__ unsigned int hash_function(unsigned int a, unsigned int seed, unsigned int rows = 0)
{
    a ^= seed;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}

// ---------------------------
// Methods
// ---------------------------

//This recurring template avoids recursions and computes the greedy strategy visiting deeper and deeper neighborhoods.

template<int LEVEL, int DEPTH>
struct neighbor_checker
{
    static void check(const int *A_row_offsets, const int *A_col_indices, int *color, const int row_start, const int row_end, bool *used_colors,
                      const int start_color, int *queue, int &tail, const bool bfs, const bool use_ido_for_bfs)
    {
        for (int row_it = row_start; row_it < row_end; row_it++)
        {
            int c   = A_col_indices[row_it];
            int col = color[c];

            if (col - start_color >= 0) //if color is amongst the tracked ones
            {
                used_colors[col - start_color] = 1; //use the color
            }

            if (!use_ido_for_bfs) //if using normal BFS, enqueue discovered nodes
            {
                if (col == 0 && bfs) //uncolored node discovered and using bfs (this is called also by pgreedy_equivalent)
                {
                    color[c] = -1;    //mark: visited state: each node is enqueued only once
                    queue[tail] = c;  //enqueue
                    tail++;
                }
            }

            //Compile time recursion using templates.
            neighbor_checker < LEVEL - 1, DEPTH + 1 >::check(A_row_offsets, A_col_indices, color, A_row_offsets[c], A_row_offsets[c + 1], used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
    }

};

//Recurring template final step.
template<int DEPTH>
struct neighbor_checker<0, DEPTH>
{
    static void check(const int *A_row_offsets, const int *A_col_indices, int *color, const int row_start, const int row_end, bool *used_colors, const int start_color, int *queue, int &tail, const bool bfs, const bool use_ido_for_bfs)
    {
        //end compile time recursion
    }
};


/*
 * This algorithm runs the Greedy Algorithm using, as coloring order, the order that comes from Breadth-First-Search visit.
 *
 * It's strictly linear time in the number of edges, if the graph is connected.
 * It's has been measured to be much faster than the IDO ordering (next algorithm), while keeping its ability to color bipartite graphs optimally.
 *
 * */
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::run_cpu_serial_bfs_greedy(
    Matrix_d &A, int *color, int *queue, int *A_row_offsets, int *A_col_indices
)
{
    bool use_ido_for_bfs = false;
    bool bfs = true;
    int num_rows = A.get_num_rows();
    int num_nonzero = A.get_num_nz();

    //initialize to 0
    for (int r = 0; r < num_rows; r++) { color[r] = 0; }

    int max_color_used = 0;
    int head = 0;
    int tail = 0;
    //add the central node to the BFS queue
    queue[tail++] = num_rows / 2;
    color[num_rows / 2] = -1; //mark visited
    const int MAXCOLORS = 128; //if need more colors, it changes start_color += MAXCOLORS
    int start_color = 0;
    int num_uncolored = num_rows;
    int all_colored_before = 0;

    while (num_uncolored > 0) //until all colored
    {
        if (head == tail) //nothing to do and not all colored: it means that the graph is disconnected, restart from an uncolored node
        {
            //TODO find something more efficient.
            for (int i = all_colored_before; i < num_rows; i++)
            {
                if (color[i] == 0)
                {
                    all_colored_before = i;
                    color[i] = -1; //mark as visited
                    queue[tail++] = i;
                    break;
                }
            }

            if (head == tail)
            {
                break;
            }

            //continue;
        }

        int r = queue[head]; //peek BFS queue
        ++head;
        //Task cancellation was here, see P4
        int row_start = A_row_offsets[r  ];
        int row_end   = A_row_offsets[r + 1];
        //used colors buffer
        bool used_colors[MAXCOLORS];

        for (int i = 0; i < MAXCOLORS; i++)
        {
            used_colors[i] = 0;
        }

        if (this->m_coloring_level == 1)
        {
            //duplicate code for clarity: a "neighbor_checker<1,0>::check(&A_row_offsets[0],&A_col_indices[0],&color[0],row_start,row_end,used_colors,start_color, queue, tail, bfs, use_ido_for_bfs);" would be enough
            for (int row_it = row_start; row_it < row_end; row_it++)
            {
                int c   = A_col_indices[row_it];
                int col = color[c];

                if (col - start_color >= 0) //if color is amongst the tracked ones
                {
                    used_colors[col - start_color] = 1; //use the color
                }

                if (col == 0)      //uncolored node discovered
                {
                    color[c] = -1;   //mark: visited state: each node is enqueued only once
                    queue[tail] = c; //enqueue column
                    tail++;
                }
            }
        }
        else if (this->m_coloring_level == 2)
        {
            neighbor_checker<2, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 3)
        {
            neighbor_checker<3, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 4)
        {
            neighbor_checker<4, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 5)
        {
            neighbor_checker<5, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }

        //unsigned long long int available_colors = ~used_colors;
        //int my_color = 64 - utils::bfind( ~used_colors );
        //find the first available color
        int my_color = MAXCOLORS;

        for (int i = 1; i < MAXCOLORS; i++)
        {
            if (used_colors[i] == 0)
            {
                my_color = i + start_color;
                break;
            }
        }

        if (my_color - start_color >= MAXCOLORS - 1)
        {
            start_color += MAXCOLORS; //no available color
        }

        --num_uncolored;
        color[r] = my_color;

        if (my_color > max_color_used)
        {
            max_color_used = my_color; //track the max color used, for counting colors
        }
    }

    this->m_num_colors = max_color_used + 1;
}

/*
 * This algorithm runs the Greedy Algorithm using, as coloring order, the Incidence-Degree-Ordering:
 * the next vertex to be colored is the one with the MOST colored neighbors.
 * */
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::run_cpu_serial_IDO_ordering_greedy(
    Matrix_d &A, int *color, int *queue, int *A_row_offsets, int *A_col_indices
)
{
    bool use_ido_for_bfs = false;
    bool bfs = true;
    int num_rows = A.get_num_rows();
    int num_nonzero = A.get_num_nz();

    for (int r = 0; r < num_rows; r++) { color[r] = 0; }

    int max_color_used = 0;
    int head = 0;
    int tail = 0;
    queue[tail++] = num_rows / 2;
    color[num_rows / 2] = -1; //mark visited
    const int MAXCOLORS = 128; //if need more colors, change start_color += MAXCOLORS
    int start_color = 0;
    typedef std::pair<int, int> ido_entry;
    std::set<ido_entry> ido_priority_queue;
    std::vector<int> ido_vertex_colored_count;
    tail = num_rows;
    ido_vertex_colored_count.resize(num_rows);
    ido_priority_queue.insert(ido_entry(0, num_rows / 2));
    int num_uncolored = num_rows;
    int all_colored_before = 0; //all vertices all colored, before this

    while (num_uncolored > 0)
    {
        if (head == tail) //nothing to do
        {
            //TODO find something more efficient. This happens only if the graph is disconnected.
            for (int i = all_colored_before; i < num_rows; i++)
            {
                if (color[i] == 0)
                {
                    all_colored_before = i;
                    color[i] = -1; //mark as visited
                    queue[tail++] = i;
                    break;
                }
            }

            if (head == tail)
            {
                break;
            }

            //continue;
        }

        int r = 0;

        if (ido_priority_queue.empty())
        {
            break;
        }

        //ido takes the vertex with the most colored neighbors
        std::pair<int, int> to_process = *ido_priority_queue.rbegin() ; //peek vertex from queue
        ido_priority_queue.erase(to_process);
        r = ( to_process ).second;
        ++head;
        //Task cancellation was here, see P4
        bool used_colors[MAXCOLORS];

        for (int i = 0; i < MAXCOLORS; i++)
        {
            used_colors[i] = 0;
        }

        int row_start = A_row_offsets[r  ];
        int row_end   = A_row_offsets[r + 1];

        if (this->m_coloring_level == 1)
        {
            neighbor_checker<1, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 2)
        {
            neighbor_checker<2, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 3)
        {
            neighbor_checker<3, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 4)
        {
            neighbor_checker<4, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }
        else if (this->m_coloring_level == 5)
        {
            neighbor_checker<5, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, use_ido_for_bfs);
        }

        //unsigned long long int available_colors = ~used_colors;
        //int my_color = 64 - utils::bfind( ~used_colors );
        int my_color = MAXCOLORS;

        for (int i = 1; i < MAXCOLORS; i++)
        {
            if (used_colors[i] == 0)
            {
                my_color = i + start_color;
                break;
            }
        }

        if (my_color - start_color >= MAXCOLORS - 1)
        {
            start_color += MAXCOLORS;
        }

        //if(color[r] <= 0)
        --num_uncolored;
        color[r] = my_color;

        if (my_color > max_color_used)
        {
            max_color_used = my_color;
        }

        //Priority queue update: update my neighbors' colored neigbors count

        for (int row_it = row_start; row_it < row_end; row_it++)
        {
            int col_id   = A_col_indices[row_it];
            int count = ido_vertex_colored_count[col_id];
            std::pair<int, int> elem(count, col_id);

            if (ido_priority_queue.count(elem))
            {
                ido_priority_queue.erase(elem);
            }

            elem.first = count + 1;

            if (color[col_id] == 0)
            {
                ido_priority_queue.insert(elem);
                ido_vertex_colored_count[col_id] = count + 1;
            }
        }
    }

    this->m_num_colors = max_color_used + 1;
}

/*
 * This algorithm runs the Greedy Algorithm using, as coloring order, the same ordering as in PARALLEL_GREEDY.
 * This is to have a CPU algorithm performing as similar as possible as PARALLEL_GREEDY.
 * This can be easily parallelized.
 * Currently, it works for any-ring but it's not equivalent to PARALLEL_GREEDY (the order of visit is 1-ring).
 * */
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::run_cpu_parallel_greedy_equivalent(
    Matrix_d &A, int *color, int *queue, int *A_row_offsets, int *A_col_indices
)
{
    bool bfs = false;
    int num_rows = A.get_num_rows();
    int num_uncol = num_rows;

    for (int r = 0; r < num_rows; r++)
    {
        color[r] = 0;
        queue[r] = r;
    }

    int max_color_used = 0;
    const int MAXCOLORS = 128; //if need more colors, change start_color += MAXCOLORS
    int start_color = 0;

    while (num_uncol > 0)
    {
        int tail = num_uncol;

        for (int it = 0; it < num_rows; it++)
        {
            int r = it;//this->queue[it];

            //if(skip[r]) continue;
            if (color[r] > 0) { continue; }

            bool is_r_max = true;
            int hash_r = hash_function(r, 0);
            int row_start = A_row_offsets[r  ];
            int row_end   = A_row_offsets[r + 1];

            for (int row_it = row_start; row_it < row_end; row_it++)
            {
                int c   = A_col_indices[row_it];
                int col = color[c];
                int hash_c = hash_function(c, 0);

                if (col == 0 && (hash_c > hash_r || (hash_c == hash_r && c > r)))
                {
                    is_r_max = false;
                    break;
                }
            }

            if (is_r_max == false) { continue; }

            bool used_colors[MAXCOLORS];

            for (int i = 0; i < MAXCOLORS; i++)
            {
                used_colors[i] = 0;
            }

            if (this->m_coloring_level == 1)
            {
                neighbor_checker<1, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, 0);
            }
            else if (this->m_coloring_level == 2)
            {
                neighbor_checker<2, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, 0);
            }
            else if (this->m_coloring_level == 3)
            {
                neighbor_checker<3, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, 0);
            }
            else if (this->m_coloring_level == 4)
            {
                neighbor_checker<4, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, 0);
            }
            else if (this->m_coloring_level == 5)
            {
                neighbor_checker<5, 0>::check(&A_row_offsets[0], &A_col_indices[0], &color[0], row_start, row_end, used_colors, start_color, queue, tail, bfs, 0);
            }

            int my_color = MAXCOLORS;
#pragma unroll 8

            for (int i = 1; i < MAXCOLORS; i++)
            {
                if (used_colors[i] == 0)
                {
                    my_color = i + start_color;
                    break;
                }
            }

            if (my_color - start_color >= MAXCOLORS - 1)
            {
                start_color += MAXCOLORS;
            }

            color[ r ] = my_color;
            --tail;

            if (my_color > max_color_used)
            {
                max_color_used = my_color;
            }
        }

        num_uncol = tail;
    }

    this->m_num_colors = max_color_used + 1;
}

//helper structure for std::sort
struct index_color
{
    int index;
    int color;
    bool operator<(const index_color &b) const
    {
        return color < b.color;
    }
};

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::run_createColorArrays_on_cpu(int num_rows, int *color, int *sorted_rows_by_color)
{
    std::vector<index_color> index_colors(num_rows);
    std::vector<int> offsets_rows_per_color(this->m_num_colors + 1);

    for (int i = 0; i < num_rows; i++)
    {
        index_colors[i].index = i;
        index_colors[i].color = color[i];
    }

    {
        std::stable_sort(index_colors.begin(), index_colors.end());
    }

    int prev = -1;

    for (int i = 0; i < num_rows; i++)
    {
        sorted_rows_by_color[i] = index_colors[i].index;
        int col = index_colors[i].color;

        if (col != prev)
        {
            offsets_rows_per_color[col] = i;
        }

        prev = col;
    }

    offsets_rows_per_color[this->m_num_colors] = num_rows;
    this->m_offsets_rows_per_color.resize(this->m_num_colors + 1);

    for (int i = 0; i < this->m_num_colors + 1; i++)
    {
        this->m_offsets_rows_per_color[i] = offsets_rows_per_color[i];
    }
}

/*
 * Memcopies to a pinned buffer: to avoid expensive allocations of arbitrary size pinned memory. Very efficient if the allocated buffer fits in the pinned pool.
 * */
void copy_using_buffer_d2h(void *dst, void *src, size_t size)
{
    static cudaEvent_t event = 0;
    cudaStream_t stream = thrust::global_thread_handle::get_stream();
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming); //TODO it never gets destroyed, allocate somewhere safer

    void *buffer = 0;
    size_t buffer_size = std::min((size_t)(1024 * 1024 * 1), size);
    thrust::global_thread_handle::cudaMallocHost((void **)&buffer, buffer_size);
    size_t offset = 0;

    while (offset < size)
    {
        size_t end = offset + buffer_size;

        if (end > size) { end = size; }

        cudaMemcpyAsync(buffer, ((unsigned char *)src) + offset, end - offset, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(event, stream);
        cudaEventSynchronize(event);
        memcpy(((unsigned char *)dst) + offset, buffer, end - offset);
        offset = end;
    }

    thrust::global_thread_handle::cudaFreeHost(buffer);
}

void copy_using_buffer_h2d(void *dst, void *src, size_t size)
{
    static cudaEvent_t event = 0;

    if (event == 0)
    {
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming); //TODO it never gets destroyed, allocate somewhere safer
    }

    cudaStream_t stream = thrust::global_thread_handle::get_stream();
    void *buffer = 0;
    size_t buffer_size = std::min((size_t)(1024 * 1024 * 1), size);
    thrust::global_thread_handle::cudaMallocHost((void **)&buffer, buffer_size);
    size_t offset = 0;

    while (offset < size)
    {
        size_t end = offset + buffer_size;

        if (end > size) { end = size; }

        memcpy(buffer, ((unsigned char *)src) + offset, end - offset);
        cudaMemcpyAsync( ((unsigned char *)dst) + offset, buffer, end - offset, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(event, stream);
        cudaEventSynchronize(event);
        offset = end;
    }

    thrust::global_thread_handle::cudaFreeHost(buffer);
}

// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::colorMatrix( Matrix_d &A )
{
    typedef typename Matrix<TemplateConfig<AMGX_host, V, M, I> >::IVector IVector_h;
    ViewType oldView = A.currentView();

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    //initialize auxillary data
    int num_rows = A.get_num_rows();
    int *color = new int[num_rows];
    int *A_row_offsets = new int[A.row_offsets.size()];
    int *A_col_indices = new int[A.col_indices.size()];
    int *sorted_rows_by_color = new int[num_rows];
    int *queue = new int[num_rows * 2];
    //Perforn D2H copies
    copy_using_buffer_d2h(A_row_offsets, thrust::raw_pointer_cast(A.row_offsets.data()), A.row_offsets.size()*sizeof(int));
    copy_using_buffer_d2h(A_col_indices, thrust::raw_pointer_cast(A.col_indices.data()), A.col_indices.size()*sizeof(int));
    //Dispatching various cases
    int amg_level = A.template getParameter<int>("level");
    float sparsity = ((float)A.get_num_nz()) / float(A.get_num_rows() * A.get_num_rows());
    //Config argument.
    bool use_ido_for_bfs  = this->m_coloring_custom_arg == "IDO";                    //Incidence-degree-ordering, use to benchmark it against BFS ordering
    bool dummy_coloring   = this->m_coloring_custom_arg == "DUMMY_COLORING";   //Dont' color first 3 levels: emulates JACOBI there
    bool use_bfs          = this->m_coloring_custom_arg != "PARALLEL_GREEDY_EQUIVALENT"; //Run a CPU algorithm which emulated PARALLEL_GREEDY and gives same results
    bool cpu_color_arrays = true;

    //Small and not sparse matrix: use either parallel greedy equivalent (if NOSERIALFIRSTLEVEL specified)
    //or a "serial coloring of the rows"
    if (sparsity > 0.1 && num_rows < 32)
    {
        if (this->m_coloring_custom_arg == "NOSERIALFIRSTLEVEL") // || num_rows > 32)
        {
            run_cpu_parallel_greedy_equivalent(A, color, queue, A_row_offsets, A_col_indices);
        }
        else
        {
            for (int i = 0; i < num_rows; i++)
            {
                color[i] = i + 1;
            }

            this->m_num_colors = num_rows + 1;
        }

        /* //BFS 'natural' visiting order as color.
            for(int i=0; i<num_rows; i++)
                        color[i] = 0;
            int head = 0;
            int tail = 0;
            queue[tail++] = 0;
            int index = 0;
            while(tail > head)
            {
                int r = queue[head];
                color[r] = head;
                ++head;
                int row_start = A_row_offsets[r];
                int row_end   = A_row_offsets[r+1];
                for(int row_it=row_start; row_it<row_end; row_it++)
                {
                    int c   = A_col_indices[row_it];
                    if(color[c] == 0)
                    {
                        queue[tail++] = c;
                    }
                }
            }
            return;*/
    }
    //if dummy coloring is specified: color 3 first levels (1,2,3) with color 1 (emulates a jacobi smoother)
    else if (dummy_coloring && amg_level < 4)
    {
        for (int i = 0; i < num_rows; i++)
        {
            color[i] = 1;
        }

        this->m_num_colors = 2;
    }
    //run the algorithm that emulates PARALLEL_GREEDY on host
    else if (use_bfs == 0)
    {
        run_cpu_parallel_greedy_equivalent(A, color, queue, A_row_offsets, A_col_indices);
    }
    //Use incidence degree ordering as coloring order
    else if (use_ido_for_bfs)
    {
        run_cpu_serial_IDO_ordering_greedy(A,
                                           color,
                                           &queue[0],
                                           A_row_offsets,
                                           A_col_indices
                                          );
    }
    //default behaviour
    else
    {
        run_cpu_serial_bfs_greedy(A,
                                  color,
                                  &queue[0],
                                  A_row_offsets,
                                  A_col_indices
                                 );
    }

    this->m_row_colors.resize(A.get_num_rows());

    if (cpu_color_arrays)
    {
        this->m_sorted_rows_by_color.resize(A.get_num_rows());
        this->run_createColorArrays_on_cpu(A.get_num_rows(), color, sorted_rows_by_color);
        copy_using_buffer_h2d(thrust::raw_pointer_cast(this->m_sorted_rows_by_color.data()), sorted_rows_by_color, A.get_num_rows()*sizeof(int));
    }

    //copies color -> m_row_colors, using a pinned memory buffer
    copy_using_buffer_h2d(thrust::raw_pointer_cast(this->m_row_colors.data()), color, A.get_num_rows()*sizeof(int));
    delete[] color;
    delete[] A_row_offsets;
    delete[] A_col_indices;
    delete[] sorted_rows_by_color;
    delete[] queue;
    A.setView(oldView);
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::createColorArrays(Matrix<TConfig_d> &A)
{
    if (this->ready_for_coloring_arrays == false)
    {
        return;
    }

    MatrixColoring<TConfig_d>::createColorArrays(A);
}

template< class T_Config >
Serial_Greedy_BFS_MatrixColoring_Base<T_Config>::Serial_Greedy_BFS_MatrixColoring_Base( AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    fallback_config = cfg;
    fallback_config_scope = cfg_scope;
    fallback_config.setParameter("matrix_coloring_scheme", std::string("PARALLEL_GREEDY"), fallback_config_scope); //TODO let fallback method choosable with param
    ready_for_coloring_arrays = true;

    //if( this->m_coloring_level != 1 &&  this->m_coloring_level != 2  &&  this->m_coloring_level != 3)
    if ( this->m_coloring_level > 5 || this->m_coloring_level < 0)
    {
        FatalError( "Not implemented for coloring_level != 1", AMGX_ERR_NOT_SUPPORTED_TARGET );
    }

    m_coloring_custom_arg = cfg.AMG_Config::template getParameter<std::string>( "coloring_custom_arg", cfg_scope );
    m_coloring_try_remove_last_color_ = cfg.AMG_Config::template getParameter<int>( "coloring_try_remove_last_colors", cfg_scope );
}

#define AMGX_CASE_LINE(CASE) template class Serial_Greedy_BFS_MatrixColoring_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Serial_Greedy_BFS_MatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx

