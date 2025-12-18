// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/copy.h>
#include <cusp/dia_matrix.h>

#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/functional.h>

namespace cusp
{
namespace gallery
{
namespace detail
{

template <typename StencilPoint, typename GridDimension, typename IndexType, int i, int n>
struct inside_grid_helper
{
    __host__ __device__
    static bool inside_grid(StencilPoint point, GridDimension grid, IndexType index)
    {
        IndexType x = index % amgx::thrust::get<i>(grid) + amgx::thrust::get<i>(amgx::thrust::get<0>(point));
        
        if (x < 0 || x >= amgx::thrust::get<i>(grid))
            return false;
        else
            return inside_grid_helper<StencilPoint,GridDimension,IndexType,i + 1,n>::inside_grid(point, grid, index / amgx::thrust::get<i>(grid));
    }
};

template <typename StencilPoint, typename GridDimension, typename IndexType, int n>
struct inside_grid_helper<StencilPoint,GridDimension,IndexType,n,n>
{
    __host__ __device__
    static bool inside_grid(StencilPoint point, GridDimension grid, IndexType index)
    {
        return true;
    }
};

template <typename StencilPoint, typename GridDimension, typename IndexType>
__host__ __device__
bool inside_grid(StencilPoint point, GridDimension grid, IndexType index)
{
    return inside_grid_helper<StencilPoint,GridDimension,IndexType, 0, amgx::thrust::tuple_size<GridDimension>::value >::inside_grid(point, grid, index);
}


template <typename Tuple, typename UnaryFunction, int i, int size>
struct tuple_for_each_helper
{
    static UnaryFunction for_each(Tuple& t, UnaryFunction f)
    {
        f(amgx::thrust::get<i>(t));

        return tuple_for_each_helper<Tuple,UnaryFunction,i + 1,size>::for_each(t, f);
    }
};

template <typename Tuple, typename UnaryFunction, int size>
struct tuple_for_each_helper<Tuple,UnaryFunction,size,size>
{
    static UnaryFunction for_each(Tuple& t, UnaryFunction f)
    {
        return f;
    }
};

template <typename Tuple, typename UnaryFunction>
UnaryFunction tuple_for_each(Tuple& t, UnaryFunction f)
{
    return tuple_for_each_helper<Tuple, UnaryFunction, 0, amgx::thrust::tuple_size<Tuple>::value>::for_each(t, f);
}


template <typename Tuple, typename OutputIterator>
struct unpack_tuple_functor
{
    OutputIterator output;
    unpack_tuple_functor(OutputIterator output) : output(output) {}

    template <typename T>
    void operator()(T v)
    {
        *output++ = v;
    }
};


template <typename Tuple, typename OutputIterator>
OutputIterator unpack_tuple(const Tuple & t, OutputIterator output)
{
    return tuple_for_each(t, unpack_tuple_functor<Tuple,OutputIterator>(output)).output;
}


template <typename IndexType,
          typename ValueType,
          typename StencilPoint,
          typename GridDimension>
struct fill_diagonal_entries
{
    StencilPoint point;
    GridDimension grid;

    fill_diagonal_entries(StencilPoint point, GridDimension grid)
        : point(point), grid(grid) {}

    __host__ __device__
    ValueType operator()(IndexType index)
    {
        if (inside_grid(point, grid, index))
            return amgx::thrust::get<1>(point);
        else
            return ValueType(0);
    }
};

} // end namespace detail

template <typename IndexType,
          typename ValueType,
          typename MemorySpace,
          typename StencilPoint,
          typename GridDimension>
void generate_matrix_from_stencil(      cusp::dia_matrix<IndexType,ValueType,MemorySpace>& matrix,
                                  const cusp::array1d<StencilPoint,cusp::host_memory>& stencil,
                                  const GridDimension& grid)
{
    IndexType num_dimensions = amgx::thrust::tuple_size<GridDimension>::value;

    cusp::array1d<IndexType,cusp::host_memory> grid_indices(num_dimensions);
    detail::unpack_tuple(grid, grid_indices.begin());

    IndexType num_rows = amgx::thrust::reduce(grid_indices.begin(), grid_indices.end(), IndexType(1), amgx::thrust::multiplies<IndexType>());

    IndexType num_diagonals = stencil.size();

    cusp::array1d<IndexType,cusp::host_memory> strides(grid_indices.size());
    amgx::thrust::exclusive_scan(grid_indices.begin(), grid_indices.end(), strides.begin(), IndexType(1), amgx::thrust::multiplies<IndexType>());

    cusp::array1d<IndexType,cusp::host_memory> offsets(stencil.size(), 0);
    for(size_t i = 0; i < offsets.size(); i++)
    {
        cusp::array1d<IndexType,cusp::host_memory> stencil_indices(num_dimensions);
        detail::unpack_tuple(amgx::thrust::get<0>(stencil[i]), stencil_indices.begin());

        for(IndexType j = 0; j < num_dimensions; j++)
        {
            offsets[i] += strides[j] * stencil_indices[j];
        }
    }

    // TODO compute num_entries directly from stencil
    matrix.resize(num_rows, num_rows, 0, num_diagonals); // XXX we set NNZ to zero for now

    cusp::copy(offsets, matrix.diagonal_offsets);

    // ideally we'd have row views and column views here
    for(IndexType i = 0; i < num_diagonals; i++)
    {
        amgx::thrust::transform(amgx::thrust::counting_iterator<IndexType>(0),
                          amgx::thrust::counting_iterator<IndexType>(num_rows),
                          matrix.values.values.begin() + matrix.values.pitch * i, 
                          detail::fill_diagonal_entries<IndexType,ValueType,StencilPoint,GridDimension>(stencil[i], grid));
    }

    matrix.num_entries = matrix.values.values.size() - amgx::thrust::count(matrix.values.values.begin(), matrix.values.values.end(), ValueType(0));
}

// TODO add an entry point and make this the default path
template <typename MatrixType,
          typename StencilPoint,
          typename GridDimension>
void generate_matrix_from_stencil(      MatrixType& matrix,
                                  const cusp::array1d<StencilPoint,cusp::host_memory>& stencil,
                                  const GridDimension& grid)
{
    CUSP_PROFILE_SCOPED();

    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    cusp::dia_matrix<IndexType,ValueType,MemorySpace> dia;
    generate_matrix_from_stencil(dia, stencil, grid);

    cusp::convert(dia, matrix);
}
                            
} // end namespace gallery
} // end namespace cusp

