// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "test_utils.h"
#include "matrix_coloring/min_max.h"
#include "matrix_coloring/multi_hash.h"
#include "matrix_coloring/parallel_greedy.h"
#include <thrust/iterator/counting_iterator.h>
#include "matrix_analysis.h"

namespace amgx
{

__global__ void start_coloring_mark() {}
__global__ void end_coloring_mark() {}

template<int LEVEL>
struct neighbor_color_validity_checker
{
    __host__ __device__ static int check(const int *A_row_offsets, const int *A_col_indices, const int *color, const int row_start, const int row_end, const int my_color, const int self)
    {
        for (int row_it = row_start; row_it < row_end; row_it++)
        {
            int c   = A_col_indices[row_it];
            int col = color[c];

            if (my_color == col && c != self)
            {
                return 0;
            }

            return neighbor_color_validity_checker < LEVEL - 1 >::check(A_row_offsets, A_col_indices, color, A_row_offsets[c], A_row_offsets[c + 1], my_color, self);
        }

        return 1; //empty row: ok
    }

};

template<>
struct neighbor_color_validity_checker<0>
{
    __host__ __device__ static int check(const int *A_row_offsets, const int *A_col_indices, const int *color, const int row_start, const int row_end, const int my_color, const int self)
    {
        //end compile time recursion
        return 1;
    }
};

template<typename IndexType>
struct CheckColoring
{
    const IndexType *_A_offsets;
    const IndexType *_A_cols;
    const IndexType *_colors;
    const int _coloring_level;

    CheckColoring(const IndexType *A_offsets, const IndexType *A_cols, const IndexType *colors, const int coloring_level) : _A_offsets(A_offsets), _A_cols(A_cols), _colors(colors), _coloring_level(coloring_level) { }

    __host__ __device__ IndexType operator()(IndexType i)
    {
        IndexType col = _colors[i];
        // do not count uncolored nodes

        if (col == 0)
        {
            return 0;
        }

        IndexType row_start = _A_offsets[i];
        IndexType row_end = _A_offsets[i + 1];

        if (_coloring_level == 4)
        {
            return neighbor_color_validity_checker<4>::check(_A_offsets, _A_cols, _colors, row_start, row_end, col, i) == 0;
        }
        else if (_coloring_level == 5)
        {
            return neighbor_color_validity_checker<5>::check(_A_offsets, _A_cols, _colors, row_start, row_end, col, i) == 0;
        }
        else if (_coloring_level == 6)
        {
            return neighbor_color_validity_checker<6>::check(_A_offsets, _A_cols, _colors, row_start, row_end, col, i) == 0;
        }
        else if (_coloring_level == 7)
        {
            return neighbor_color_validity_checker<7>::check(_A_offsets, _A_cols, _colors, row_start, row_end, col, i) == 0;
        }
        else
        {
            for (IndexType r = row_start; r < row_end; r++)
            {
                IndexType j  = _A_cols[r];

                // skip diagonal
                if (j == i) { continue; }

                IndexType jcol = _colors[j];

                if (jcol == 0) { continue; }

                // if 2 colors are adjacent, return 1
                if (jcol == col)
                {
                    //printf("invalid coloring, row=%d, col=%d, row_color=%d, col_color=%d\n",i,j,col,jcol);
                    return 1;
                }

                if (_coloring_level >= 2)
                {
                    IndexType row_start = _A_offsets[j];
                    IndexType row_end = _A_offsets[j + 1];

                    for (IndexType r = row_start; r < row_end; r++)
                    {
                        IndexType k  = _A_cols[r];

                        // skip diagonal
                        if (k == i) { continue; }

                        IndexType jcol = _colors[k];

                        if (jcol == 0) { continue; }

                        // if 2 colors are adjacent, return 1
                        if (jcol == col)
                        {
                            //printf("invalid 2 coloring, row=%d, col=%d, row_color=%d, col_color=%d\n",i,j2,col,jcol);
                            return 1;
                        }

                        if (_coloring_level >= 3)
                        {
                            IndexType row_start = _A_offsets[k];
                            IndexType row_end = _A_offsets[k + 1];

                            for (IndexType r = row_start; r < row_end; r++)
                            {
                                IndexType l  = _A_cols[r];

                                // skip diagonal
                                if (l == i) { continue; }

                                IndexType jcol = _colors[l];

                                if (jcol == 0) { continue; }

                                // if 2 colors are adjacent, return 1
                                if (jcol == col)
                                {
                                    //printf("invalid 2 coloring, row=%d, col=%d, row_color=%d, col_color=%d\n",i,j2,col,jcol);
                                    return 1;
                                }
                            }
                        }
                    }
                }
            }
        } // loop over first ring neighbours

        // no conflict => return 0
        return 0;
    }
};

static std::map<std::string, double> total_runtime; //for global aggregate scoring across tests
static std::map<std::string, double> coloring_score;
static std::map<std::string, double> coloring_runs;

static FILE *f_csv_file = 0; //once per test rewrite
struct is_zero
{
    __host__ __device__
    bool operator()(int x)
    {
        return x == 0;
    }
};

DECLARE_UNITTEST_BEGIN(MatrixColoringTest_Base);

std::string base_keywords()
{
    return "coloring";
}

// dense histogram using binary search
template <typename Vector1,
          typename Vector2>
void color_histogram(const Vector1 &row_colors, Vector2 &histogram)
{
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector2::value_type IndexType; // histogram index type
    // copy input data (could be skipped if input is allowed to be modified)
    device_vector_alloc<ValueType> data(row_colors);
    // sort data to bring equal elements together
    thrust_wrapper::sort<AMGX_device>(data.begin(), data.end());
    // number of histogram bins is equal to the maximum value plus one
    IndexType num_bins = data.back() + 1;
    // resize histogram storage
    histogram.resize(num_bins);
    // find the end of each bin of values
    amgx::thrust::counting_iterator<IndexType> search_begin(0);
    amgx::thrust::upper_bound(data.begin(), data.end(),
                        search_begin, search_begin + num_bins,
                        histogram.begin());
    // compute the histogram by taking differences of the cumulative histogram
    amgx::thrust::adjacent_difference(histogram.begin(), histogram.end(),
                                histogram.begin());
}




template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
color_matrix_file(const std::string &filename, int max_coloring_level = 3)
{
    AMG_Config cfg;
    size_t pool_size = cfg.getParameter<size_t>("device_mem_pool_size", "default");
    size_t max_alloc_size = cfg.getParameter<size_t>("device_mem_pool_max_alloc_size", "default");

    if ( memory::hasPinnedMemoryPool() )
    {
        memory::setPinnedMemoryPool( new memory::PinnedMemoryPool() );
    }

    if ( memory::hasDeviceMemoryPool() )
    {
        memory::setDeviceMemoryPool( new memory::DeviceMemoryPool(pool_size, max_alloc_size, 0) );
    }

    cudaDeviceSynchronize(); // why ????
    cudaCheckError();
    bool verbose_output = 0;
    const std::string cfg_scope = "default";
    std::stringstream config_string_base;
    config_string_base <<  "max_uncolored_percentage=0, coloring_level=1";
    std::vector<std::string> coloring_schemes;
#define ALL 1
#if ALL
    coloring_schemes.push_back("GREEDY_RECOLOR");
    coloring_schemes.push_back("MIN_MAX");
    coloring_schemes.push_back("MIN_MAX_2RING");
    coloring_schemes.push_back("PARALLEL_GREEDY");
    coloring_schemes.push_back("MULTI_HASH,max_uncolored_percentage=0.0");
    coloring_schemes.push_back("GREEDY_RECOLOR,coloring_custom_arg=FIRSTSTEP");
    coloring_schemes.push_back("GREEDY_MIN_MAX_2RING");
    //coloring_schemes.push_back("SERIAL_GREEDY_BFS, coloring_custom_arg=IDO");
    coloring_schemes.push_back("SERIAL_GREEDY_BFS");
#else
    //max_coloring_level = 2;
#endif

    if (max_coloring_level >= 2)
    {
#if 1
        coloring_schemes.push_back(""); //placeholder, newline in output if enabled
        coloring_schemes.push_back("MIN_MAX, coloring_level=2");
        coloring_schemes.push_back("PARALLEL_GREEDY, coloring_level=2");
        coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=2");
        //coloring_schemes.push_back("GREEDY_RECOLOR, coloring_level=2");
        //coloring_schemes.push_back("GREEDY_RECOLOR, coloring_level=2,coloring_custom_arg=FIRSTSTEP");
        //coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=2, coloring_custom_arg=FULL64");
        //coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=2, coloring_custom_arg=FULL256");
        //coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=2, coloring_custom_arg=CN1");
        //coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=2, coloring_custom_arg=CN2");
        coloring_schemes.push_back("SERIAL_GREEDY_BFS, coloring_level=2");
        /*coloring_schemes.push_back("SERIAL_GREEDY_BFS, coloring_custom_arg=IDO, coloring_level=2");
        */
#else
        coloring_schemes.push_back("PARALLEL_GREEDY, coloring_level=2");
        coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=2");
#endif
    }

    if (max_coloring_level >= 3)
    {
        coloring_schemes.push_back(""); //placeholder, newline in output if enabled
        coloring_schemes.push_back("PARALLEL_GREEDY, coloring_level=3");
        coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=3");
        coloring_schemes.push_back("SERIAL_GREEDY_BFS, coloring_level=3");
    }

    if (max_coloring_level >= 4)
    {
        coloring_schemes.push_back(""); //placeholder, newline in output if enabled
        coloring_schemes.push_back("PARALLEL_GREEDY, coloring_level=4");
        coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=4");
        coloring_schemes.push_back("SERIAL_GREEDY_BFS, coloring_level=4");
    }

    if (max_coloring_level >= 5)
    {
        coloring_schemes.push_back(""); //placeholder, newline in output if enabled
        coloring_schemes.push_back("PARALLEL_GREEDY, coloring_level=5");
        coloring_schemes.push_back("GREEDY_MIN_MAX_2RING, coloring_level=5");
        coloring_schemes.push_back("SERIAL_GREEDY_BFS, coloring_level=5");
    }

    if (verbose_output) //enable for writing a csv with all the info
    {
        if (!f_csv_file)
        {
            f_csv_file = fopen("matrix_coloring_test_out.txt", "w");
        }
        else
        {
            f_csv_file = fopen("matrix_coloring_test_out.txt", "a");
        }
    }

    if (verbose_output) { std::cout << std::endl; }

    std::vector<std::string> text_outputs(coloring_schemes.size());
    int min_time_i = -1;
    float min_time = 0;
    int min_colors_i = -1;
    int min_colors = 0;
    int min_colors_iX = -1;
    int min_colorsX = 0;
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_h> Vector_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<Config_d> Vector_d;
    Matrix_h A_h;
    Vector_h x_h, b_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    std::string fail_msg = "Cannot open " + filename;
    this->PrintOnFail(fail_msg.c_str());
    //UNITTEST_ASSERT_TRUE(this->read_system( filename.c_str(), A_h, b_h, x_h ));
    UNITTEST_ASSERT_TRUE(MatrixIO<Config_h>::readSystem( filename.c_str(), A_h, b_h, x_h ) == AMGX_OK);;
    A_h.set_initialized(1);
    MatrixAnalysis<TConfig_h> ana(&A_h, &b_h);
    bool str_sym, sym;
    bool verbose = false;
    ana.checkSymmetry(str_sym, sym, verbose);
    Matrix_d A_d( A_h );

    for (int i = 0; i < coloring_schemes.size(); i++)
    {
        if (coloring_schemes[i] == "")
        {
            if (verbose_output) { text_outputs[i] = "\n"; }

            if (verbose_output) { std::cout << "\n"; }

            min_colorsX = 0;
            min_colors_iX = -1;
            continue;
        }

        std::stringstream config_string;
        std::stringstream text_output;
        config_string << config_string_base.str() << ", matrix_coloring_scheme=" << coloring_schemes[i] << std::endl;
        AMG_Config  cfg;
        UNITTEST_ASSERT_TRUE ( cfg.parseParameterString(config_string.str().c_str()) == AMGX_OK);
        UNITTEST_ASSERT_TRUE_DESC("Matrix does not have structural symmetry, unit test invalid", str_sym);;
        A_d.set_initialized(0);
        A_d.addProps(CSR);
//      cudaDeviceSynchronize();
cudaCheckError();
        cudaEvent_t color_start, color_stop;
        cudaEventCreate(&color_start);
        cudaCheckError();
        cudaEventCreate(&color_stop);
        cudaCheckError();
        //A_d.colorMatrix(cfg,cfg_scope);
        //A_d.set_initialized(0);
        float elapsed_time;
        cudaEventRecord( color_start);
        cudaCheckError();
        A_d.colorMatrix(cfg, cfg_scope);
        //amgx::testing_tools::hash_path_determinism_checker::singleton()->checkpoint("colors",  (void*)A_d.getMatrixColoring().getRowColors().raw(), A_d.getMatrixColoring().getRowColors().size()*4);
        //amgx::testing_tools::hash_path_determinism_checker::singleton()->checkpoint("cols",  (void*)A_d.col_indices.raw(), A_d.col_indices.size()*4);
        //amgx::testing_tools::hash_path_determinism_checker::singleton()->checkpoint("rows",  (void*)A_d.row_offsets.raw(), A_d.row_offsets.size()*4);
        cudaEventRecord( color_stop);
        cudaCheckError();
        cudaEventSynchronize( color_stop);
        cudaCheckError();
        cudaEventElapsedTime( &elapsed_time, color_start, color_stop);
        elapsed_time *= 1e-3f;
        A_d.set_initialized(1);
        // check coloring
        const IndexType *A_row_offsets_ptr    = A_d.row_offsets.raw();
        const IndexType *A_column_indices_ptr = A_d.col_indices.raw();
        const IndexType *row_colors_ptr = A_d.getMatrixColoring().getRowColors().raw();
        IndexType num_bad, num_bad_plus;
        {
            //order +1
            CheckColoring<IndexType> checker(A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, 1 + A_d.getMatrixColoring().getColoringLevel());
            num_bad_plus =
                thrust_wrapper::transform_reduce<AMGX_device>(amgx::thrust::counting_iterator<IndexType>(0), amgx::thrust::counting_iterator<IndexType>(A_d.get_num_rows()),
                                         checker, (IndexType)0, amgx::thrust::plus<IndexType>());
        }
        {
            CheckColoring<IndexType> checker(A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, A_d.getMatrixColoring().getColoringLevel());
            num_bad =
                thrust_wrapper::transform_reduce<AMGX_device>(amgx::thrust::counting_iterator<IndexType>(0), amgx::thrust::counting_iterator<IndexType>(A_d.get_num_rows()),
                                         checker, (IndexType)0, amgx::thrust::plus<IndexType>());
        }
        int num_uncolored = (int) amgx::thrust::count_if( A_d.getMatrixColoring().getRowColors().begin(), A_d.getMatrixColoring().getRowColors().begin() + A_d.get_num_rows(), is_zero() );
        int mincol        = (int) amgx::thrust::reduce( A_d.getMatrixColoring().getRowColors().begin(), A_d.getMatrixColoring().getRowColors().begin() + A_d.get_num_rows(), 0, amgx::thrust::minimum<int>() );
        int maxcol        = (int) amgx::thrust::reduce( A_d.getMatrixColoring().getRowColors().begin(), A_d.getMatrixColoring().getRowColors().begin() + A_d.get_num_rows(), 0, amgx::thrust::maximum<int>() );
        std::stringstream error_str;
        error_str << "Coloring scheme " << coloring_schemes[i] << " with " << num_bad << " invalid rows, " << num_uncolored << "(" << mincol << "," << maxcol << ")" << " uncolored rows and " << A_d.getMatrixColoring().getNumColors()  << " used colors and bonus = " << num_bad_plus;

        if (verbose_output)
        {
            text_output << std::setw(28) << coloring_schemes[i] << " : num_colors=" << std::setw(6) << A_d.getMatrixColoring().getNumColors() << " , runtime= " << std::setw(8) << std::fixed << std::setprecision(5) << elapsed_time << " mM(" << mincol << "," << maxcol << ")"; // << std::endl;;
        }

        if (num_bad == 0 && num_uncolored == 0)
        {
            if (elapsed_time < min_time || min_time_i == -1)
            {
                min_time_i = i;
                min_time = elapsed_time;
            }

            if (A_d.getMatrixColoring().getNumColors() < min_colors || min_colors_i == -1)
            {
                min_colors_i = i;
                min_colors = A_d.getMatrixColoring().getNumColors();
            }

            if (A_d.getMatrixColoring().getNumColors() < min_colorsX || min_colors_iX == -1)
            {
                min_colors_iX = i;
                min_colorsX = A_d.getMatrixColoring().getNumColors();
            }
        }

        int degree;
        {
            device_vector_alloc<int> hist;
            amgx::thrust::host_vector<int> h_hist;
            device_vector_alloc<int> columns(A_d.get_num_rows());
            amgx::thrust::adjacent_difference(A_d.row_offsets.begin(), A_d.row_offsets.end(), columns.begin());
            degree = amgx::thrust::reduce(columns.begin(), columns.end(), -1, amgx::thrust::maximum<int>());
            color_histogram(A_d.getMatrixColoring().getRowColors(), hist);
            h_hist = hist;
            float total = 0, m = h_hist[1], M = h_hist[1]; //0 is 0

            for (int j = 1; j < h_hist.size(); j++)
            {
                if (h_hist[j] > M) { M = h_hist[j]; }

                if (h_hist[j] < m) { m = h_hist[j]; }

                total += h_hist[j];
            }

            float avg = total / (h_hist.size() - 1);
            float rms = 0;

            for (int j = 0; j < h_hist.size(); j++)
            {
                rms = (h_hist[j] - avg) * (h_hist[j] - avg);
            }

            rms = sqrt(rms / (h_hist.size() - 1));

            if (verbose_output) { text_output << " [" << " min=" << m << ", max=" << M << ", avg=" << avg << ", rms=" << rms << ", deg=" << degree << ", uncol=" << num_uncolored << " bonus=" << num_bad_plus << "]"; }

            if (f_csv_file) { fprintf(f_csv_file, "%s;%d;%d;%d;%.3f;%.3f;%.3f;%.3f;%.3f\n", coloring_schemes[i].c_str(), A_d.getMatrixColoring().getNumColors(), num_bad, num_uncolored, elapsed_time * 1e3, m, M, avg, rms); }
        }
        total_runtime[coloring_schemes[i]]  += elapsed_time;
        coloring_score[coloring_schemes[i]] += double(A_d.getMatrixColoring().getNumColors() - 1) / pow(static_cast<double>(degree), static_cast<double>(A_d.getMatrixColoring().getColoringLevel()));
        coloring_runs[coloring_schemes[i]]  += 1.0;

        if (verbose_output) { text_output << std::endl; }

        if (verbose_output) { printf("%s", text_output.str().c_str()); }

        text_outputs[i] = text_output.str();
        UNITTEST_ASSERT_EQUAL_DESC(error_str.str().c_str(), num_bad, 0);
    }

    if (verbose_output && 0)
    {
        for (int i = 0; i < text_outputs.size(); i++)
        {
            if (i == min_colors_i) { std::cout << "*"; }

            if (i == min_time_i) { std::cout << "#"; }

            std::cout << text_outputs[i];
        }
    }

    if (f_csv_file)
    {
        fprintf(f_csv_file, "\n");
        fclose(f_csv_file);
    }

    if (verbose_output)
    {
        printf("\n\n\n");

        for (int i = 0; i < text_outputs.size(); i++)
        {
            if (coloring_schemes[i] == "")
            {
                std::cout << "\n";
                continue;
            }

            std::string name = coloring_schemes[i];
            std::cout << std::setw(80) << coloring_schemes[i];
            std::cout << " runtime=" << std::setw(8) << std::fixed << std::setprecision(5) << total_runtime[name];
            std::cout << " inefficiency=" << std::setw(6) << std::fixed << std::setprecision(3) << coloring_score[name] / coloring_runs[name]; // << std::endl;;
            std::cout << "\n";
        }

        printf("\n\n\n");
    }

    cudaDeviceSynchronize();
    cudaCheckError();
}

DECLARE_UNITTEST_END(MatrixColoringTest_Base);


DECLARE_UNITTEST_BEGIN_EXTD(MatrixColoringTestatmosmodj, MatrixColoringTest_Base<T_Config>);

void run()
{
    //std::cout << std::endl;
    MatrixColoringTest_Base<T_Config>::template color_matrix_file<T_Config::vecPrec, T_Config::matPrec>( UnitTest::get_configuration().data_folder + "Public/Florida/atmosmodj.mtx.bin");
}

DECLARE_UNITTEST_END(MatrixColoringTestatmosmodj)
MatrixColoringTestatmosmodj<TemplateMode<AMGX_mode_dDDI>::Type> MatrixColoringTestatmosmodj_dDDI;


//--------------------------------------------------------------------------------------------------
DECLARE_UNITTEST_BEGIN_EXTD(MatrixColoringTestatmosmodl, MatrixColoringTest_Base<T_Config>);

void run()
{
    //std::cout << std::endl;
    MatrixColoringTest_Base<T_Config>::template color_matrix_file<T_Config::vecPrec, T_Config::matPrec>( UnitTest::get_configuration().data_folder + "Public/Florida/atmosmodl.mtx.bin");
}

DECLARE_UNITTEST_END(MatrixColoringTestatmosmodl)
MatrixColoringTestatmosmodl<TemplateMode<AMGX_mode_dDDI>::Type> MatrixColoringTestatmosmodl_dDDI;


//--------------------------------------------------------------------------------------------------
DECLARE_UNITTEST_BEGIN_EXTD(MatrixColoringTestpoisson, MatrixColoringTest_Base<T_Config>);

void run()
{
    //std::cout << std::endl;
    MatrixColoringTest_Base<T_Config>::template color_matrix_file<T_Config::vecPrec, T_Config::matPrec>( UnitTest::get_configuration().data_folder + "Internal/poisson/poisson27x50x50x50.mtx.bin");
}

DECLARE_UNITTEST_END(MatrixColoringTestpisson)
MatrixColoringTestpoisson<TemplateMode<AMGX_mode_dDDI>::Type> MatrixColoringTestpoisson_dDDI;

} //namespace amgx
