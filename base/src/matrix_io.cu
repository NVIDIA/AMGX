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

#include <matrix_io.h>
#include "misc.h"
#include "util.h"
#include <string>
#include <iostream>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <types.h>
#include <iomanip>
#include <map>
#include <basic_types.h>
#include <matrix.h>
#include <amgx_timer.h>

#include "amgx_types/util.h"
#include "amgx_types/io.h"

using namespace std;

namespace amgx
{

template<class T_Config>
MatrixIO<T_Config>::readerMap &MatrixIO<T_Config>::getReaderMap()
{
    static readerMap readers_map;
    return readers_map;
}

template<class T_Config>
void MatrixIO<T_Config>::registerReader(string key, readerFunc func)
{
    readerMap &readers_map = getReaderMap();
    typename readerMap::const_iterator iter = readers_map.find(key);

    if (iter != readers_map.end())
    {
        std::string err = "Reader '" + key + "' is already registered";
        FatalError(err, AMGX_ERR_CORE);
    }

    readers_map[key] = func;
}

template<class T_Config>
void MatrixIO<T_Config>::unregisterReaders()
{
    readerMap &readers_map = getReaderMap();
    readers_map.clear();
}

template<class T_Config>
MatrixIO<T_Config>::writerMap &MatrixIO<T_Config>::getWriterMap()
{
    static writerMap writer_map;
    return writer_map;
}

template<class T_Config>
void MatrixIO<T_Config>::registerWriter(string key, writerFunc func)
{
    writerMap &writer_map = getWriterMap();
    typename writerMap::const_iterator iter = writer_map.find(key);

    if (iter != writer_map.end())
    {
        std::string err = "Reader '" + key + "' is already registered";
        FatalError(err, AMGX_ERR_CORE);
    }

    writer_map[key] = func;
}

template<class T_Config>
void MatrixIO<T_Config>::unregisterWriters()
{
    writerMap &writer_map = getWriterMap();
    writer_map.clear();
}

template<class T_Config>
bool MatrixIO<T_Config>::writeSystemMatrixMarket(const char *fname, const Matrix<T_Config> *pA, const VVector *pb, const VVector *px)
{
    typedef typename T_Config::MatPrec ValueTypeA;
    typedef typename T_Config::VecPrec ValueTypeB;

    if (!fname)
    {
        FatalError( "Bad filename", AMGX_ERR_BAD_PARAMETERS);
    }

    if (!pA)
    {
        FatalError( "MatrixMarket should contain matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    std::ofstream fout;
    std::string err = "Writing system to file " + string(fname) + "\n";
    amgx_output(err.c_str(), err.length());
    fout.open(fname);

    if (!fout)
    {
        FatalError( "Cannot open file for writing!", AMGX_ERR_BAD_PARAMETERS);
    }

    const Matrix<T_Config> &A = *pA;
    bool is_mtx = true;
    bool is_rhs = pb != NULL && pb->size() > 0;
    bool is_soln = px != NULL && px->size() > 0;
    fout << "%%MatrixMarket";

    if (is_mtx)
    {
        fout << " matrix coordinate ";

        if (types::util<typename Matrix<T_Config>::value_type>::is_real)
        {
            fout << "real ";
        }
        else
        {
            fout << "complex ";
        }

        fout << "general";
    }
    else
    {
        if (types::util<typename Matrix<T_Config>::value_type>::is_real)
        {
            fout << "real ";
        }
        else
        {
            fout << "complex ";
        }
    }

    fout << std::endl;
    fout << "%%NVAMG " << A.get_block_dimx() << " " << A.get_block_dimy() << " ";

    if (A.hasProps(DIAG) && is_mtx) { fout << "diagonal "; }

    if (is_mtx) { fout << "sorted "; }

    if (is_rhs) { fout << "rhs "; }

    if (is_soln) { fout << "solution"; }

    fout << std::endl;
    fout << A.get_num_rows()*A.get_block_dimx()  << " " << A.get_num_cols()*A.get_block_dimy() << " " << A.get_num_nz()*A.get_block_size() <<  std::endl;
    // rules are simple: If there is csr property - write csr and coo (if exists). Else write coo.
    fout << std::setprecision(std::numeric_limits<ValueTypeA>::digits10 + 1) << std::scientific;

    if (is_mtx)
    {
        if (A.hasProps(COO))
        {
            for (int i = 0; i < A.get_num_nz(); i++)
            {
                for (int kx = 0; kx < A.get_block_dimx(); kx++)
                    for (int ky = 0; ky < A.get_block_dimy(); ky++)
                    {
                        fout << A.row_indices[i]*A.get_block_dimx() + kx + 1 << " " << A.col_indices[i]*A.get_block_dimy() + ky + 1 << " " << A.values[i * A.get_block_size() + kx * A.get_block_dimy() + ky] << std::endl;
                    }
            }
        }
        else if (A.hasProps(CSR))
        {
            for (int i = 0; i < A.get_num_rows(); i++)
            {
                for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
                {
                    int c = A.col_indices[j];

                    //      typename Matrix::value_type v=A.values[j];
                    for (int kx = 0; kx < A.get_block_dimx(); kx++)
                        for (int ky = 0; ky < A.get_block_dimy(); ky++)
                        {
                            fout << i *A.get_block_dimx() + kx + 1 << " " << c *A.get_block_dimy() + ky + 1 << " " << A.values[j * A.get_block_size() + kx * A.get_block_dimy() + ky] << std::endl;
                        }
                }
            }
        }

        if (A.hasProps(DIAG))
        {
            for (int i = 0; i < A.get_num_rows(); i++)
            {
                for (int k = 0; k < A.get_block_size(); k++)
                {
                    fout << A.values[A.diag[i]*A.get_block_size() + k] << " ";
                }

                fout << std::endl;
            }
        }
    } // End of writing matrix

    fout << std::setprecision(std::numeric_limits<ValueTypeB>::digits10 + 1) << std::scientific;

    //write rhs
    if (is_rhs)
    {
        const VVector &b = *pb;
        fout << b.size() << std::endl;

        for (int i = 0; i < b.size(); i++)
        {
            fout << b[i] << std::endl;
        }
    }

    // write initial guess if we have it
    if (is_soln)
    {
        const VVector &x = *px;
        fout << x.size() << std::endl;

        for (int i = 0; i < x.size(); i++)
        {
            fout << x[i] << std::endl;
        }
    }

    fout.close();
    err = "Done writing system to file!\n";
    amgx_output(err.c_str(), err.length());
    return true;
}


template<class T_Config>
bool MatrixIO<T_Config>::writeSystemBinary(const char *fname, const Matrix<T_Config> *pA, const VVector *pb, const VVector *px)
{
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    typedef Vector<TConfig_h> VVector_h;
    typedef typename Matrix<TConfig_h>::MVector MVector_h;
    typedef typename Matrix<T_Config>::index_type IndexType;
    typedef typename Matrix<T_Config>::value_type ValueTypeA;
    typedef typename Vector<T_Config>::value_type ValueTypeB; // change back to matrix type later
    typedef typename types::util<ValueTypeA>::uptype UpValueType;

    if (!fname)
    {
        FatalError( "Bad filename", AMGX_ERR_BAD_PARAMETERS);
    }

    if (!pA)
    {
        FatalError( "MatrixMarket should contain matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    FILE *fout;
    const char header [] = "%%NVAMGBinary\n";
    std::string err = "Writing system to file " + string(fname) + "\n";
    amgx_output(err.c_str(), err.length());
    fout = fopen(fname, "wb");

    if (!fout)
    {
        FatalError( "Cannot open output file!11", AMGX_ERR_BAD_PARAMETERS);
    }

    bool is_mtx = true;
    bool is_rhs = pb != NULL && pb->size() > 0;
    bool is_soln = px != NULL && px->size() > 0;
    const Matrix<T_Config> &A = *pA;
    uint32_t matrix_format = 42;

    if (A.hasProps(CSR))
    {
        matrix_format = 0;
    }
    else if (A.hasProps(COO))
    {
        matrix_format = 1;
    }
    else
    {
        FatalError("Unsupported matrix format", AMGX_ERR_BAD_PARAMETERS);
    }

    if (types::util<ValueTypeA>::is_complex)
    {
        matrix_format += COMPLEX;
    }

    const int system_header_size = 9;
    uint32_t system_flags [] = { (uint32_t)(is_mtx), (uint32_t)(is_rhs), (uint32_t)(is_soln), matrix_format, (uint32_t)(A.hasProps(DIAG)),
                                 (uint32_t)(A.get_block_dimx()), (uint32_t)(A.get_block_dimy()), (uint32_t)(A.get_num_rows()), (uint32_t)(A.get_num_nz())
                               };
    fwrite(header, sizeof(char), strlen(header), fout);
    fwrite(system_flags, sizeof(uint32_t), system_header_size, fout);
    std::vector< UpValueType > tempv(A.values.size(), types::util< UpValueType >::get_zero());

    if (is_mtx)
    {
        if (A.hasProps(CSR))
        {
            IVector_h t_int = A.row_offsets;
            fwrite(t_int.raw(), sizeof(int), A.get_num_rows() + 1, fout); //assuming int as an index
            t_int = A.col_indices;
            fwrite(t_int.raw(), sizeof(int), A.get_num_nz(), fout); //assuming int as an index

            for (int k = 0; k < A.values.size(); k++)
            {
                types::util<ValueTypeA>::to_uptype(A.values[k], tempv[k]);
            }

            fwrite(&tempv[0], sizeof(UpValueType), A.get_block_dimx() * A.get_block_dimy() * (A.get_num_nz() + (A.hasProps(DIAG) ? A.get_num_rows() : 0) ), fout); // including diag in the end if exists.
        }
        else
        {
            FatalError("Unsupported matrix format for now", AMGX_ERR_IO);
        }
    } // End of writing matrix

    VVector_h tvec;

    //write rhs
    if (is_rhs)
    {
        if (pb->size() != A.get_num_rows()*A.get_block_dimy())
        {
            FatalError("rhs vector and matrix dimension does not match", AMGX_ERR_BAD_PARAMETERS);
        }

        tempv.resize(A.get_num_rows()*A.get_block_dimy());

        for (int k = 0; k < pb->size(); k++)
        {
            types::util<ValueTypeB>::to_uptype((*pb)[k], tempv[k]);
        }

        fwrite(&tempv[0], sizeof(UpValueType), pb->size(), fout);
    }

    // write initial guess if we have it
    if (is_soln)
    {
        if (px->size() != A.get_num_rows()*A.get_block_dimx())
        {
            FatalError("solution vector and matrix dimension does not match", AMGX_ERR_BAD_PARAMETERS);
        }

        tempv.resize(A.get_num_rows()*A.get_block_dimy());

        for (int k = 0; k < px->size(); k++)
        {
            types::util<ValueTypeB>::to_uptype((*px)[k], tempv[k]);
        }

        fwrite(&tempv[0], sizeof(UpValueType), px->size(), fout);
    }

    fclose(fout);
    err = "Done writing system to file!\n";
    amgx_output(err.c_str(), err.length());
    return true;
}


template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::readSystem(const char *fname
        , Matrix<T_Config> &A
        , VVector &b
        , VVector &x
        , const AMG_Config &cfg
        , unsigned int props
        , const IVector_h &rank_rows // row indices for given rank
                                         )
{
    AMGX_CPU_PROFILER( "MatrixIO::read_sytem " );

    try
    {
        readerMap &readers_map = getReaderMap();
        //open file
        std::string err;

        if (io_config::hasProps(io_config::SIZE, props))
        {
            err = "Reading matrix dimensions in file: " + string(fname) + "\n";
        }
        else if (io_config::hasProps(io_config::PRINT, props))
        {
            err = "Reading matrix in file: " + string(fname) + "\n";
        }

        amgx_output(err.c_str(), err.length());
        std::ifstream fin(fname);

        if (!fin)
        {
            err = "Error opening file '" + string(fname) + "'\n";
            FatalError(err.c_str(), AMGX_ERR_IO);
        }

        // Extract the file format from the file
        std::string fformat;
        fin >> fformat;

        if (fformat.substr(0, 2) != "%%")
        {
            err = "Invalid header line in file " + string(fname) + " First line should begin with: %%MatrixFormat\n";
            FatalError(err.c_str(), AMGX_ERR_IO);
        }
        else
        {
            fformat = fformat.substr(2, fformat.size());
        }

        typename readerMap::const_iterator iter = readers_map.find(fformat);

        if (iter == readers_map.end())
        {
            err = "Could not find a reader for matrix of type '" + fformat + "'\n";
            FatalError(err.c_str(), AMGX_ERR_IO);
        }

        //call reader
        A.set_initialized(0);
        (iter->second)(fin
                       , fname
                       , A
                       , b
                       , x
                       , cfg
                       , props
                       , rank_rows
                      );
        A.computeDiagonal();
        A.set_initialized(1);
        fin.close();
    }
    catch (amgx_exception e)
    {
        std::string err = "Error while reading matrix: ";
        amgx_output(err.c_str(), err.length());
        amgx_output(e.what(), strlen(e.what()));
        return AMGX_ERR_IO;
    }

    return AMGX_OK;
}

template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::writeSystem (const char *filename, const Matrix<T_Config> *A, const VVector *b, const VVector *x)
{
    std::string format;

    try
    {
        AMG_Config *cfg = NULL;

        if (A)
        {
            cfg = A->getResources()->getResourcesConfig();
        }

        if (b)
        {
            cfg = b->getResources()->getResourcesConfig();
        }

        if (!cfg)
        {
            FatalError("Couldn't get resources from matrix or vector", AMGX_ERR_BAD_PARAMETERS);
        }

        format = cfg->AMG_Config::getParameter<std::string>("matrix_writer", "default");
    }
    catch (amgx_exception e)
    {
        std::string err = "Error while writing matrix: ";
        amgx_output(err.c_str(), err.length());
        amgx_output(e.what(), strlen(e.what()));
        return AMGX_ERR_IO;
    }

    // call to actual writeMatrixWithFormat:
    return writeSystemWithFormat (filename, format.c_str(), A, b, x);
}

template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::writeSystemWithFormat (const char *filename, const char *format, const Matrix<T_Config> *A, const VVector *b, const VVector *x)
{
    AMGX_CPU_PROFILER( "MatrixIO::sytem " );

    try
    {
        writerMap &writers_map = getWriterMap();
        typename writerMap::const_iterator iter = writers_map.find(format);

        if (iter == writers_map.end())
        {
            std::string err;
            err = "Could not find a writer: '" + std::string(format) + "'\n";
            FatalError(err.c_str(), AMGX_ERR_IO);
        }

        if ( !(iter->second)( filename, A, b, x ) )
        {
            return AMGX_ERR_IO;
        }
    }
    catch (amgx_exception e)
    {
        std::string err = "Error while writing matrix: ";
        amgx_output(err.c_str(), err.length());
        amgx_output(e.what(), strlen(e.what()));
        return AMGX_ERR_IO;
    }

    return AMGX_OK;
}



template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::readSystem(const char *fname
        , Matrix<T_Config> &A
        , const AMG_Config &cfg
        , unsigned int props
        , const IVector_h &rank_rows // row indices for given rank
                                         )
{
    VVector b = VVector(0);
    VVector x = VVector(0);
    return readSystem(fname, A, b, x, cfg, props, rank_rows);
}

template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::readSystem(const char *fname
        , Matrix<T_Config> &A
        , VVector &b
        , const AMG_Config &cfg
        , unsigned int props
        , const IVector_h &rank_rows // row indices for given rank
                                         )
{
    VVector v = VVector(0);

    if (io_config::hasProps(io_config::RHS, props))
    {
        return readSystem(fname, A, b, v, cfg, props, rank_rows);
    }
    else
    {
        return readSystem(fname, A, v, b, cfg, props, rank_rows);
    }
}

template<class T_Config>
string MatrixIO<T_Config>::readSystemFormat(const char *fname)
{
    readerMap &readers_map = getReaderMap();
    //open file
    std::string out = "Reading matrix format in file: " + string(fname) + "\n";
    amgx_output(out.c_str(), out.length());
    std::ifstream fin(fname);

    if (!fin)
    {
        out = "Error opening file: " + string(fname) + "\n";
        FatalError(out.c_str(), AMGX_ERR_IO);
    }

    // Extract the file format from the file
    std::string fformat;
    fin >> fformat;

    if (fformat.substr(0, 2) != "%%")
    {
        out = "Invalid header line in file " + std::string( fname ) + " First line should begin with: %%MatrixFormat\n";
        FatalError(out.c_str(), AMGX_ERR_IO);
    }
    else
    {
        fformat = fformat.substr(2, fformat.size());
    }

    return fformat;
}

/*template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::readGeometry( AuxData* obj, const char* fname)
{
  std::string err;
  err = "Reading matrix in file: " + string(fname) + "\n";
  amgx_output(err.c_str(), err.length());

  std::ifstream fin(fname);
  if(!fin) {
    err = "Error opening file '" + string(fname) + "'\n";
      FatalError(err.c_str(), AMGX_ERR_IO);
  }

  int n,dimension;
  fin >> n >> dimension;

  MVector_h hgeo_x;
  MVector_h hgeo_y;
  MVector* geo_x = new MVector;
  MVector* geo_y = new MVector;
  hgeo_x.resize(n);
  hgeo_y.resize(n);

  if (dimension == 3)
  {
    MVector_h hgeo_z;
    MVector* geo_z = new MVector;
    hgeo_z.resize(n);
    for(int i = 0;i < n;i ++)
        fin >> hgeo_x[i] >> hgeo_y[i] >> hgeo_z[i];
    *geo_z = hgeo_z;
    obj->setParameterPtr< MVector > ("geo.z", geo_z);
  }
  else if (dimension == 2)
  {
    for(int i = 0;i < n;i ++)
        fin >> hgeo_x[i] >> hgeo_y[i];
  }

  obj->setParameter<int>("dim", dimension);
  obj->setParameter<int>("geo_size",(int)(hgeo_x.size()));
  *geo_x = hgeo_x;
  *geo_y = hgeo_y;
  obj->setParameterPtr< MVector > ("geo.x", geo_x);
  obj->setParameterPtr< MVector > ("geo.y", geo_y);

return AMGX_OK;
}

template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::readColoring( AuxData* obj, const char* fname)
{
  std::string err;
  err = "Reading matrix in file: " + string(fname) + "\n";
  amgx_output(err.c_str(), err.length());

  std::ifstream fin(fname);
  if(!fin) {
    err = "Error opening file '" + string(fname) + "'\n";
      FatalError(err.c_str(), AMGX_ERR_IO);
  }

  int num_rows, num_colors;
  fin >> num_rows >> num_colors;

  typedef TemplateConfig<AMGX_host, T_Config::vecPrec, T_Config::matPrec, T_Config::indPrec> TConfig_h;
  typedef typename Matrix<TConfig_h>::IVector IVector_h;
  IVector_h* row_coloring = new IVector_h;

  row_coloring->resize(num_rows);

  for(int i = 0;i < num_rows;i ++)
      fin >> (*row_coloring)[i];

  obj->setParameter<int>("coloring_size", num_rows);
  obj->setParameter<int>("colors_num", num_colors);
  obj->setParameterPtr< IVector_h > ("coloring", row_coloring);

return AMGX_OK;
}


template<class T_Config>
AMGX_ERROR MatrixIO<T_Config>::readGeometry( AuxData* obj, int n,int dimension )
{
  typedef typename Matrix<T_Config>::MVector VVector;
  MVector_h geo_x;
  MVector_h geo_y;
  geo_x.resize(n);
  geo_y.resize(n);

  int num_one_dim;
  if (dimension == 3)
  {
    MVector_h geo_z;
    geo_z.resize(n);
    num_one_dim = (int) cbrt((double)n);
    for (int i = 0;i < num_one_dim;i++)
        for (int j = 0;j < num_one_dim;j++)
            for (int k = 0;k < num_one_dim;k++)
            {
                geo_x[i + j*num_one_dim + k*num_one_dim*num_one_dim] = 1.0*i/(num_one_dim-1);
                geo_y[i + j*num_one_dim + k*num_one_dim*num_one_dim] = 1.0*j/(num_one_dim-1);
                geo_z[i + j*num_one_dim + k*num_one_dim*num_one_dim] = 1.0*k/(num_one_dim-1);
            }
    VVector *dgeo_z = new VVector;
    *dgeo_z = geo_z;
    obj->setParameterPtr< VVector > ("geo.z", dgeo_z);
  }
  else if (dimension == 2)
  {
    num_one_dim = (int) sqrt((double)n);
    for (int i = 0;i < num_one_dim;i++)
        for (int j = 0;j < num_one_dim;j++)
        {
            geo_x[i + j*num_one_dim] = 1.0*i/(num_one_dim-1);
            geo_y[i + j*num_one_dim] = 1.0*j/(num_one_dim-1);
            //(*geo_z)[i + j*num_one_dim] = 0;
        }
  }

  VVector *dgeo_y = new VVector;
  VVector *dgeo_x = new VVector;
  *dgeo_y = geo_y;
  *dgeo_x = geo_x;
  obj->setParameter<int>("dim", dimension);
  obj->setParameter<int>("geo_size",(int)(n));
  obj->setParameterPtr< VVector > ("geo.x", dgeo_x);
  obj->setParameterPtr< VVector > ("geo.y", dgeo_y);

  return AMGX_OK;
}*/
/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MatrixIO<TemplateMode<CASE>::Type >;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
//AMGX_FORCOMPLEX_BUILDS_DEVICE(AMGX_CASE_LINE)
//  template class MatrixIO<Matrix_d>;
//  template class MatrixIO<Matrix_h>;
} // end namespace amgx
