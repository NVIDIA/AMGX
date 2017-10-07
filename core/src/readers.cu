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

#include <readers.h>
#include <multiply.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <iterator>
#include <algorithm>
#include <amgx_types/util.h>
#include <amgx_types/io.h>

namespace amgx
{


template <typename T>
void LoadValueFromStream(std::ifstream &fin, T &val);

template <>
void LoadValueFromStream(std::ifstream &fin, float &val)
{
    fin >> val;
}

template <>
void LoadValueFromStream(std::ifstream &fin, double &val)
{
    fin >> val;
}

template <>
void LoadValueFromStream(std::ifstream &fin, cuComplex &val)
{
    float x, y;
    fin >> x >> y;
    val = make_cuComplex(x, y);
}

template <>
void LoadValueFromStream(std::ifstream &fin, cuDoubleComplex &val)
{
    double x, y;
    fin >> x >> y;
    val = make_cuDoubleComplex(x, y);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
bool LoadVector(std::ifstream &fin, bool read_all, int rows_total, int block_size, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &vec, const std::map<const int, int> &GlobalToLocalRowMap = std::map<const int, int>())
{
    std::map<const int, int>::const_iterator gtl_it;
    //std::vector<double> block_vals(block_size);
    typedef typename VecPrecisionMap<t_vecPrec>::Type value_type;
    std::vector<value_type> block_vals(block_size);
    //for each entry
    int idx = 0;

    if (fin)
    {
        for (int i = 0; i < rows_total; i++)
        {
            //read entry
            for (int k = 0; k < block_size; k++)
            {
                LoadValueFromStream(fin, block_vals[k]);
            }

            //fin >> block_vals[k];
            if (read_all)
                for (int k = 0; k < block_size; k++)
                {
                    vec[i * block_size + k] = block_vals[k];
                    idx++;
                }
            else
            {
                gtl_it = GlobalToLocalRowMap.find(i);

                if (gtl_it != GlobalToLocalRowMap.end())
                {
                    for (int k = 0; k < block_size; k++)
                    {
                        vec[gtl_it->second * block_size + k] = block_vals[k];
                        idx++;
                    }
                }
            }
        }

        if (idx != vec.size())
        {
            FatalError("Matrix Market reader rows mismatch", AMGX_ERR_IO);
        }
    }
    else
    {
        return false;
    }

    return true;
}

// Distrubuted version
void skip_vals(ifstream &fin, int num_values)
{
    double val;

    for (long int i = 0; i < num_values; i++)
    {
        fin >> val;
    }
}


template <typename T>
T getBoostValue();

template <>
float getBoostValue()
{
    return 1.e-6f;
}

template <>
double getBoostValue()
{
    return 1.e-6;
}

template <>
cuComplex getBoostValue()
{
    return make_cuComplex(1e-6f, 0.f);
}

template <>
cuDoubleComplex getBoostValue()
{
    return make_cuDoubleComplex(1e-6, 0.);
}

template<AMGX_VecPrecision prec>
struct vecRealToComplexPrec
{
    static const AMGX_VecPrecision CPrec = prec;
};

template <>
struct vecRealToComplexPrec<AMGX_vecDouble>
{
    static const AMGX_VecPrecision CPrec = AMGX_vecDoubleComplex;
};

template <>
struct vecRealToComplexPrec<AMGX_vecFloat>
{
    static const AMGX_VecPrecision CPrec = AMGX_vecComplex;
};

template<AMGX_MatPrecision prec>
struct matRealToComplexPrec
{
    static const AMGX_MatPrecision CPrec = prec;
};

template <>
struct matRealToComplexPrec<AMGX_matDouble>
{
    static const AMGX_MatPrecision CPrec = AMGX_matDoubleComplex;
};

template <>
struct matRealToComplexPrec<AMGX_matFloat>
{
    static const AMGX_MatPrecision CPrec = AMGX_matComplex;
};

template <class TReal, class TComplex, class PartVec, bool init_flag>
struct ReadAndConvert;

template <class TReal, class TComplex, class PartVec>
struct ReadAndConvert<TReal, TComplex, PartVec, true>
{
    static void readAndConvert(std::ifstream &fin, const char *fname, int conversion_type
                               , Matrix<TReal> &A
                               , Vector<TReal> &b
                               , Vector<TReal> &x
                               , unsigned int props
                               , const PartVec &rank_rows)
    {
        FatalError("Converversion from complex matrix to ERF, but one of the complex modes is specified", AMGX_ERR_IO);
    }
};

template <class TReal, class TComplex, class PartVec>
struct ReadAndConvert<TReal, TComplex, PartVec, false>
{
    static void readAndConvert(std::ifstream &fin, const char *fname, int conversion_type
                               , Matrix<TReal> &A
                               , Vector<TReal> &b
                               , Vector<TReal> &x
                               , unsigned int props
                               , const PartVec &rank_rows)
    {
        AMG_Config tcfg;
        Matrix<TComplex> Ac;
        Vector<TComplex> xc, bc;
        typedef typename TReal::MatPrec RValueTypeA;
        typedef typename TReal::VecPrec RValueTypeB;
        typedef typename TComplex::MatPrec CValueTypeA;
        typedef typename TComplex::VecPrec CValueTypeB;
        printf("ERF conversion: reading complex valued system\n");
        fflush(stdout);
        ReadMatrixMarket<TComplex>::readMatrixMarket(fin, fname, Ac, bc, xc, tcfg);

        // modes = 1..4 - convert to the scalar system of 2x size using K1..K4 formulation
        if (conversion_type > 0 && conversion_type < 5)
        {
            // fill CSR values, common for all modes
            int cnrows = Ac.get_num_rows();
            int cnnz = Ac.get_num_nz();
            int nrows = cnrows * 2;
            int nnz = Ac.get_num_nz() * 4;
            A.addProps(CSR);
            A.resize(nrows, nrows, nnz);

            // set row offsets
            for (int i = 0; i < cnrows; i++)
            {
                A.row_offsets[i]          = Ac.row_offsets[i] * 2;
                A.row_offsets[i + cnrows] = Ac.row_offsets[i] * 2 + cnnz * 2;
            }

            A.row_offsets[nrows] = nnz;

            // set col indices
            for (int r = 0; r < nrows ; r++)
            {
                int *Ac_col_ptr = Ac.col_indices.raw() + Ac.row_offsets[r % cnrows];
                int row_nnz = A.row_offsets[r + 1] - A.row_offsets[r];

                for (int c = 0; c < (row_nnz / 2); c++)
                {
                    A.col_indices[A.row_offsets[r] + c] = Ac_col_ptr[c];
                    A.col_indices[A.row_offsets[r] + c + row_nnz / 2] = Ac_col_ptr[c] + nrows / 2;
                }
            }

            // set values
            for (int r = 0; r < cnrows; r++)
            {
                CValueTypeA *Ac_values = Ac.values.raw() + Ac.row_offsets[r];
                int row_nnz = Ac.row_offsets[r + 1] - Ac.row_offsets[r];

                for (int c = 0; c < row_nnz; c++)
                {
                    switch (conversion_type)
                    {
                        case 1:
                            A.values[A.row_offsets[r] + c]                    =  types::get_re(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz]          = -types::get_im(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + 2 * cnnz]           =  types::get_im(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz + 2 * cnnz] =  types::get_re(Ac_values[c]);
                            break;

                        case 2:
                            A.values[A.row_offsets[r] + c]                    =  types::get_re(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz]          =  types::get_im(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + 2 * cnnz]           =  types::get_im(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz + 2 * cnnz] = -types::get_re(Ac_values[c]);
                            break;

                        case 3:
                            A.values[A.row_offsets[r] + c]                    =  types::get_im(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz]          =  types::get_re(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + 2 * cnnz]           =  types::get_re(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz + 2 * cnnz] = -types::get_im(Ac_values[c]);
                            break;

                        case 4:
                            A.values[A.row_offsets[r] + c]                    =  types::get_im(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz]          = -types::get_re(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + 2 * cnnz]           =  types::get_re(Ac_values[c]);
                            A.values[A.row_offsets[r] + c + row_nnz + 2 * cnnz] =  types::get_im(Ac_values[c]);
                            break;
                    }
                }
            }

            // set b
            b.set_block_dimx(1);
            b.set_block_dimy(1);
            b.resize(nrows);

            for (int r = 0; r < cnrows; r++)
            {
                switch (conversion_type)
                {
                    case 1:
                    case 2:
                        b[r]          = types::get_re(bc[r]);
                        b[r + cnrows] = types::get_im(bc[r]);
                        break;

                    case 3:
                    case 4:
                        b[r]          = types::get_im(bc[r]);
                        b[r + cnrows] = types::get_re(bc[r]);
                        break;
                }
            }

            //set x if needed
            x.set_block_dimx(1);
            x.set_block_dimy(1);

            if (xc.size() > 0)
            {
                // set b
                x.resize(nrows);

                for (int r = 0; r < cnrows; r++)
                {
                    switch (conversion_type)
                    {
                        case 1:
                        case 3:
                            x[r]          =  types::get_re(xc[r]);
                            x[r + cnrows] =  types::get_im(xc[r]);
                            break;

                        case 2:
                        case 4:
                            x[r]          =  types::get_re(xc[r]);
                            x[r + cnrows] = -types::get_im(xc[r]);
                            break;
                    }
                }
            }

            A.computeDiagonal();
            std::stringstream info;
            info << "Converted complex matrix " << cnrows << "x" << cnrows << " with " << cnnz << " nonzeros to the ERF - using K" << conversion_type << " formulation." << std::endl;
            std::cout << info.str();
        }
        // modes 221..224 - convert to the system of the same size but with 2x2 blocks,
        // where each block converted from original Aij value using K1..K4 formulation
        // this switch is for original blocksize of 1
        else if (conversion_type > 220 && conversion_type < 225  && Ac.get_block_dimy()*Ac.get_block_dimx() == 1)
        {
            // fill CSR values, common for all modes
            int nrows = Ac.get_num_rows();
            int nnz = Ac.get_num_nz();
            A.addProps(Ac.hasProps(DIAG) ? CSR | DIAG : CSR);
            A.resize(nrows, nrows, nnz, 2 * Ac.get_block_dimx(), 2 * Ac.get_block_dimy(), 1);
            thrust::copy(Ac.row_offsets.begin(), Ac.row_offsets.end(), A.row_offsets.begin());
            thrust::copy(Ac.col_indices.begin(), Ac.col_indices.end(), A.col_indices.begin());

            for (int i = 0; i < nnz; i++)
            {
                switch (conversion_type)
                {
                    case 221:
                        A.values[4 * i    ] =  types::get_re(Ac.values[i]);
                        A.values[4 * i + 1] = -types::get_im(Ac.values[i]);
                        A.values[4 * i + 2] =  types::get_im(Ac.values[i]);
                        A.values[4 * i + 3] =  types::get_re(Ac.values[i]);
                        break;

                    case 222:
                        A.values[4 * i    ] =  types::get_re(Ac.values[i]);
                        A.values[4 * i + 1] =  types::get_im(Ac.values[i]);
                        A.values[4 * i + 2] =  types::get_im(Ac.values[i]);
                        A.values[4 * i + 3] = -types::get_re(Ac.values[i]);
                        break;

                    case 223:
                        A.values[4 * i    ] =  types::get_im(Ac.values[i]);
                        A.values[4 * i + 1] =  types::get_re(Ac.values[i]);
                        A.values[4 * i + 2] =  types::get_re(Ac.values[i]);
                        A.values[4 * i + 3] = -types::get_im(Ac.values[i]);
                        break;

                    case 224:
                        A.values[4 * i    ] =  types::get_im(Ac.values[i]);
                        A.values[4 * i + 1] = -types::get_re(Ac.values[i]);
                        A.values[4 * i + 2] =  types::get_re(Ac.values[i]);
                        A.values[4 * i + 3] =  types::get_im(Ac.values[i]);
                        break;
                }
            }

            A.computeDiagonal();
            b.resize(nrows * 2);
            b.set_block_dimx(1);
            b.set_block_dimy(2);

            for (int r = 0; r < nrows; r++)
            {
                switch (conversion_type)
                {
                    case 221:
                    case 222:
                        b[2 * r    ] = types::get_re(bc[r]);
                        b[2 * r + 1] = types::get_im(bc[r]);
                        break;

                    case 223:
                    case 224:
                        b[2 * r    ] = types::get_im(bc[r]);
                        b[2 * r + 1] = types::get_re(bc[r]);
                        break;
                }
            }

            //set x if needed
            if (xc.size() > 0)
            {
                // set b
                x.resize(nrows * 2);
                x.set_block_dimx(1);
                x.set_block_dimy(2);

                for (int r = 0; r < nrows; r++)
                {
                    switch (conversion_type)
                    {
                        case 221:
                        case 223:
                            x[2 * r    ] =  types::get_re(xc[r]);
                            x[2 * r + 1] =  types::get_im(xc[r]);
                            break;

                        case 222:
                        case 224:
                            x[2 * r    ] =  types::get_re(xc[r]);
                            x[2 * r + 1] = -types::get_im(xc[r]);
                            break;
                    }
                }
            }

            std::stringstream info;
            info << "Converted complex matrix " << nrows << "x" << nrows << " with " << nnz << " nonzeros to the (2x2) block-ERF - using K" << conversion_type - 220 << " formulation." << std::endl;
            std::cout << info.str();
        }
        // modes 221..224 - convert to the system of the same size but with 2x2 blocks,
        // where each block converted from original Aij value using K1..K4 formulation
        // this switch is for original blocksize of 1
        else if (conversion_type > 220 && conversion_type < 225)
        {
            // fill CSR values, common for all modes
            int nrows = Ac.get_num_rows();
            int nnz = Ac.get_num_nz();
            A.addProps(Ac.hasProps(DIAG) ? (CSR | DIAG) : CSR);
            int bdimx = 2 * Ac.get_block_dimx();
            int bdimy = 2 * Ac.get_block_dimy();
            A.resize(nrows, nrows, nnz, bdimx, bdimy, 1);
            thrust::copy(Ac.row_offsets.begin(), Ac.row_offsets.end(), A.row_offsets.begin());
            thrust::copy(Ac.col_indices.begin(), Ac.col_indices.end(), A.col_indices.begin());
            thrust::fill(A.values.begin(), A.values.end(), amgx::types::util<RValueTypeA>::get_zero());
            std::cout << "Input block system " << Ac.get_block_dimx() << "x" << Ac.get_block_dimy() << " will be converted to system with blocks " << bdimx << "x" << bdimy << std::endl;
            std::cout << "Converting values...\n";

            // iterate through blocks
            for (int i = 0; i < nnz; i++)
            {
                int block_offsetc = Ac.get_block_dimx() * Ac.get_block_dimy() * i;
                int block_offset = bdimx * bdimy * i;

                // iterate through values in the blocks
                for (int j = 0; j < Ac.get_block_dimx()*Ac.get_block_dimy(); j++)
                {
                    int cx = j / Ac.get_block_dimy();
                    int cy = j % Ac.get_block_dimy();
                    // interleaved blocks
                    int val_offset = block_offset + cx * bdimx + cy;

                    // in-place blocks
                    //int val_offset = block_offset + 2 * cx * bdimx + 2 * cy;
                    switch (conversion_type)
                    {
                        case 221:
                            // interleaved blocks
                            A.values[val_offset                                                 ] =  types::get_re(Ac.values[block_offsetc + j]);
                            A.values[val_offset +                           Ac.get_block_dimx() ] = -types::get_im(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 2 * Ac.get_block_size()                       ] =  types::get_im(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 2 * Ac.get_block_size() + Ac.get_block_dimx() ] =  types::get_re(Ac.values[block_offsetc + j]);
                            // in-place blocks
                            //A.values[val_offset             ] =  types::get_re(Ac.values[block_offsetc + j]);
                            //A.values[val_offset + 1         ] = -types::get_im(Ac.values[block_offsetc + j]);
                            //A.values[val_offset     + bdimx ] =  types::get_im(Ac.values[block_offsetc + j]);
                            //A.values[val_offset + 1 + bdimx ] =  types::get_re(Ac.values[block_offsetc + j]);
                            break;
                        case 222:
                            A.values[val_offset            ] =  types::get_re(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 1        ] =  types::get_im(Ac.values[block_offsetc + j]);
                            A.values[val_offset     + bdimx] =  types::get_im(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 1 + bdimx] = -types::get_re(Ac.values[block_offsetc + j]);
                            break;

                        case 223:
                            A.values[val_offset            ] =  types::get_im(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 1        ] =  types::get_re(Ac.values[block_offsetc + j]);
                            A.values[val_offset     + bdimx] =  types::get_re(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 1 + bdimx] = -types::get_im(Ac.values[block_offsetc + j]);
                            break;

                        case 224:
                            A.values[val_offset            ] =  types::get_im(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 1        ] = -types::get_re(Ac.values[block_offsetc + j]);
                            A.values[val_offset     + bdimx] =  types::get_re(Ac.values[block_offsetc + j]);
                            A.values[val_offset + 1 + bdimx] =  types::get_im(Ac.values[block_offsetc + j]);
                            break;
                    }
                }
            }

            std::cout << "Compute diagonal\n";
            A.computeDiagonal();

            // if external diagonal - convert those values too
            if (A.hasProps(DIAG))
            {
                std::cout << "Convert diagonal (warning!)\n";

                for (int i = 0; i < Ac.get_num_rows(); i++)
                {
                    int block_offsetc = Ac.diag[i] * Ac.get_block_dimx() * Ac.get_block_dimy();
                    int block_offset = A.diag[i] * bdimx * bdimy;

                    for (int j = 0; j < Ac.get_block_dimx()*Ac.get_block_dimy(); j++)
                    {
                        int val_offset = block_offset + (j / bdimx) * 2 * bdimx + (j % bdimx) * 2;

                        switch (conversion_type)
                        {
                            case 221:
                                A.values[val_offset            ] =  types::get_re(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1        ] = -types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset     + bdimx] =  types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1 + bdimx] =  types::get_re(Ac.values[block_offsetc + j]);
                                break;

                            case 222:
                                A.values[val_offset            ] =  types::get_re(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1        ] = -types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset     + bdimx] =  types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1 + bdimx] =  types::get_re(Ac.values[block_offsetc + j]);
                                break;

                            case 223:
                                A.values[val_offset            ] =  types::get_re(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1        ] = -types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset     + bdimx] =  types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1 + bdimx] =  types::get_re(Ac.values[block_offsetc + j]);
                                break;

                            case 224:
                                A.values[val_offset            ] =  types::get_re(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1        ] = -types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset     + bdimx] =  types::get_im(Ac.values[block_offsetc + j]);
                                A.values[val_offset + 1 + bdimx] =  types::get_re(Ac.values[block_offsetc + j]);
                                break;
                        }
                    }
                }
            }

            std::cout << "Convert rhs\n";
            b.resize(nrows * bdimy);
            b.set_block_dimx(1);
            b.set_block_dimy(bdimy);

            // interleaved blocks
            for (int r = 0; r < nrows; r++)
            {
                for (int j = 0; j < Ac.get_block_dimy(); j++)
                {
                    switch (conversion_type)
                    {
                        case 221:
                        case 222:
                            b[r * bdimy + j                      ] = types::get_re(bc[r * Ac.get_block_dimy() + j]);
                            b[r * bdimy + j + Ac.get_block_dimy()] = types::get_im(bc[r * Ac.get_block_dimy() + j]);
                            break;

                        case 223:
                        case 224:
                            b[r * bdimy + j                      ] = types::get_im(bc[r * Ac.get_block_dimy() + j]);
                            b[r * bdimy + j + Ac.get_block_dimy()] = types::get_re(bc[r * Ac.get_block_dimy() + j]);
                            break;
                    }
                }
            }

            std::cout << "Convert soln\n";

            //set x if needed
            if (xc.size() > 0)
            {
                x.resize(nrows * bdimx);
                x.set_block_dimx(1);
                x.set_block_dimy(bdimy);

                // interleaved blocks
                for (int r = 0; r < nrows; r++)
                {
                    for (int j = 0; j < Ac.get_block_dimx(); j++)
                    {
                        switch (conversion_type)
                        {
                            case 221:
                            case 223:
                                x[r * bdimx + j                      ] = types::get_re(xc[r * Ac.get_block_dimx() + j]);
                                x[r * bdimx + j + Ac.get_block_dimx()] = types::get_im(xc[r * Ac.get_block_dimx() + j]);
                                break;

                            case 222:
                            case 224:
                                x[r * bdimx + j                      ] = types::get_re(xc[r * Ac.get_block_dimx() + j]);
                                x[r * bdimx + j + Ac.get_block_dimx()] =-types::get_im(xc[r * Ac.get_block_dimx() + j]);
                                break;
                        }
                    }
                }
            }

            std::stringstream info;
            info << "Converted complex matrix " << nrows << "x" << nrows << " with " << nnz << " nonzeros to the (2x2) block-ERF - using K" << conversion_type - 220 << " formulation." << std::endl;
            std::cout << info.str();
        }
    }
};


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
bool ReadMatrixMarket<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::readMatrixMarket(std::ifstream &fin, const char *fname, Matrix_h &A
        , Vector_h &b
        , Vector_h &x
        , const AMG_Config &cfg
        , unsigned int props
        , const IVector_h &rank_rows // row indices for given rank
                                                                                                    )
{
    fin.seekg(std::ios::beg);
    typedef typename Matrix_h::index_type IndexType;
    typedef typename Matrix_h::value_type ValueTypeA;// change later back to load in high precision!
    typedef typename TConfig_h::VecPrec ValueTypeB;
    std::string warning;
    int complex_conversion = cfg.AMG_Config::getParameter<IndexType>("complex_conversion", "default");

    // if we are in the real-valued mode and complex conversion is specified and we are reading actual matrix
    if (complex_conversion != 0 && !types::util<ValueTypeA>::is_complex && !types::util<ValueTypeB>::is_complex && !io_config::hasProps(io_config::SIZE, props))
    {
        // read system as complex valued system of same precision and convert it to our matrices
        typedef typename TConfig_h::template setVecPrec<vecRealToComplexPrec<TConfig_h::vecPrec>::CPrec>::Type TConfig_h_cvec;
        typedef typename TConfig_h_cvec::template setMatPrec<matRealToComplexPrec<TConfig_h::matPrec>::CPrec>::Type TComplex_h;
        ReadAndConvert<TConfig_h, TComplex_h, IVector_h, types::util<ValueTypeA>::is_complex>::readAndConvert(fin, fname, complex_conversion, A, b, x, props, rank_rows);
        return true;
    }

    //skip comments and read amgx relevant parameters
    std::list<string> nvConfig;
    std::list<string> mmConfig;
    // Workaround section to convert external diagonal into internal
    // in CLASSICAL
    bool isClassical = false;
    std::string solver_scope, solver_value;
    std::string precond_scope, precond_value;
    AlgorithmType algorithm_s, algorithm_p;
    Resources *resources = A.getResources();

    if (resources != NULL)
    {
        resources->getResourcesConfig()->getParameter<std::string>("solver", solver_value, "default", solver_scope);
        algorithm_s = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", solver_scope);
        resources->getResourcesConfig()->getParameter<std::string>("preconditioner", precond_value, solver_scope, precond_scope);
        algorithm_p = resources->getResourcesConfig()->getParameter<AlgorithmType>("algorithm", precond_scope);

        if (algorithm_s == CLASSICAL && algorithm_p == CLASSICAL)
        {
            isClassical = true;
        }
    }

    // End of CLASSICAL workaround
    bool has_zero_diagonal_element = false;
    bool check_zero_diagonal = false;
    const bool boost_zero_diagonal = false;
    ValueTypeA boostValue = getBoostValue<ValueTypeA>();

    if (boost_zero_diagonal) { check_zero_diagonal = true; }

    while (fin.peek() == '%')
    {
        std::string nvString;
        int fpos = fin.tellg(); // store current position
        getline(fin, nvString);
        std::transform(nvString.begin(), nvString.end(), nvString.begin(), ::tolower);
        std::istringstream nvString_s(nvString);
        std::string nvFormat;
        nvString_s >> nvFormat;

        if (nvFormat.size() > 2)
        {
            if ((nvFormat.substr(2, nvFormat.size()) == "nvamg") ||
                    (nvFormat.substr(2, nvFormat.size()) == "amgx"))
            {
                std::copy(istream_iterator<string>(nvString_s), istream_iterator<string>(),
                          back_inserter<list<string> >(nvConfig));
            }

            if (nvFormat.substr(2, nvFormat.size()) == "matrixmarket")
            {
                std::copy(istream_iterator<string>(nvString_s), istream_iterator<string>(),
                          back_inserter<list<string> >(mmConfig));
            }
        }

        fin.seekg(fpos, std::ios_base::beg);
        fin.ignore(INT_MAX, '\n');
    }

    // process MatrixMarket config string
    bool symmetric = false;
    bool skew_symmetric = false;
    bool complex = false;
    bool hermitian = false;

    if (mmConfig.size() > 0)
    {
        for (list<string>::const_iterator it = mmConfig.begin(); it != mmConfig.end(); ++it)
        {
            if (*it == "symmetric") {symmetric = true; continue;}

            if (*it == "complex")
            {
                if (!types::util<ValueTypeA>::is_complex && complex_conversion == 0)
                {
                    FatalError("Trying to load file with complex matrix to real valued matrix structure", AMGX_ERR_IO);
                }

                complex = true;
                continue;
            }

            if (*it == "real")
            {
                if (!types::util<ValueTypeA>::is_real)
                {
                    FatalError("Trying to load file with real matrix to complex valued matrix structure", AMGX_ERR_IO);
                }
            }

            if (*it == "pattern") {FatalError("'pattern' is not supported in %%MatrixMarket format string", AMGX_ERR_IO);}

            if (*it == "skew-symmetric") {symmetric = true; skew_symmetric = true; continue;}

            //if (*it == "skew-symmetric") {FatalError("'skew-symmetric' is not supported in %%MatrixMarket format string", AMGX_ERR_IO);}
            if (*it == "hermitian") {hermitian = true; continue;}
        }
    }

    // process amgx config string
    int block_dimx = 1, block_dimy = 1, index_base = 1;
    bool diag_prop = false, rhs = false, soln = false, mtx = false, sorted = false;
    list<int> block_sizes;

    if (nvConfig.size() > 0)
    {
        for (list<string>::const_iterator it = nvConfig.begin(); it != nvConfig.end(); ++it)
        {
            if (*it == "diagonal") {diag_prop = true; continue;}

            if (*it == "rhs") {rhs = true; continue;}

            if (*it == "solution") {soln = true; continue;}

            if (*it == "sorted") {sorted = true; continue;}

            if (*it == "base0") {index_base = 0; continue;}

            if (isdigit((*it)[0])) { int bsize; istringstream(*it) >> bsize; block_sizes.push_back(bsize); continue;};
        }
    }

    // CLASSICAL fix
    if (sorted && isClassical && diag_prop) { sorted = false; }

    // Currently not implemented sorted symmetric matrices
    if (sorted && symmetric || sorted && hermitian) { sorted = false; }

    if (std::find(mmConfig.begin(), mmConfig.end(), "matrix") != mmConfig.end()) { mtx = true; }

    if (block_sizes.size() == 2)
    {
        block_dimy = block_sizes.back();
        block_dimx = block_sizes.front();
    }
    else if (block_sizes.size() == 1)
    {
        block_dimy = block_dimx = block_sizes.back();
    }

    int fpos = fin.tellg(); // store current position
    int rows, cols, entries;
    //read rows cols entries
    fin >> rows >> cols >> entries;

    if (rows % block_dimx != 0 || cols % block_dimy != 0 || entries % (block_dimx * block_dimy) != 0)
    {
        FatalError("Matrix dimensions do not match with block sizes", AMGX_ERR_IO);
    }

    rows /= block_dimx;
    cols /= block_dimy;
    entries /= (block_dimx * block_dimy);

    if (io_config::hasProps(io_config::SIZE, props))
    {
        if (complex_conversion != 0 && block_dimy * block_dimx != 1)
        {
            FatalError("Complex conversion is supported only for non-coupled matrices with blocks of 1x1", AMGX_ERR_IO);
        }

        if (complex_conversion == 0)
        {
            A.set_num_rows(rows);
            A.set_num_cols(cols);
            A.set_block_dimy(block_dimy);
            A.set_block_dimx(block_dimx);
        }
        else if (complex_conversion > 0 && complex_conversion < 5)
        {
            // general ERF
            A.set_num_rows(rows * 2);
            A.set_num_cols(cols * 2);
            A.set_block_dimy(block_dimy);
            A.set_block_dimx(block_dimx);
        }
        else if (complex_conversion > 220 && complex_conversion < 225)
        {
            // 2x2 block ERF
            A.set_num_rows(rows);
            A.set_num_cols(cols);
            A.set_block_dimy(block_dimy * 2); // complex 1x1 only supported, which converts to 2x2 real blocks
            A.set_block_dimx(block_dimx * 2);
        }
        else
        {
            FatalError("Unsupported complex_conversion mode", AMGX_ERR_IO);
        }

        int num_entries = 0;

        if (symmetric || hermitian)
        {
            int i, j;
            int idiag = 0;
            ValueTypeA v;

            for (int e  = 0; e < entries * (block_dimx * block_dimy); e++)
            {
                fin >> i >> j;
                LoadValueFromStream(fin, v);

                // skip explicit zeroes, only block_size=1 is supported
                if (block_dimx == 1 && block_dimy == 1 && types::util<ValueTypeA>::is_zero(v))
                {
                    continue;
                }

                if (i == j)
                {
                    idiag++;
                }
            }

            num_entries = 2 * entries - idiag / (block_dimx * block_dimy);
        }
        else
        {
            if (isClassical && diag_prop)
            {
                num_entries = entries + rows;
            }
            else
            {
                num_entries = entries;
            }
        }

        if (complex_conversion == 0)
        {
            A.set_num_nz(num_entries);
        }
        else if (complex_conversion > 0 && complex_conversion < 5)
        {
            // general ERF
            A.set_num_nz(num_entries * 4);
        }
        else if (complex_conversion > 220 && complex_conversion < 225)
        {
            // 2x2 block ERF
            A.set_num_nz(num_entries);
        }
        else
        {
            FatalError("Unsupported complex_conversion mode", AMGX_ERR_IO);
        }

        return true;
    }

    warning = "Reading data...\n";

    if (isClassical && diag_prop)
    {
        warning += "Warning: external diagonal will be converted into internal for CLASSICAL path\n";
    }

    amgx_output(warning.c_str(), warning.length());

    // check for consistent input
    if (io_config::hasProps(io_config::MTX, props))
    {
        if (!mtx)
        {
            FatalError("Expecting 'matrix' keyword in %%MatrixMarket format string", AMGX_ERR_IO);
        }
    }
    else
    {
        if (mtx)
        {
            skip_vals(fin, 3 * entries * (block_dimy * block_dimx));

            if (diag_prop)
            {
                skip_vals(fin, rows * block_dimy * block_dimx);
            }
        }
    }

    bool read_all = (rank_rows.size() == 0) ? true : false;
    const IVector_h &partRowVec = rank_rows;
    int n_rows_part = (read_all) ? rows : partRowVec.size();
    std::map<const int, int> GlobalToLocalRowMap; // should try unordered_map
    std::map<const int, int>::const_iterator gtl_i;
    std::map<const int, int>::const_iterator gtl_j;

    // Generate inverse map for faster searching during the read
    if (!read_all)
        for (int i = 0; i < n_rows_part; i++)
        {
            GlobalToLocalRowMap.insert(std::pair<const int, int>(partRowVec[i], i));
        }

    if (io_config::hasProps(io_config::MTX, props))
    {
        int ival = 0, idiag = 0;
        int block_size = block_dimy * block_dimx;
        typedef std::map<const int, std::vector<ValueTypeA> > ColValuesMap_t;
        typedef std::pair<const int, std::vector<ValueTypeA> > ColValuesPair_t;
        typedef std::vector<ValueTypeA> ValuesVector_t;
        typedef std::vector<int> ColVector_t;
        std::vector<ColValuesMap_t> input;
        std::vector<int> nnz_per_row;
        //typename Matrix_h::MVector input_sorted_v;
        //IVector_h input_sorted_c;
        ValuesVector_t input_sorted_v;
        ColVector_t input_sorted_c;
        std::vector<int>  trackDiag;

        if (check_zero_diagonal)
        {
            trackDiag.resize(n_rows_part, 0);
        }

        if (sorted)
        {
            nnz_per_row.resize(n_rows_part, 0);

            if (read_all)
            {
                input_sorted_v.resize(entries * block_size);
                input_sorted_c.resize(entries);
            }
        }
        else
        {
            input.resize(n_rows_part);
        }

        typename Matrix_h::MVector diag(n_rows_part * block_size, types::util<ValueTypeA>::get_zero());
        std::vector<ValueTypeA> block_vals(block_size);
        //for each entry
        int i, j, ii, jj, i_old = -1;
        bool skip = false;
        bool has_ii = true, has_jj = false;

        if (symmetric || hermitian) { has_jj = true; }

        int explicit_zeroes = 0;

        for (int e = 0; e < entries; e++)
        {
            for (int kx = 0; kx < block_dimx; kx++)
                for (int ky = 0; ky < block_dimy; ky++)
                {
                    //read entry
                    fin >> i >> j;
                    LoadValueFromStream(fin, block_vals[kx * block_dimy + ky]);

                    // check we haven't been given a 0-indexed matrix
                    if ((i == 0 || j == 0) && index_base == 1)
                    {
                        FatalError("Matrix Market format requires 1-based indexing. Use 'base0' AMGX format option to override.", AMGX_ERR_IO);
                    }
                }

            // skip explicit zeroes, only block_size=1 is supported
            if (block_dimx == 1 && block_dimy == 1 && types::util<ValueTypeA>::is_zero(block_vals[0]))
            {
                explicit_zeroes++;

                if (i == j)
                {
                    idiag++;
                    has_zero_diagonal_element = true;

                    if (check_zero_diagonal)
                    {
                        trackDiag[i - index_base] = 0;
                    }
                }

                continue;
            }
            else
            {
                if (i == j)
                {
                    if (check_zero_diagonal)
                    {
                        trackDiag[i - index_base] = 1;
                    }
                }
            }

            i = (i - index_base) / block_dimx;
            j = (j - index_base) / block_dimy;

            if (!read_all)
                if (!symmetric && !hermitian)
                {
                    if (i != i_old) // reduce overhead of searching in GlobalToLocalRowMap
                    {
                        has_ii = false;
                        i_old = i;
                        gtl_i = GlobalToLocalRowMap.find(i);

                        if (gtl_i == GlobalToLocalRowMap.end())
                        {
                            skip = true;
                            continue;
                        }
                        else
                        {
                            has_ii = true;
                            skip = false;
                            ii =  gtl_i->second;
                        }
                    }
                    else if (skip)
                    {
                        continue;
                    }
                }
                else
                {
                    ii = i;
                    jj = j;

                    if (!read_all)
                    {
                        gtl_i = GlobalToLocalRowMap.find(i);
                        gtl_j = GlobalToLocalRowMap.find(j);
                        has_ii = has_jj = false;

                        if (gtl_i != GlobalToLocalRowMap.end()) { has_ii = true; }

                        if (gtl_j != GlobalToLocalRowMap.end()) { has_jj = true; }

                        if (!has_ii && !has_jj)
                        {
                            continue;
                        }
                        else
                        {
                            if (has_ii)
                            {
                                ii =  gtl_i->second;
                            }

                            if (has_jj)
                            {
                                jj =  gtl_j->second;
                            }
                        }
                    }
                }
            else
            {
                ii = i;

                if (symmetric || hermitian)
                {
                    jj = j;
                }
            }

            if (sorted)
            {
                nnz_per_row[ii]++;

                if (!read_all)
                {
                    input_sorted_v.insert(input_sorted_v.end(), block_vals.begin(), block_vals.end());
                    input_sorted_c.push_back(j);
                }
                else
                {
                    std::copy(block_vals.begin(), block_vals.end(), &input_sorted_v[ival * block_size]);
                    input_sorted_c[ival] = j;
                }

                ival++;
            }
            else
            {
                if (has_ii)
                {
                    ival++;
                    input[ii].insert(ColValuesPair_t(j, block_vals));
                }

                if (has_jj)
                {
                    ival++;

                    if ((skew_symmetric || hermitian) && i != j)
                        for (int k = 0; k < block_dimx * block_dimy; k++)
                        {
                            if (skew_symmetric)
                            {
                                block_vals[k] = types::util<ValueTypeA>::invert(block_vals[k]);
                            }
                            else if (hermitian)
                            {
                                block_vals[k] = types::util<ValueTypeA>::conjugate(block_vals[k]);
                            }
                        }

                    input[jj].insert(ColValuesPair_t(i, block_vals));
                }
            }

            if (i == j)
            {
                idiag++;
                std::copy(block_vals.begin(), block_vals.end(), &diag[ii * block_size]);
            }
        } // end of entries loop

        int diagIdx = 0;

        if (check_zero_diagonal)
        {
            for (int i = 0; i < rows; i++)
            {
                if (trackDiag[i] == 0)
                {
                    trackDiag[diagIdx] = i;
                    diagIdx++;
                }
            }
        }
        else
        {
            diagIdx = idiag;
        }

        if (has_zero_diagonal_element && block_dimx == 1 && block_dimy == 1)
        {
            if (check_zero_diagonal)
            {
                printf("Warning! Input matrix has zeroes on diagonal: %d %d\nZero diagonal elements are:\n", rows, diagIdx);

                for (int i = 0; i < diagIdx; i++)
                {
                    printf("%d ", trackDiag[i]);
                }

                printf("\n");
            }
        }

        if (boost_zero_diagonal && has_zero_diagonal_element && block_dimx == 1 && block_dimy == 1)
        {
            for (int i = 0; i < diagIdx; i++)
            {
                block_vals[0] = boostValue;
                input[ii].insert(ColValuesPair_t(trackDiag[i], block_vals));
            }
        }

        if (!(symmetric || hermitian) && (ival + explicit_zeroes) != entries && read_all)
        {
            FatalError("Matrix Market mismatch in number of entries", AMGX_ERR_IO);
        }

        IndexType n_nonzeros_part;

        if (symmetric || hermitian)
        {
            n_nonzeros_part = ival - idiag;
        }
        else
        {
            n_nonzeros_part = ival;
        }

        //if (symmetric)
        //  printf("Matrix is symmetric. Counted %d entries and %d diag elements, corresponding to %d nonzeroes\n ", ival, idiag, n_nonzeros_part);

        if (sorted && input_sorted_c.size() != n_nonzeros_part)
        {
            //printf("input_sorted_c.size() = %d n_nonzeros_part = %d\n", input_sorted_c.size(), n_nonzeros_part);
            FatalError("Matrix Market mismatch in number of entries", AMGX_ERR_IO);
        }

        if (sorted && input_sorted_v.size() != n_nonzeros_part * block_size)
        {
            //printf("input_sorted_v.size() = %d n_nonzeros_part*block_size = %d\n", input_sorted_v.size(), n_nonzeros_part*block_size);
            FatalError("Matrix Market mismatch in number of entries", AMGX_ERR_IO);
        }

        A.resize(0, 0, 0);
        //A.delProps(COO);
        A.addProps(CSR);

        if (diag_prop && !isClassical)
        {
            A.addProps(DIAG);
        }
        else
        {
            A.delProps(DIAG);
        }

        if (diag_prop)
        {
            LoadVector(fin, read_all, rows, block_size, diag, GlobalToLocalRowMap);
        }

        if (isClassical && diag_prop)
        {
            n_nonzeros_part = n_nonzeros_part + n_rows_part;

            for (int i = 0; i < n_rows_part; i++)
            {
                std::copy(&diag[i * block_size], &diag[i * block_size] + block_size, block_vals.begin());
                input[i].insert(ColValuesPair_t(read_all ? i : rank_rows[i], block_vals));
            }
        }

        A.resize(n_rows_part, cols, n_nonzeros_part, block_dimx, block_dimy);
        ValueTypeA *dia_values_ptr = thrust::raw_pointer_cast(&(A.values[block_dimx * block_dimy * n_nonzeros_part]));

        if (A.hasProps(CSR))
        {
            A.row_offsets[0] = 0;
            ival = 0;

            if (!sorted)
            {
                for (int i = 0; i < n_rows_part; i++)
                {
                    for (auto it = input[i].begin(); it != input[i].end(); it++)
                    {
                        A.col_indices[ival] = it->first;

                        for (int k = 0; k < block_size; k++)
                        {
                            A.values[ival * block_size + k] = it->second[k];
                        }

                        ival++;
                    }

                    A.row_offsets[i + 1] = ival;
                }
            }
            else
            {
                A.row_offsets[0] = 0;

                for (int i = 0; i < n_rows_part; i++)
                {
                    A.row_offsets[i + 1] = A.row_offsets[i] + nnz_per_row[i];
                }

                if (A.row_offsets[n_rows_part] != n_nonzeros_part)
                {
                    FatalError("Matrix Market mismatch in number of entries", AMGX_ERR_IO);
                }

                std::copy(input_sorted_c.begin(), input_sorted_c.end(), A.col_indices.begin());
                std::copy(input_sorted_v.begin(), input_sorted_v.end(), A.values.begin());
            }
        }
        else
        {
            FatalError("Matrix Market reader COO output is not supported", AMGX_ERR_IO);
        }

        if (diag_prop && !isClassical)
        {
            A.computeDiagonal();
        }

        if (A.hasProps(DIAG) && !isClassical)
            for (int i = 0; i < diag.size(); i++)
            {
                dia_values_ptr[i] = diag[i];
            }
    }// End of load matrix

    if (!io_config::hasProps(io_config::RHS, props))
        if (rhs)
        {
            skip_vals(fin, rows * block_dimy);
        }

    if (io_config::hasProps(io_config::RHS, props))
    {
        b.resize(n_rows_part * block_dimy);
        b.set_block_dimy(block_dimy);
        b.set_block_dimx(1);

        if (rhs)
        {
            LoadVector(fin, read_all, rows, block_dimy, b, GlobalToLocalRowMap);
        }
        else
        {
            //initialize RHS
            if (io_config::hasProps(io_config::GEN_RHS, props))
            {
                Vector_h b0(n_rows_part * block_dimy, types::util<ValueTypeB>::get_one());
                b0.set_block_dimy(block_dimy);
                b0.set_block_dimx(1);
                warning =  "RHS vector was not found. Using RHS b=A*e where e=[1,…,1]^T\n";
                A.set_initialized(true);
                multiply(A, b0, b);
                A.set_initialized(false);
            }
            else
            {
                warning = "RHS vector was not found. Using RHS b=[1,…,1]^T\n";

                for (int i = 0; i < n_rows_part * block_dimy; i++)
                {
                    b[i] = types::util<ValueTypeB>::get_one();
                }
            }

            amgx_output(warning.c_str(), warning.length());
        }
    }

    // try to read initial guess
    if (io_config::hasProps(io_config::SOLN, props))
    {
        x.resize(n_rows_part * block_dimx);
        x.set_block_dimy(block_dimy);
        x.set_block_dimx(1);

        if (soln)
        {
            LoadVector(fin, read_all, rows, block_dimx, x, GlobalToLocalRowMap);
        }
        else
        {
            warning = "Solution vector was not found. Setting initial solution to x=[0,…,0]^T\n";

            for (int i = 0; i < n_rows_part * block_dimx; i++)
            {
                x[i] = types::util<ValueTypeB>::get_zero();
            }
        }

        amgx_output(warning.c_str(), warning.length());
    }

    if (rank_rows.size() > 0)
    {
        A.set_is_matrix_read_partitioned(true);
        b.set_is_vector_read_partitioned(true);

        if (x.size() > 0)
        {
            x.set_is_vector_read_partitioned(true);
        }
    }

    warning = "";

    if (has_zero_diagonal_element || skew_symmetric)
    {
        warning += "Warning: Matrix has at least one zero on its diagonal\n";
    }

    warning = +"Finished reading\n";
    amgx_output(warning.c_str(), warning.length());
    return true;
}


// Distrubuted version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
bool ReadMatrixMarket<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::readMatrixMarketV2(std::ifstream &fin, const char *fname, Matrix_h &A
        , Vector_h &b
        , Vector_h &x
        , const AMG_Config &cfg
        , unsigned int props
        , const IVector_h &rank_rows // row indices for given rank
                                                                                                      )
{
    fin.seekg(std::ios::beg);
    typedef typename Matrix_h::index_type IndexType;
    typedef typename Matrix_h::value_type ValueTypeA;// change later back to load in high precision!
    typedef typename TConfig_h::VecPrec ValueTypeB;

    //skip comments
    while (fin.peek() == '%')
    {
        fin.ignore(INT_MAX, '\n');
    }

    int rows, cols, entries, block_dimx, block_dimy, diag_prop;
    //read rows cols entries
    fin >> rows >> cols >> entries >> block_dimx >> block_dimy >> diag_prop;

    if (io_config::hasProps(io_config::SIZE, props))
    {
        A.set_num_rows(rows);
        A.set_num_cols(cols);
        A.set_num_nz(entries);
        A.set_block_dimy(block_dimy);
        A.set_block_dimx(block_dimx);
        return true;
    }

    fflush(stdout);
    bool read_all = (rank_rows.size() == 0) ? true : false;
    const IVector_h &partRowVec = rank_rows;
    int n_rows_part = (read_all) ? rows : partRowVec.size();
    std::map<const int, int> GlobalToLocalRowMap; // should try unordered_map
    std::map<const int, int>::const_iterator gtl_it;

    // Generate inverse map for faster searching during the read
    if (!read_all)
        for (int i = 0; i < n_rows_part; i++)
        {
            GlobalToLocalRowMap.insert(std::pair<const int, int>(partRowVec[i], i));
        }

    typedef std::map<const int, std::vector<ValueTypeA> > ColValuesMap_t;
    typedef std::pair<const int, std::vector<ValueTypeA> > ColValuesPair_t;
    std::vector<ColValuesMap_t> input(n_rows_part);
    int ival = 0;
    int block_size = block_dimy * block_dimx;
    typename Matrix_h::MVector diag(n_rows_part * block_size, types::util<ValueTypeA>::get_zero());
    std::vector<ValueTypeA> block_vals(block_size);

    //for each entry
    for (int e = 0; e < entries; e++)
    {
        int i, j;
        //read entry
        fin >> i >> j;

        // check we haven't been given a 0-indexed matrix
        if (i == 0 || j == 0)
        {
            FatalError("Matrix Market format requires 1-based indexing", AMGX_ERR_IO);
        }

        for (int k = 0; k < block_size; k++)
        {
            LoadValueFromStream(fin, block_vals[k]);
        }

        if (!read_all)
        {
            gtl_it = GlobalToLocalRowMap.find(i - 1);

            if (gtl_it != GlobalToLocalRowMap.end())
            {
                input[gtl_it->second].insert(ColValuesPair_t(j - 1, block_vals));

                if (i == j)
                    for (int k = 0; k < block_size; k++)
                    {
                        diag[block_size * gtl_it->second + k] = block_vals[k];
                    }

                ival++;
            }
        }
        else
        {
            input[i - 1].insert(ColValuesPair_t(j - 1, block_vals));

            if (i == j)
                for (int k = 0; k < block_size; k++)
                {
                    diag[block_size * (i - 1) + k] = block_vals[k];
                }

            ival++;
        }
    }

    if (ival != entries && read_all)
    {
        FatalError("Matrix Market mismatch in number of entries", AMGX_ERR_IO);
    }

    IndexType n_nonzeros_part = ival;
    A.resize(0, 0, 0);
    //A.delProps(COO);
    A.addProps(CSR);

    if (diag_prop)
    {
        A.addProps(DIAG);
    }
    else
    {
        A.delProps(DIAG);
    }

    A.resize(n_rows_part, cols, n_nonzeros_part, block_dimx, block_dimy);
    ValueTypeA *dia_values_ptr = thrust::raw_pointer_cast(&(A.values[block_dimx * block_dimy * n_nonzeros_part]));

    if (A.hasProps(CSR))
    {
        A.row_offsets[0] = 0;
        ival = 0;

        for (int i = 0; i < n_rows_part; i++)
        {
            for (auto it = input[i].begin(); it != input[i].end(); it++)
            {
                A.col_indices[ival] = it->first;

                for (int k = 0; k < block_size; k++)
                {
                    A.values[ival * block_size + k] = it->second[k];
                }

                ival++;
            }

            A.row_offsets[i + 1] = ival;
        }
    }
    else
    {
        FatalError("Matrix Market reader COO output is not supported", AMGX_ERR_IO);
    }

    if (diag_prop)
    {
        A.computeDiagonal();
        LoadVector(fin, read_all, rows, block_size, diag, GlobalToLocalRowMap);
    }

    if (A.hasProps(DIAG))
        for (int i = 0; i < diag.size(); i++)
        {
            dia_values_ptr[i] = diag[i];
        }

    if (io_config::hasProps(io_config::RHS, props))
    {
        b.resize(n_rows_part * block_dimy);
        b.set_block_dimy(block_dimy);

        //initialize RHS
        for (int i = 0; i < n_rows_part * block_dimy; i++)
        {
            b[i] = types::util<ValueTypeB>::get_one();
        }

        //read num rows
        fin >> rows;
        LoadVector(fin, read_all, rows / block_dimy, block_dimy, b, GlobalToLocalRowMap);
    }

    // try to read initial guess

    if (io_config::hasProps(io_config::SOLN, props))
    {
        fin >> rows;

        if (rows)
        {
            x.resize(n_rows_part * block_dimx);
            x.set_block_dimy(block_dimx);
            LoadVector(fin, read_all, rows / block_dimx, block_dimx, x, GlobalToLocalRowMap);
        }
        else
        {
            x.resize(0);
        }
    }

    if (rank_rows.size() > 0)
    {
        A.set_is_matrix_read_partitioned(true);
        b.set_is_vector_read_partitioned(true);

        if (x.size() > 0)
        {
            x.set_is_vector_read_partitioned(true);
        }
    }

    return true;
}


template <typename TSRC, typename TDST>
void val_copy(const TSRC *src, TDST *dst, int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = static_cast<TDST>(src[i]);
    }
}

template <>
void val_copy<cuDoubleComplex, cuComplex>(const cuDoubleComplex *src, cuComplex *dst, int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = types::util<cuDoubleComplex>::to_downtype(src[i]);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
bool ReadNVAMGBinary<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::read(std::ifstream &finstr, const char *fnamec
        , Matrix_h &A
        , Vector_h &b, Vector_h &x
        , const AMG_Config &cfg
        , unsigned int props
        , const IVector_h &rank_rows
                                                                                       )
{
    typedef typename Matrix_h::index_type IndexType;
    typedef typename Matrix_h::value_type ValueTypeA;
    typedef typename Vector_h::value_type ValueTypeB; // change back to matrix type later
    typedef typename types::util<ValueTypeA>::uptype UpValueTypeA;
    size_t is_read;
    std::string err;
    finstr.close();
    FILE *fin = fopen(fnamec, "rb");

    if (fin == NULL)
    {
        err = "Error: couldn't open file " + std::string(fnamec);
    }

    char text_header[255];
    uint32_t system_flags [9];
    is_read = fread(text_header, sizeof(char), strlen("%%NVAMGBinary\n"), fin);
    is_read = fread(system_flags, sizeof(uint32_t), 9, fin);
    //bool is_mtx = system_flags[0];
    bool is_rhs = system_flags[1];
    bool is_soln = system_flags[2];
    uint32_t matrix_format = system_flags[3];
    bool diag = system_flags[4];
    uint32_t block_dimx = system_flags[5];
    uint32_t block_dimy = system_flags[6];
    uint32_t num_rows = system_flags[7];
    uint32_t num_nz = system_flags[8];

    if (io_config::hasProps(io_config::SIZE, props))
    {
        A.set_num_rows(num_rows);
        A.set_num_cols(num_rows);
        A.set_num_nz(num_nz);
        A.set_block_dimy(block_dimy);
        A.set_block_dimx(block_dimx);
        fclose(fin);
        return true;
    }

    long int data_pos = ftell(fin);
    IVector_h *partRowVec_p = NULL;

    if (rank_rows.size() == 0)
    {
        partRowVec_p = new IVector_h(num_rows);
        thrust::sequence(partRowVec_p->begin(), partRowVec_p->end());
        cudaCheckError();
    }
    else
    {
        partRowVec_p = (IVector_h *) &rank_rows;
    }

    IVector_h &partRowVec = *partRowVec_p;
    int n_rows_part = partRowVec.size();
    IVector_h row_offsets_part(n_rows_part + 1);
    IVector_h row_start_glb(n_rows_part); // Store global row start positions here
    int beginEnd[2];
    int n_nonzeros_part = 0;

    for (int i = 0; i < partRowVec.size(); i++)
    {
        if (fseek(fin, data_pos + partRowVec[i]*sizeof(int), SEEK_SET) != 0)
        {
            FatalError("fseek error", AMGX_ERR_IO);
        }

        is_read = fread(beginEnd, sizeof(int), 2, fin);

        if (is_read != 2)
        {
            err =  "fread failed reading row_offsets, exiting";
            FatalError(err, AMGX_ERR_IO);
        }

        row_start_glb[i] = beginEnd[0];
        row_offsets_part[i] = n_nonzeros_part;
        n_nonzeros_part += beginEnd[1] - beginEnd[0];
    }

    row_offsets_part[n_rows_part] = n_nonzeros_part;
    A.delProps(DIAG | COLORING);

    if ((matrix_format & COMPLEX) && types::util<ValueTypeA>::is_real)
    {
        FatalError("Matrix is in complex format, but reading as real AMGX mode", AMGX_ERR_IO);
    }

    if (!(matrix_format & COMPLEX) && types::util<ValueTypeA>::is_complex)
    {
        FatalError("Matrix is in real format, but reading as complex AMGX mode", AMGX_ERR_IO);
    }

    if (diag)
    {
        A.addProps(DIAG);
    }

    if (!(matrix_format & 1))
    {
        A.addProps(CSR);
    }
    else
    {
        FatalError("COO matrix binary format is not supported for reading.", AMGX_ERR_IO);
    }

    A.resize(n_rows_part, num_rows, n_nonzeros_part, block_dimx, block_dimy);
    IndexType *row_offsets_ptr = A.row_offsets.raw();
    IndexType *column_indices_ptr = A.col_indices.raw();
    ValueTypeA *nonzero_values_ptr = A.values.raw();
    ValueTypeA *dia_values_ptr = thrust::raw_pointer_cast(&(A.values[block_dimy * block_dimx * n_nonzeros_part]));
    //Transfer row_offsets to matrix
    thrust::copy(row_offsets_part.begin(), row_offsets_part.end(), A.row_offsets.begin());
    cudaCheckError();
    data_pos += (num_rows + 1) * sizeof(int);
    n_nonzeros_part = 0;
    int row_nnz;

    for (int i = 0; i < partRowVec.size(); i++)
    {
        if (fseek(fin, data_pos + sizeof(int)*row_start_glb[i], SEEK_SET) != 0)
        {
            FatalError("fseek error", AMGX_ERR_IO);
        }

        row_nnz = row_offsets_part[i + 1] - row_offsets_part[i];
        is_read = fread(column_indices_ptr + n_nonzeros_part, sizeof(int), row_nnz, fin);
        n_nonzeros_part += row_nnz;

        if (is_read != row_nnz)
        {
            err = "fread failed reading column_indices, exiting";
            FatalError(err, AMGX_ERR_IO);
        }
    }

    data_pos += num_nz * sizeof(int);
    //temperary array for storing ValueTypeA data
    // double storage for complex
    vector< UpValueTypeA > temp(n_nonzeros_part * block_dimy * block_dimx);
    n_nonzeros_part = 0;

    for (int i = 0; i < partRowVec.size(); i++)
    {
        if (fseek(fin, data_pos + sizeof(UpValueTypeA)*row_start_glb[i] * block_dimy * block_dimx, SEEK_SET) != 0)
        {
            FatalError("fseek error", AMGX_ERR_IO);
        }

        row_nnz = row_offsets_part[i + 1] - row_offsets_part[i];
        //read in data as a ValueTypeA
        is_read = fread(&temp[n_nonzeros_part * block_dimy * block_dimx], sizeof(UpValueTypeA), row_nnz * block_dimy * block_dimx, fin);
        n_nonzeros_part += row_nnz;

        if (is_read != row_nnz * block_dimy * block_dimx)
        {
            err = "fread failed reading off-diagonal values, exiting";
            FatalError(err, AMGX_ERR_IO);
        }
    }

    //copy with cast data to ValueTypeA
    val_copy(temp.data(), nonzero_values_ptr, n_nonzeros_part * block_dimy * block_dimx);
    data_pos += sizeof(UpValueTypeA) * num_nz * block_dimx * block_dimy;

    if (diag)
    {
        temp.resize(n_rows_part * block_dimx * block_dimy);

        //read in diagonal data as a ValueTypeA
        for (int i = 0; i < partRowVec.size(); i++)
        {
            if (fseek(fin, data_pos + sizeof(UpValueTypeA) * partRowVec[i] * block_dimx * block_dimy, SEEK_SET) != 0)
            {
                FatalError("fseek error", AMGX_ERR_IO);
            }

            is_read = fread(&temp[i * block_dimx * block_dimy], sizeof(UpValueTypeA), block_dimx * block_dimy, fin);

            if (is_read != block_dimx * block_dimy)
            {
                err = "fread failed reading diagonal values, exiting";
                FatalError(err, AMGX_ERR_IO);
            }
        }

        //copy with cast data to ValueTypeA
        val_copy(temp.data(), dia_values_ptr, n_rows_part * block_dimx * block_dimy);
        data_pos += sizeof(double) * num_rows * block_dimx * block_dimy;
    }
    else // fill last values item with zeros
    {
        thrust::fill(A.values.begin() + A.get_num_nz() * block_dimy * block_dimx, A.values.end(), types::util<ValueTypeA>::get_zero());
        cudaCheckError();
    }

    //printf("Reading values\n"); fflush(stdout);
    b.resize(n_rows_part * block_dimy);
    b.set_block_dimy(block_dimy);
    b.set_block_dimx(1);
    temp.resize(n_rows_part * block_dimy);

    if (is_rhs)
    {
        for (int i = 0; i < partRowVec.size(); i++)
        {
            if (fseek(fin, data_pos + sizeof(UpValueTypeA) * partRowVec[i] * block_dimy, SEEK_SET) != 0)
            {
                FatalError("fseek error", AMGX_ERR_IO);
            }

            //read in data as a double (doublecomplex)
            is_read = fread(&temp[i * block_dimy], sizeof(UpValueTypeA), block_dimy, fin);

            // if the rhs exists, we must have read the whole thing
            if (is_read != block_dimy)
            {
                err = "fread failed reading rhs, exiting";
                FatalError(err, AMGX_ERR_IO);
            }
        }

        //cast data to ValueTypeB
        val_copy(temp.data(), b.raw(), n_rows_part * block_dimy);
        data_pos += sizeof(UpValueTypeA) * num_rows * block_dimy;
    }
    else
    {
        thrust::fill(b.begin(), b.end(), types::util<ValueTypeB>::get_one());
        cudaCheckError();
    }

    x.resize(0);

    if (is_soln)
    {
        x.resize(n_rows_part * block_dimx);
        x.set_block_dimx(1);
        x.set_block_dimy(block_dimy);
        temp.resize(n_rows_part * block_dimx);

        for (int i = 0; i < partRowVec.size(); i++)
        {
            if (fseek(fin, data_pos + sizeof(UpValueTypeA) * partRowVec[i] * block_dimx, SEEK_SET) != 0)
            {
                FatalError("fseek error", AMGX_ERR_IO);
            }

            //read in data as a double
            is_read = fread(&temp[i * block_dimx], sizeof(UpValueTypeA), block_dimx, fin);

            if (is_read != block_dimx)
            {
                err = "fread failed reading rhs, exiting";
                FatalError(err, AMGX_ERR_IO);
            }
        }

        val_copy(temp.data(), x.raw(), n_rows_part * block_dimx);
    }

    fclose(fin);

    if (rank_rows.size() > 0)
    {
        A.set_is_matrix_read_partitioned(true);
        b.set_is_vector_read_partitioned(true);

        if (x.size() > 0)
        {
            x.set_is_vector_read_partitioned(true);
        }
    }
    else
    {
        delete partRowVec_p;
    }

    return true;
}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class ReadMatrixMarket<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class ReadNVAMGBinary<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}
