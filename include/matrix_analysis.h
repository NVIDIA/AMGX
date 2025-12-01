// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdio>
#include <matrix.h>
#include <error.h>

#include <amgx_types/pod_types.h>

namespace amgx
{


template <class T_Config>
class MatrixAnalysis
{
    public:
        typedef T_Config TConfig;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  mat_value_type;
        typedef typename TConfig::VecPrec  vec_value_type;
        typedef typename TConfig::IndPrec  index_type;

        typedef typename Matrix<T_Config>::value_type ValueTypeA;
        typedef typename types::PODTypes<ValueTypeA>::type PODTypeA;

        typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TemplateConfig<AMGX_host, types::PODTypes<typename Vector_h::value_type>::vec_prec, T_Config::matPrec, T_Config::indPrec> > PODVecHost;
        typedef Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, T_Config::matPrec, T_Config::indPrec> > DevVectorFloat;

        MatrixAnalysis(const Matrix<TConfig> *A_mat = NULL, Vector<TConfig> *b_rhs = NULL, const char *fileName = NULL ) : A(NULL), b(NULL), fout(NULL)
        {
            bind(A_mat, b_rhs, fileName);
        }
        ~MatrixAnalysis() { if (fout && fout != stdout) fclose(fout); }

        void bind(const Matrix<TConfig> *A_mat, Vector<TConfig> *b_rhs, const char *fileName )
        {
            A = A_mat;
            b = b_rhs;
            geo_x = NULL;
            geo_y = NULL;
            geo_z = NULL;

            if (fout && fout != stdout) { fclose(fout); }

            if (fileName) { fout = fopen(fileName, "w"); }

            if (!fout) { fout = stdout; }
        }

        void load_geometry(PODVecHost *geox, PODVecHost *geoy, PODVecHost *geoz )
        {
            geo_x = geox;
            geo_y = geoy;
            geo_z = geoz;
            std::cout << "input size: " << geox->size() << "\t" << geo_x->size() << std::endl;
        }

        void valueDistribution(double minAbs, double maxAbs)
        {
            int bs = A->get_block_size();
            std::vector<double> minAbsVec(bs, minAbs), maxAbsVec(bs, maxAbs);
            valueDistribution( &minAbsVec[0], &maxAbsVec[0] );
        }

        void valueDistribution(double *minAbs, double *maxAbs);

        void checkSymmetry(bool &structuralSymmetric, bool &symmetric, bool &verbose);
        void checkDiagDominate();
        void draw_matrix_connection();

        bool check_Z_matrix();

        float aggregatesQuality(typename Matrix<T_Config>::IVector &aggregates, DevVectorFloat &edge_weights);
        void aggregatesQuality2(const typename Matrix<T_Config>::IVector &aggregates, int num_aggregates, const Matrix<T_Config> &Aorig);
        void visualizeAggregates(typename Matrix<T_Config>::IVector &aggregates);

    private:
        const Matrix<TConfig> *A;
        Vector<TConfig> *b;
        PODVecHost *geo_x;
        PODVecHost *geo_y;
        PODVecHost *geo_z;
        FILE *fout;
};


} //end namespace amgx
