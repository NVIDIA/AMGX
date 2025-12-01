// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <distributed/distributed_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(GeneratedMatrixDistributedIOTest);

void run()
{
    this->randomize( time(NULL) );

    for (int bin = 0; bin < 2; bin++)
    {
        std::string write_format;

        if (bin)
        {
            write_format = "binary";
        }
        else
        {
            write_format = "matrixmarket";
        }

        int partitions = 5 + rand() % 7;
        Matrix_h A;
        std::vector<Matrix_h> An (partitions);
        VVector b, x;
        std::vector<VVector> bn (partitions), xn(partitions);
        std::vector<IVector_h> partitionVec (partitions); //need 'partitions' number of copies, since partitionVec gets overwritten on read

        for (int block_dim = 1; block_dim < 5; block_dim++)
        {
            std::vector<IVector_h> partSize (partitions); // need 0 size on each run if not initialized
            int bsize = block_dim * block_dim;
            generateMatrixRandomStruct<TConfig>::generateExact(A, 1000, true, block_dim, false);
            random_fill(A);
            b.resize(A.get_num_rows()*block_dim);
            random_fill(b);
            x.resize(A.get_num_rows()*block_dim);
            random_fill(x);
            int num_rows = A.get_num_rows();

            for (int part = 0; part < partitions; part ++)
            {
                partitionVec[part].resize(num_rows);
            }

            for (int i = 0; i < num_rows; i++)
            {
                int part_val =  rand() % partitions;

                for (int part = 0; part < partitions; part ++) // each rank has the same partitioning
                {
                    partitionVec[part][i] = part_val;
                }
            }

            MatrixIO<TConfig_h>::writeSystemWithFormat(".temp_matrix.mtx", write_format.c_str(), &A, &b, &x);
            int rows_sum = 0;
            std::vector<std::vector<int> > globalRows(partitions);

            for (int i = 0; i < num_rows; i++)
                for (int part = 0; part < partitions; part ++)
                    if (part == partitionVec[part][i])
                    {
                        globalRows[part].push_back(i);
                    }

            for (int part = 0; part < partitions; part++)
            {
                //printf("Partition %d/%d\n", part, partitions);
                UNITTEST_ASSERT_TRUE(DistributedRead<TConfig_h>::distributedRead(".temp_matrix.mtx", An[part], bn[part], xn[part], 1, part, partitions, partSize[part], partitionVec[part]) == AMGX_OK);

                for (int i = 0; i < An[part].get_num_rows(); i++)
                {
                    rows_sum++;

                    for (int col = 0; col < An[part].row_offsets[i + 1] - An[part].row_offsets[i]; col++)
                        for (int k = 0; k < bsize; k++)
                        {
                            //printf("%f %f ; ", An[part].values[ (An[part].row_offsets[i] + col)*bsize], A.values[(A.row_offsets[globalRows[part][i]] + col)*bsize]);
                            UNITTEST_ASSERT_EQUAL(An[part].values[ (An[part].row_offsets[i] + col)*bsize + k], A.values[(A.row_offsets[globalRows[part][i]] + col)*bsize + k]);
                        }

                    //printf("\n");
                    for (int k = 0; k < bsize; k++)   // test diagonal
                    {
                        UNITTEST_ASSERT_EQUAL(An[part].values[An[part].get_num_nz()*bsize + i * bsize + k], A.values[A.get_num_nz()*bsize + globalRows[part][i]*bsize + k]);
                    }

                    for (int k = 0; k < block_dim; k++)
                    {
                        UNITTEST_ASSERT_EQUAL(bn[part][i * block_dim + k], b[globalRows[part][i]*block_dim + k]);
                        UNITTEST_ASSERT_EQUAL(xn[part][i * block_dim + k], x[globalRows[part][i]*block_dim + k]);
                    }
                }
            }

            UNITTEST_ASSERT_EQUAL(rows_sum, num_rows);
        } // block sizes iter
    } // write format iter
}

DECLARE_UNITTEST_END(GeneratedMatrixIOTest);

#define AMGX_CASE_LINE(CASE) GeneratedMatrixDistributedIOTest <TemplateMode<CASE>::Type>  GeneratedMatrixDistributedIOTest_##CASE;
AMGX_FORALL_BUILDS_HOST(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} //namespace amgx
