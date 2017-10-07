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
