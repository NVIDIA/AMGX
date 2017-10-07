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
 
#include <cusp/coo_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <fstream>
#include <cstdlib>

int main(int argc, char **argv)
{
  cusp::coo_matrix<int,double,cusp::host_memory> A;
  char *fname = NULL;
  // check command line arguments
  for (int i=1; i < argc; i++)
  {
    if (strncmp(argv[i],"-p",100) == 0)
    {
      int points = atoi(argv[++i]);
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z;

      switch(points)
      {
        case 5:
          cusp::gallery::poisson5pt(A,x,y);
          break;
        case 7:
          z = atoi(argv[++i]);
          cusp::gallery::poisson7pt(A,x,y,z);
          break;
        case 9:
          cusp::gallery::poisson9pt(A,x,y);
          break;
        case 27:
          z = atoi(argv[++i]);
          cusp::gallery::poisson27pt(A,x,y,z);
          break;
        default:
          printf("Error invalid number of poisson points specified, valid numbers are 5, 7, 9, 27\n");
          exit(0);
      }  
    }
    else if (strncmp(argv[i],"-o",100) == 0)
    {
      i++;
      fname = argv[i];
    }
  }

  // output
  if (fname == NULL)
  {
    fname = "output.mtx";
  }

  cusp::io::write_matrix_market_file(A,fname);
}

