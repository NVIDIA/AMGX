// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/coo_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <fstream>
#include <cstdlib>

int main(int argc, char **argv)
{
  cusp::coo_matrix<int,double,cusp::host_memory> A;
  const char *fname = NULL;
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

