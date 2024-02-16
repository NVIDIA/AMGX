// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/coo_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <fstream>
#include <cstdlib>


template <typename index_type>
index_type internal_index(index_type gi, index_type gj, index_type gk, index_type nx, index_type ny, index_type nz, index_type P, index_type Q, index_type R)
{
    int i = gi % nx; // Position in x
    int j = gj % ny; // Position in y
    int k = gk % nz; // Position in z

    int p = gi / nx; // Position in x direction
    int q = gj / ny; // Position in y
    int r = gk / nz; // Position in z

    return (r*P*Q + q*P + p)*nx*ny*nz + (k*nx*ny + j*nx + i);
}

int main(int argc, char **argv)
{
  const char *fname = NULL;
  cusp::coo_matrix<int,double,cusp::host_memory> A;
  // check command line arguments
  for (int i=1; i < argc; i++)
  {
    if (strncmp(argv[i],"-p",100) == 0)
    {
      int nx = atoi(argv[++i]);
      int ny = atoi(argv[++i]);
      int nz = atoi(argv[++i]);

      int P = atoi(argv[++i]);
      int Q = atoi(argv[++i]);
      int R = atoi(argv[++i]);

      int nx_glob = nx*P;
      int ny_glob = ny*Q;
      int nz_glob = nz*R;

      int num_rows = nx_glob*ny_glob*nz_glob;
      int num_nonzeros = num_rows*7; // Ignoring any boundary, 7 nnz per row
      int num_substract = 0;

      num_substract += ny_glob*nz_glob;
      num_substract += ny_glob*nz_glob;
      num_substract += nx_glob*nz_glob;
      num_substract += nx_glob*nz_glob;
      num_substract += nx_glob*ny_glob;
      num_substract += nx_glob*ny_glob;

      num_nonzeros -= num_substract;

      A.resize(num_rows,num_rows,num_nonzeros);

      int count = 0;
      int nz_count = 0;

      for (int r = 0; r < R ; r++)
      {
      for (int q = 0; q < Q ; q++)
      {
      for (int p = 0; p < P ; p++)
      {
        for (int k = 0 ; k < nz ; k++)
        {
        for (int j = 0 ; j < ny ; j++)
        {
        for (int i = 0 ; i < nx ; i++)
        {
           int gi = p*nx + i;
           int gj = q*ny + j;
           int gk = r*nz + k;

           //Diagonal term
           A.row_indices[nz_count] = count;
           A.column_indices[nz_count] = count;
           A.values[nz_count] = 6.;
           nz_count++;

           // Given gi,gj, gk, find p,q,r, i_loc, j_loc, k_loc

           if ((i==0) && (p==0)) {}
            // do nothing 
           else
           {
              A.row_indices[nz_count] = count;
              A.column_indices[nz_count] = internal_index(gi-1,gj,gk,nx,ny,nz,P,Q,R);
              A.values[nz_count] = -1.;
              nz_count++;
           }

           if ((i==nx-1) && (p==P-1)) {}
            // do nothing, no right neighbor
           else
           {
              A.row_indices[nz_count] = count;
              A.column_indices[nz_count] = internal_index(gi+1,gj,gk,nx,ny,nz,P,Q,R);
              A.values[nz_count] = -1.;
              nz_count++;
           }

           if ((j==0) && (q==0)) {}
            // do nothing, no right neighbor
           else
           {
              A.row_indices[nz_count] = count;
              A.column_indices[nz_count] = internal_index(gi,gj-1,gk,nx,ny,nz,P,Q,R);
              A.values[nz_count] = -1.;
              nz_count++;
           }

           if ((j==ny-1) && (q==Q-1)) {}
            // do nothing, no right neighbor
           else
           {
              A.row_indices[nz_count] = count;
              A.column_indices[nz_count] = internal_index(gi,gj+1,gk,nx,ny,nz,P,Q,R);
              A.values[nz_count] = -1.;
              nz_count++;
           }


           if ((k==0) && (r==0)) {}
            // do nothing, no right neighbor
           else
           {
              A.row_indices[nz_count] = count;
              A.column_indices[nz_count] = internal_index(gi,gj,gk-1,nx,ny,nz,P,Q,R);
              A.values[nz_count] = -1.;
              nz_count++;
           }


           if ((k==nz-1) && (r==R-1)) {}
            // do nothing, no right neighbor
           else
           {
              A.row_indices[nz_count] = count;
              A.column_indices[nz_count] = internal_index(gi,gj,gk+1,nx,ny,nz,P,Q,R);
              A.values[nz_count] = -1.;
              nz_count++;
           }

           count++;

        }
        }
        }
      }
      }
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

