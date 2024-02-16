// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amgx_c.h"
#include "amgx_eig_c.h"

int findParamIndex(const char** argv, int argc, const char* parm)
{

  int count = 0;
  int index = -1;
  for (int i = 0; i < argc; i++)
  {
    if (strncmp(argv[i], parm, 100)==0)
    {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1)
  {
    return index;
  }
  else
  {
    printf("Error, parameter %s has been specified more than once, exiting\n",parm);
    exit(1);
  }
  return -1;
}

void errAndExit(const char* err)
{
  printf("%s\n", err);
  exit(1);
//  printUsageAndExit();
}

void printUsageAndExit()
{
  printf("%s", "Usage: ./eigeneigensolver [-help] [-mode [hDDI | hDFI | hFFI | dDDI | dDFI | dFFI] [-m mformat mfile] [-c config_file]\n");
  printf("%s", "     -help display the command options\n");
  printf("%s", "     -m mfile: reads matrix stored in file mfile\n");
  printf("%s", "     -mode : which of the eigensolver modes to use\n");
  printf("%s", "     -c set the amg eigensolver options from the configuration file\n");

  exit(0);
}

void print_callback(const char *msg, int length)
{
    // custom string handler
    printf("%s", msg);
}

int main(int argc, const char **argv)
{
  if (argc == 1)
  {
    printUsageAndExit();
  }

// library init
  int pidx = 0;
  if ((pidx = findParamIndex(argv, argc, "--version")) != -1)
  {
    char *ver, *date, *time;

	AMGX_get_build_info_strings(&ver, &date, &time);
	printf("AMGX build version: %s\nBuild date and time: %s %s\n", ver, date, time);
	exit(0);
  }

  AMGX_SAFE_CALL(AMGX_solver_register_print_callback(&print_callback));

  AMGX_SAFE_CALL(AMGX_initialize());

  AMGX_SAFE_CALL(AMGX_install_signal_handler());

// setup eigensolver
  AMGX_config_handle cfg;
  AMGX_eigensolver_handle eigensolver;

  AMGX_Mode mode;
  if ((pidx = findParamIndex(argv, argc, "-mode")) != -1)
  {
      if (strncmp(argv[pidx + 1], "hDDI", 100) == 0)
      {
          mode = AMGX_mode_hDDI;
      }
      else if (strncmp(argv[pidx + 1], "hDFI", 100) == 0)
      {
          mode = AMGX_mode_hDFI;
      }
      else if (strncmp(argv[pidx + 1], "hFFI", 100) == 0)
      {
          mode = AMGX_mode_hFFI;
      }
      else if (strncmp(argv[pidx + 1], "dDDI", 100) == 0)
      {
          mode = AMGX_mode_dDDI;
      }
      else if (strncmp(argv[pidx + 1], "dDFI", 100) == 0)
      {
          mode = AMGX_mode_dDFI;
      }
      else if (strncmp(argv[pidx + 1], "dFFI", 100) == 0)
      {
          mode = AMGX_mode_dFFI;
      }
      else if (strncmp(argv[pidx + 1], "hCCI", 100) == 0)
      {
          mode = AMGX_mode_hZZI;
      }
      else if (strncmp(argv[pidx + 1], "hZCI", 100) == 0)
      {
          mode = AMGX_mode_hZCI;
      }
      else if (strncmp(argv[pidx + 1], "hZZI", 100) == 0)
      {
          mode = AMGX_mode_hZZI;
      }
      else if (strncmp(argv[pidx + 1], "dCCI", 100) == 0)
      {
          mode = AMGX_mode_dCCI;
      }
      else if (strncmp(argv[pidx + 1], "dZCI", 100) == 0)
      {
          mode = AMGX_mode_dZCI;
      }
      else if (strncmp(argv[pidx + 1], "dZZI", 100) == 0)
      {
          mode = AMGX_mode_dZZI;
      }
      else
      {
          errAndExit("ERROR: invalid mode");
      }
  }
  else
  {
      printf("Warning: No mode specified, using dDDI by default.\n");
      mode = AMGX_mode_dDDI;
  }

  int pidy = findParamIndex(argv,argc,"-c");
  if (pidy != -1)
  {
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, argv[pidy+1]));
  }
  else
  {
    // default configuration?
    errAndExit("No eigensolver configuration is specified!");
  }

  // using simple create resources
  AMGX_resources_handle rsrc = NULL;
  AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

  AMGX_SAFE_CALL(AMGX_eigensolver_create(&eigensolver, rsrc, mode, cfg));

// import data
  AMGX_matrix_handle A;
  AMGX_vector_handle x, b, soln;
  AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&soln, rsrc, mode));

  if ((pidx = findParamIndex(argv, argc, "-m")) != -1)
  {
    AMGX_SAFE_CALL(AMGX_read_system(A, b, soln, argv[pidx + 1]));
    int n = 0;
    int xsize_x = 0, xsize_y = 0;
    AMGX_SAFE_CALL(AMGX_matrix_get_size(A, &n, &xsize_x, &xsize_y));
    AMGX_SAFE_CALL(AMGX_vector_set_random(x, n));
  }
  else
  {
    errAndExit("No problem specified for eigensolver!");
  }
  // write - uncomment to enable
  //AMGX_SAFE_CALL(AMGX_write_system(A, b, NULL, "output.system"));
  
  // setup
  AMGX_SAFE_CALL(AMGX_eigensolver_setup(eigensolver, A));
  // PageRank setup
  AMGX_SAFE_CALL(AMGX_eigensolver_pagerank_setup(eigensolver, b));
  // solve
  AMGX_SAFE_CALL(AMGX_eigensolver_solve(eigensolver, x));

  // prints all residual, uncomment to enable
  /*int iters = 0;
  int t_n, t_dimx, t_dimy;
  AMGX_SAFE_CALL(AMGX_matrix_get_size(A, &t_n, &t_dimx, &t_dimy));
  AMGX_SAFE_CALL(AMGX_eigensolver_get_iterations_number(eigensolver, &iters));

    for (int i = 0; i < iters; i++)
    {
      printf("Residual from iteration %d = ", i);
      for (int j = 0; j < t_dimy; j++)
      {
        double res;
        AMGX_eigensolver_get_iteration_residual(eigensolver, i, j, &res);
        printf("%f ", (float)(res));
      }
      printf("\n");
    }
  */


  AMGX_SAFE_CALL(AMGX_vector_destroy(b));
  AMGX_SAFE_CALL(AMGX_vector_destroy(x));
  AMGX_SAFE_CALL(AMGX_vector_destroy(soln));
  AMGX_SAFE_CALL(AMGX_eigensolver_destroy(eigensolver));
  AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
  AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

  AMGX_SAFE_CALL(AMGX_resources_destroy( rsrc ));

  AMGX_SAFE_CALL(AMGX_finalize());

  return 0;
}
