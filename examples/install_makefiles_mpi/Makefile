# Copyright (c) 2011-2016 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# CUDA Toolkit location
INC_D = -I/usr/local/cuda/include
LIB_D = -L/usr/local/cuda/lib64
LIB_L = -lcusparse -lcublas -lcudart -ldl

# Examples
amgx_capi:
	gcc -O2 -std=c99 amgx_capi.c -c $(INC_D)
	g++ -O2 amgx_capi.o -o amgx_capi $(LIB_D) $(LIB_L) -L../lib -lamgxsh -Wl,-rpath=../lib

amgx_mpi_capi:
	mpicc -O2 -std=c99 amgx_mpi_capi.c -c $(INC_D) 
	mpic++ -O2 amgx_mpi_capi.o -o amgx_mpi_capi $(LIB_D) $(LIB_L) -L../lib -lamgxsh -lmpi -Wl,-rpath=../lib

amgx_capi_dynamic:
	gcc -O2 -std=c99 amgx_capi.c -c -o amgx_capi_dynamic.o $(INC_D) -DAMGX_DYNAMIC_LOADING
	g++ -O2 amgx_capi_dynamic.o -o amgx_capi_dynamic $(LIB_D) $(LIB_L) -Wl,-rpath=../lib

amgx_mpi_capi_dynamic:
	mpicc -O2 -std=c99 amgx_mpi_capi.c -c -o amgx_mpi_capi_dynamic.o $(INC_D) -DAMGX_DYNAMIC_LOADING
	mpic++ -O2 amgx_mpi_capi_dynamic.o -o amgx_mpi_capi_dynamic $(LIB_D) $(LIB_L) -lmpi -Wl,-rpath=../lib

amgx_mpi_capi_agg:
	mpicc -O2 -std=c99 amgx_mpi_capi_agg.c -c $(INC_D) 
	mpic++ -O2 amgx_mpi_capi_agg.o -o amgx_mpi_capi_agg $(LIB_D) $(LIB_L) -L../lib -lamgxsh -lmpi -Wl,-rpath=../lib

amgx_mpi_capi_cla:
	mpicc -O2 -std=c99 amgx_mpi_capi_cla.c -c $(INC_D) 
	mpic++ -O2 amgx_mpi_capi_cla.o -o amgx_mpi_capi_cla $(LIB_D) $(LIB_L) -L../lib -lamgxsh -lmpi -Wl,-rpath=../lib

# All
all: amgx_capi amgx_mpi_capi amgx_capi_dynamic amgx_mpi_capi_dynamic amgx_mpi_capi_agg amgx_mpi_capi_cla

# Clean
clean:
	rm -f amgx_capi
	rm -f amgx_mpi_capi
	rm -f amgx_capi_dynamic
	rm -f amgx_mpi_capi_dynamic
	rm -f amgx_mpi_capi_cla
	rm -f amgx_mpi_capi_agg
	rm -f *.o
