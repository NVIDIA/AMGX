# Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

#
# Makefile for building spmv example (single GPU and multi GPU) using AMGX C++ code directly 

# System paths
CC = /usr/bin/cc
CPP = /usr/bin/c++
CUDA_PATH = /usr/local/cuda
MPI_PATH = /opt/openmpi-1.10.2
NVCC = $(CUDA_PATH)/bin/nvcc
ARCH = --gpu-architecture=compute_35 --gpu-code=compute_35,sm_35

NVCC_FLAGS = -std=c++11 $(ARCH) -Xcompiler ,\"-static-libgcc\",\"-fopenmp\",\"-DRAPIDJSON_DEFINED\",\"-O3\",\"-DNDEBUG\" -Xcompiler=-rdynamic -Xcompiler=-fPIC -Xcompiler=-fvisibility=default -DCUDA90_COMPILER -O3 -DNDEBUG --Werror cross-execution-space-call -DNVCC
NVCC_D_FLAGS = -Xcompiler ,\"-DAMGX_WITH_MPI\"
NVCC_L_FLAGS = $(ARCH)
MPI_L_FLAGS = -lmpi -L$(MPI_PATH)/lib
EXTRA_LIBS = -lcublas -lcusparse -lcusolver -Xlinker=-rpath=$(CUDA_PATH)/lib64
AMGX_ROOT = ../..
AMGX_INCLUDE = -I$(AMGX_ROOT)/../../thrust -I$(AMGX_ROOT)/include -I$(CUDA_PATH)/include -I$(AMGX_ROOT)/external/rapidjson/include -I$(MPI_PATH)/include

BASE_CU_FILES := ../../src/misc.cu ../../src/device_properties.cu ../../src/logger.cu ../../src/auxdata.cu ../../src/amgx_cusparse.cu ../../src/amgx_cublas.cu ../../src/amg_config.cu ../../src/global_thread_handle.cu ../../src/error.cu ../../src/thread_manager.cu ../../src/resources.cu ../../src/matrix.cu ../../src/multiply.cu ../../src/hash_workspace.cu ../../src/csr_multiply.cu ../../src/csr_multiply_detail.cu ../../src/matrix_coloring/matrix_coloring.cu ../../src/distributed/distributed_manager.cu ../../src/distributed/distributed_arranger.cu ../../src/distributed/comms_visitors1.cu ../../src/distributed/comms_visitors2.cu ../../src/distributed/comms_visitors3.cu ../../src/distributed/comms_visitors4.cu ../../src/distributed/comms_mpi_hostbuffer_stream.cu ../../src/distributed/comms_mpi_gpudirect.cu ../../src/distributed/distributed_comms.cu

OBJDIR_S_GPU := obj_s
OBJDIR_M_GPU := obj_m

OBJS_S := $(BASE_CU_FILES:%.cu=%_s.o)
OBJS_M := $(BASE_CU_FILES:%.cu=%_m.o)


%_s.o: %.cu
	mkdir -p $(OBJDIR_S_GPU); $(NVCC) $(NVCC_FLAGS) $(AMGX_INCLUDE) -c $< -o $(OBJDIR_S_GPU)/$(notdir $@)

%_m.o: %.cu
	mkdir -p $(OBJDIR_M_GPU); $(NVCC) $(NVCC_FLAGS) $(NVCC_D_FLAGS) $(AMGX_INCLUDE) -c $< -o $(OBJDIR_M_GPU)/$(notdir $@)

example: $(OBJS_S)
	$(NVCC) $(NVCC_FLAGS) $(AMGX_INCLUDE) -c amgx_spmv_internal.cu -o $(OBJDIR_S_GPU)/amgx_spmv_internal.o
	$(NVCC) $(NVCC_L_FLAGS) $(EXTRA_LIBS) $(OBJDIR_S_GPU)/*.o -o amgx_spmv

example_distributed: $(OBJS_M)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_D_FLAGS) $(AMGX_INCLUDE) -c amgx_spmv_distributed_internal.cu -o $(OBJDIR_M_GPU)/amgx_spmv_distributed_internal.o
	$(NVCC) $(NVCC_L_FLAGS) $(MPI_L_FLAGS) $(EXTRA_LIBS) $(OBJDIR_M_GPU)/*.o -o amgx_spmv_distributed
