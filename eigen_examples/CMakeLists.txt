# Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

IF (WIN32)
cmake_minimum_required (VERSION 2.8.8)
ELSE (WIN32)
cmake_minimum_required (VERSION 2.8.0)
ENDIF (WIN32)

CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../base/include)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../core/include)

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${cusparse_library} )
  set(dyn_libs amgxsh cudart.lib)
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${cusparse_library} rt dl)
  set(dyn_libs amgxsh rt dl cudart)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl)

CUDA_ADD_EXECUTABLE(eigensolver eigensolver.cu OPTIONS -DAMGX_API_NO_IMPORTS -DDEVICE)
target_link_libraries(eigensolver amgx ${libs_all})

if(${AMGX_PUBLIC_RELEASE} MATCHES "FALSE")
  if(MPI_FOUND)
    CUDA_ADD_EXECUTABLE(eigensolver_mpi eigensolver_mpi.cu OPTIONS -DAMGX_API_NO_IMPORTS -DDEVICE -Xcompiler=${OpenMP_C_FLAGS} -Xcompiler=-I/opt/mpi/include -Xcompiler=-pthread)
    target_link_libraries(eigensolver_mpi amgx ${libs_all} ${omp_lib} ${MPI_LIBRARIES})
    if(MPI_COMPILE_FLAGS)
    set_target_properties(eigensolver_mpi PROPERTIES
      COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif(MPI_COMPILE_FLAGS)

    if(MPI_LINK_FLAGS)
    set_target_properties(eigensolver_mpi PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif(MPI_LINK_FLAGS)
  endif(MPI_FOUND)
endif(${AMGX_PUBLIC_RELEASE} MATCHES "FALSE")
