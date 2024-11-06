# SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.18)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
  set(libs_all CUDA::cusparse CUDA::cusolver)
  set(dyn_libs amgxsh CUDA::cudart CUDA::cublas)
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
  set(libs_all rt dl CUDA::cusparse CUDA::cusolver)
  set(dyn_libs amgxsh rt dl CUDA::cudart CUDA::cublas)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)

ADD_EXECUTABLE(eigensolver eigensolver.c)

target_link_libraries(eigensolver ${dyn_libs})

IF(WIN32)
    IF(MSVC)
      set_source_files_properties( eigensolver.c PROPERTIES LANGUAGE CXX)
      set_target_properties ( eigensolver PROPERTIES LINKER_LANGUAGE CXX )
    ENDIF(MSVC)
ELSE(WIN32)
  SET_SOURCE_FILES_PROPERTIES( eigensolver.c PROPERTIES COMPILE_FLAGS -std=c99 )
ENDIF(WIN32)

if(MPI_FOUND)
  ADD_EXECUTABLE(eigensolver_mpi eigensolver_mpi.c)

  IF(WIN32)
      IF(MSVC)
        set_source_files_properties( eigensolver_mpi.c PROPERTIES LANGUAGE CXX)
        set_target_properties ( eigensolver_mpi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties ( eigensolver_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(eigensolver_mpi ${MPI_C_LIBRARIES} ${dyn_libs})
      ENDIF(MSVC)
    ELSE(WIN32)
      SET_SOURCE_FILES_PROPERTIES( eigensolver_mpi.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( eigensolver_mpi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( eigensolver_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(eigensolver_mpi amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})
    ENDIF(WIN32) 
    
    if(MPI_COMPILE_FLAGS)
      set_target_properties(eigensolver_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif(MPI_COMPILE_FLAGS)

endif(MPI_FOUND)
