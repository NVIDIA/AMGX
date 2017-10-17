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

CUDA_INCLUDE_DIRECTORIES(${THRUST_DIR})
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../base/include)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../core/include)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../base/include" "${CUDA_TOOLKIT_ROOT_DIR}/include")
if(WIN32)
  if(MSVC)
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
  endif(MSVC)
else(WIN32)
  link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
endif(WIN32)

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${cusparse_library} ${cusolver_library})
  set(dyn_libs amgxsh cudart.lib)
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${cusparse_library} ${cusolver_library} rt dl)
  set(dyn_libs amgxsh rt dl cudart)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl)

ADD_EXECUTABLE(amgx_capi amgx_capi.c)

target_link_libraries ( amgx_capi ${dyn_libs} )

IF(WIN32)
    IF(MSVC)
      set_source_files_properties( amgx_capi.c PROPERTIES LANGUAGE CXX)
      set_target_properties ( amgx_capi PROPERTIES LINKER_LANGUAGE CXX )
    ENDIF(MSVC)
ELSE(WIN32)
  SET_SOURCE_FILES_PROPERTIES( amgx_capi.c PROPERTIES COMPILE_FLAGS -std=c99 )
ENDIF(WIN32)


CUDA_ADD_EXECUTABLE(generate_poisson generate_poisson.cu OPTIONS "-Xcompiler=-I$ENV{METIS_INSTALL_PATH}/include")
target_link_libraries(generate_poisson amgx ${libs_all} )
set_target_properties(generate_poisson PROPERTIES LINK_FLAGS "")

CUDA_ADD_EXECUTABLE(generate_poisson7_dist_renum generate_poisson7_dist_renum.cu OPTIONS "-Xcompiler=-I$ENV{METIS_INSTALL_PATH}/include")
target_link_libraries(generate_poisson7_dist_renum amgx ${libs_all} )
set_target_properties(generate_poisson7_dist_renum PROPERTIES LINK_FLAGS "")

if(MPI_FOUND)
    ADD_EXECUTABLE(amgx_mpi_capi amgx_mpi_capi.c )
    ADD_EXECUTABLE(amgx_mpi_capi_agg amgx_mpi_capi_agg.c )
    ADD_EXECUTABLE(amgx_mpi_capi_cla amgx_mpi_capi_cla.c )
    ADD_EXECUTABLE(amgx_mpi_poisson7 amgx_mpi_poisson7.c )
#    ADD_EXECUTABLE(amgx_spmv_example amgx_spmv_test.c )
    IF(WIN32)
      IF(MSVC)
        set_source_files_properties( amgx_mpi_capi.c PROPERTIES LANGUAGE CXX)
        set_target_properties ( amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties ( amgx_mpi_capi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties( amgx_mpi_capi_agg.c PROPERTIES LANGUAGE CXX)
        set_target_properties ( amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties ( amgx_mpi_capi_agg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi_agg ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties( amgx_mpi_capi_cla.c PROPERTIES LANGUAGE CXX)
        set_target_properties ( amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties ( amgx_mpi_capi_cla PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi_cla ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties( amgx_mpi_poisson7.c PROPERTIES LANGUAGE CXX)
        set_target_properties ( amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties ( amgx_mpi_poisson7 PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries( amgx_mpi_poisson7  ${MPI_C_LIBRARIES} ${dyn_libs})

        #set_source_files_properties( amgx_spmv_example.c PROPERTIES LANGUAGE CXX)
        #set_target_properties ( amgx_spmv_example PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        #set_target_properties ( amgx_spmv_example PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        #target_link_libraries( amgx_spmv_example  ${MPI_C_LIBRARIES} ${dyn_libs})

      ENDIF(MSVC)
    ELSE(WIN32)
      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_capi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi cudart amgxsh ${omp_lib} ${MPI_C_LIBRARIES})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi_agg.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_capi_agg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi_agg cudart amgxsh ${omp_lib} ${MPI_C_LIBRARIES})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi_cla.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_capi_cla PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi_cla cudart amgxsh ${omp_lib} ${MPI_C_LIBRARIES})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_poisson7.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_poisson7 PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries( amgx_mpi_poisson7 cudart amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})

#      SET_SOURCE_FILES_PROPERTIES( amgx_spmv_test.c PROPERTIES COMPILE_FLAGS -std=c99 )
#      set_target_properties ( amgx_spmv_example PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
#      set_target_properties ( amgx_spmv_example PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#      target_link_libraries( amgx_spmv_example cudart amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})

    ENDIF(WIN32) 
    
    if(MPI_COMPILE_FLAGS)
      set_target_properties(amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif(MPI_COMPILE_FLAGS)

endif(MPI_FOUND)

install(TARGETS amgx_capi DESTINATION "lib/examples")
if(MPI_FOUND)
  install(TARGETS amgx_mpi_capi amgx_mpi_capi_agg amgx_mpi_capi_cla amgx_mpi_poisson7 DESTINATION "lib/examples")
endif(MPI_FOUND)

