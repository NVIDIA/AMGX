# SPDX-FileCopyrightText: 2011-2025 NVIDIA CORPORATION. All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.18)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
  set(libs_all CUDA::cusparse CUDA::cusolver)
  set(dyn_libs amgxsh CUDA::cudart_static CUDA::cublas)
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
  set(libs_all CUDA::cusparse CUDA::cusolver rt dl)
  set(dyn_libs amgxsh rt dl CUDA::cudart_static CUDA::cublas)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)

add_executable(amgx_capi amgx_capi.c)

set(AMGX_INCLUDES ${THRUST_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories( amgx_capi PUBLIC ${AMGX_INCLUDES})
target_link_libraries ( amgx_capi ${dyn_libs} )

IF(WIN32)
    IF(MSVC)
      set_source_files_properties( amgx_capi.c PROPERTIES LANGUAGE CXX)
      set_target_properties ( amgx_capi PROPERTIES LINKER_LANGUAGE CXX )
    ENDIF(MSVC)
ELSE(WIN32)
  set_source_filES_PROPERTIES( amgx_capi.c PROPERTIES COMPILE_FLAGS -std=c99 )
ENDIF(WIN32)

add_executable(generate_poisson generate_poisson.cu)
target_compile_options(generate_poisson PUBLIC "-Xcompiler=-I$ENV{METIS_INSTALL_PATH}/include")
target_link_libraries(generate_poisson amgx ${libs_all} )
set_target_properties(generate_poisson PROPERTIES LINK_FLAGS "")
target_include_directories(generate_poisson PUBLIC ${AMGX_INCLUDES})
# CMAKE_CUDA_ARCHITECTURES automatically inherits to all targets

add_executable(generate_poisson7_dist_renum generate_poisson7_dist_renum.cu)
target_compile_options(generate_poisson7_dist_renum PUBLIC "-Xcompiler=-I$ENV{METIS_INSTALL_PATH}/include")
target_link_libraries(generate_poisson7_dist_renum amgx ${libs_all} )
set_target_properties(generate_poisson7_dist_renum PROPERTIES LINK_FLAGS "")
target_include_directories(generate_poisson7_dist_renum PUBLIC ${AMGX_INCLUDES})
# CMAKE_CUDA_ARCHITECTURES automatically inherits to all targets

if(MPI_FOUND)
    add_executable(amgx_mpi_capi amgx_mpi_capi.c )
    add_executable(amgx_mpi_capi_agg amgx_mpi_capi_agg.c )
    add_executable(amgx_mpi_capi_cla amgx_mpi_capi_cla.c )
    add_executable(amgx_mpi_poisson7 amgx_mpi_poisson7.c )
    IF(WIN32)
      IF(MSVC)
        set_source_files_properties( amgx_mpi_capi.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_capi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(amgx_mpi_capi_agg.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_capi_agg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi_agg ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(amgx_mpi_capi_cla.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_capi_cla PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi_cla ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(amgx_mpi_poisson7.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_poisson7 PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_poisson7  ${MPI_C_LIBRARIES} ${dyn_libs})
      ENDIF(MSVC)
    ELSE(WIN32)

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi amgxsh ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi_agg.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi_agg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi_agg amgxsh ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi_cla.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi_cla PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi_cla amgxsh ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_poisson7.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_poisson7 PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries( amgx_mpi_poisson7 amgxsh ${MPI_C_LIBRARIES} ${dyn_libs})

    ENDIF(WIN32) 
    
    if(MPI_COMPILE_FLAGS)
      set_target_properties(amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif(MPI_COMPILE_FLAGS)

    target_include_directories(amgx_mpi_capi PUBLIC ${AMGX_INCLUDES})
    target_include_directories(amgx_mpi_capi_agg PUBLIC ${AMGX_INCLUDES})
    target_include_directories(amgx_mpi_capi_cla PUBLIC ${AMGX_INCLUDES})
    target_include_directories(amgx_mpi_poisson7 PUBLIC ${AMGX_INCLUDES})

    # CMAKE_CUDA_ARCHITECTURES automatically inherits to all targets

endif(MPI_FOUND)

install(TARGETS amgx_capi DESTINATION "lib/examples")

if(MPI_FOUND)
  install(TARGETS amgx_mpi_capi amgx_mpi_capi_agg amgx_mpi_capi_cla amgx_mpi_poisson7 DESTINATION "lib/examples")
endif(MPI_FOUND)

add_subdirectory(eigen_examples)

