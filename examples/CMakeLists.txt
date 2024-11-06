# SPDX-FileCopyrightText: 2011-2024 NVIDIA CORPORATION. All Rights Reserved.
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
add_executable(eigensolver eigensolver.c)

set(AMGX_INCLUDES ${THRUST_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories( amgx_capi PUBLIC ${AMGX_INCLUDES})
target_link_libraries ( amgx_capi ${dyn_libs} )

set(AMGX_INCLUDES ${THRUST_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories( eigensolver PUBLIC ${AMGX_INCLUDES})
target_link_libraries ( eigensolver ${dyn_libs} )

IF(WIN32)
    IF(MSVC)
      set_source_files_properties( amgx_capi.c PROPERTIES LANGUAGE CXX)
      set_target_properties ( amgx_capi PROPERTIES LINKER_LANGUAGE CXX )

      set_source_files_properties( eigensolver.c PROPERTIES LANGUAGE CXX)
      set_target_properties ( eigensolver PROPERTIES LINKER_LANGUAGE CXX )
    ENDIF(MSVC)
ELSE(WIN32)
  set_source_filES_PROPERTIES( amgx_capi.c PROPERTIES COMPILE_FLAGS -std=c99 )
  set_source_filES_PROPERTIES( eigensolver.c PROPERTIES COMPILE_FLAGS -std=c99 )
ENDIF(WIN32)

add_executable(generate_poisson generate_poisson.cu)
target_compile_options(generate_poisson PUBLIC "-Xcompiler=-I$ENV{METIS_INSTALL_PATH}/include")
target_link_libraries(generate_poisson amgx ${libs_all} )
set_target_properties(generate_poisson PROPERTIES LINK_FLAGS "")
target_include_directories(generate_poisson PUBLIC ${AMGX_INCLUDES})
set_target_properties(generate_poisson PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

add_executable(generate_poisson7_dist_renum generate_poisson7_dist_renum.cu)
target_compile_options(generate_poisson7_dist_renum PUBLIC "-Xcompiler=-I$ENV{METIS_INSTALL_PATH}/include")
target_link_libraries(generate_poisson7_dist_renum amgx ${libs_all} )
set_target_properties(generate_poisson7_dist_renum PROPERTIES LINK_FLAGS "")
target_include_directories(generate_poisson7_dist_renum PUBLIC ${AMGX_INCLUDES})
set_target_properties(generate_poisson7_dist_renum PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

if(MPI_FOUND)
    add_executable(amgx_mpi_capi amgx_mpi_capi.c )
    add_executable(amgx_mpi_capi_agg amgx_mpi_capi_agg.c )
    add_executable(amgx_mpi_capi_cla amgx_mpi_capi_cla.c )
    add_executable(amgx_mpi_poisson7 amgx_mpi_poisson7.c )
    add_executable(eigensolver_mpi eigensolver_mpi.c )
    IF(WIN32)
      IF(MSVC)
        set_source_files_properties( amgx_mpi_capi.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties(amgx_mpi_capi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(amgx_mpi_capi_agg.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties(amgx_mpi_capi_agg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi_agg ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(amgx_mpi_capi_cla.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties(amgx_mpi_capi_cla PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_capi_cla ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(amgx_mpi_poisson7.c PROPERTIES LANGUAGE CXX)
        set_target_properties(amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties(amgx_mpi_poisson7 PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(amgx_mpi_poisson7  ${MPI_C_LIBRARIES} ${dyn_libs})

        set_source_files_properties(eigensolver_mpi.c PROPERTIES LANGUAGE CXX)
        set_target_properties(eigensolver_mpi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS}" LINKER_LANGUAGE CXX )
        set_target_properties(eigensolver_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
        target_link_libraries(eigensolver_mpi  ${MPI_C_LIBRARIES} ${dyn_libs})
      ENDIF(MSVC)
    ELSE(WIN32)

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_capi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi_agg.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_capi_agg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi_agg amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_capi_cla.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_capi_cla PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries(amgx_mpi_capi_cla amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( amgx_mpi_poisson7.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( amgx_mpi_poisson7 PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries( amgx_mpi_poisson7 amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})

      SET_SOURCE_FILES_PROPERTIES( eigensolver_mpi.c PROPERTIES COMPILE_FLAGS -std=c99 )
      set_target_properties ( eigensolver_mpi PROPERTIES COMPILE_FLAGS "${OpenMP_C_FLAGS} -pthread" )
      set_target_properties ( eigensolver_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
      target_link_libraries( eigensolver_mpi amgxsh ${omp_lib} ${MPI_C_LIBRARIES} ${dyn_libs})
    ENDIF(WIN32) 
    
    if(MPI_COMPILE_FLAGS)
      set_target_properties(amgx_mpi_capi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_capi_agg PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_capi_cla PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(amgx_mpi_poisson7 PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
      set_target_properties(eigensolver_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif(MPI_COMPILE_FLAGS)

    target_include_directories(amgx_mpi_capi PUBLIC ${AMGX_INCLUDES})
    target_include_directories(amgx_mpi_capi_agg PUBLIC ${AMGX_INCLUDES})
    target_include_directories(amgx_mpi_capi_cla PUBLIC ${AMGX_INCLUDES})
    target_include_directories(amgx_mpi_poisson7 PUBLIC ${AMGX_INCLUDES})

    set_target_properties(amgx_mpi_capi PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
    set_target_properties(amgx_mpi_capi_agg PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
    set_target_properties(amgx_mpi_capi_cla PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
    set_target_properties(amgx_mpi_poisson7 PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

endif(MPI_FOUND)

install(TARGETS amgx_capi eigensolver DESTINATION "lib/examples")
if(MPI_FOUND)
  install(TARGETS amgx_mpi_capi amgx_mpi_capi_agg amgx_mpi_capi_cla amgx_mpi_poisson7 eigensolver_mpi DESTINATION "lib/examples")
endif(MPI_FOUND)

#copy configs to build directory
add_custom_target(copy_configs ALL)
add_custom_command(TARGET copy_configs COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/configs" "${CMAKE_BINARY_DIR}/configs")
add_custom_target(copy_eigen_configs ALL)
add_custom_command(TARGET copy_eigen_configs COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs" "${CMAKE_BINARY_DIR}/eigen_configs")

install(FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/AGGREGATION_DILU.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/AGGREGATION_GS.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/AGGREGATION_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/AGGREGATION_MULTI_PAIRWISE.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/CG_DILU.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/CLASSICAL_CG_CYCLE.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/CLASSICAL_CGF_CYCLE.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/CLASSICAL_F_CYCLE.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/CLASSICAL_V_CYCLE.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/CLASSICAL_W_CYCLE.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/FGMRES_AGGREGATION_DILU.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/FGMRES_AGGREGATION_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/FGMRES_CLASSICAL_AGGRESSIVE_HMIS.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/FGMRES_CLASSICAL_AGGRESSIVE_PMIS.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/FGMRES_NOPREC.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/GMRES_AMG_D2.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/IDR_DILU.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/IDRMSYNC_DILU.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PBICGSTAB_AGGREGATION_W_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PBICGSTAB_CLASSICAL_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PBICGSTAB_NOPREC.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCG_AGGREGATION_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCG_CLASSICAL_F_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCG_CLASSICAL_V_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCG_CLASSICAL_W_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCG_DILU.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCG_NOPREC.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCGF_CLASSICAL_F_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCGF_CLASSICAL_V_JACOBI.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/PCGF_CLASSICAL_W_JACOBI.json"
    DESTINATION "lib/configs")

install(FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/ARNOLDI"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/INVERSE_FGMRES"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/JACOBI_DAVIDSON"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/LANCZOS"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/LOBPCG"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/PAGERANK"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/POWER_ITERATION"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_configs/SUBSPACE_ITERATION"
    DESTINATION "lib/eigen_configs")

