# SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.18)

#select all sources
FILE(GLOB_RECURSE SRCS "*.cu")
list(FILTER SRCS EXCLUDE REGEX "/src/tests/")
list(FILTER SRCS EXCLUDE REGEX "/utest.cu")

target_sources(amgx_libs PRIVATE ${SRCS})

target_include_directories(amgx_libs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(AMGX_INCLUDES ../include)

set(tests_all ${tests_all} testframework.cu test_utils.cu unit_test.cu)

add_library(amgx_tests_libs OBJECT ${tests_all})

target_include_directories(amgx_tests_libs PRIVATE ${AMGX_INCLUDES}) 

add_library(amgx_tests_library STATIC $<TARGET_OBJECTS:amgx_tests_libs>)

target_link_libraries(amgx_tests_libs Thrust)

add_dependencies(amgx_tests_library amgx)

target_link_libraries(amgx_tests_library amgx)

add_executable(amgx_tests_launcher utest.cu $<TARGET_OBJECTS:amgx_tests_libs>)

# CMAKE_CUDA_ARCHITECTURES automatically inherits to all targets
# No need to set per-target properties

target_compile_options(amgx_tests_launcher PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS} >)
target_compile_options(amgx_tests_libs PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS} >)
target_compile_options(amgx_tests_library PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS} >)

target_include_directories(amgx_tests_launcher PRIVATE ${AMGX_INCLUDES}) 

# create executables for host and device, scalar and block version

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl)
    set(libs_all CUDA::cusparse CUDA::cublas)
ELSE()
    set(libs_all CUDA::cusparse CUDA::cublas rt dl)
ENDIF()

target_link_libraries(amgx_tests_launcher amgx_tests_library amgxsh ${libs_all})

if(MPI_FOUND)
    target_link_libraries(amgx_tests_launcher MPI::MPI_CXX)
endif(MPI_FOUND)

if(NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")
  install(TARGETS amgx_tests_launcher DESTINATION "lib/tests")
endif(NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")

#copy configs to build directory
add_custom_target(copy_configs ALL)
add_custom_command(TARGET copy_configs COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/configs" "${CMAKE_BINARY_DIR}/configs")

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
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/ARNOLDI"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/INVERSE_FGMRES"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/JACOBI_DAVIDSON"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/LANCZOS"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/LOBPCG"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/PAGERANK"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/POWER_ITERATION"
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/eigen_configs/SUBSPACE_ITERATION"
    DESTINATION "lib/configs/eigen_configs")

