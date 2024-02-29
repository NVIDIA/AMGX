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

set_target_properties(amgx_tests_launcher PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(amgx_tests_libs PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(amgx_tests_library PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

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

target_link_libraries(amgx_tests_launcher amgx_tests_library amgxsh ${libs_all} OpenMP::OpenMP_C)

if(MPI_FOUND)
    target_link_libraries(amgx_tests_launcher MPI::MPI_CXX)
endif(MPI_FOUND)

if(NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")
  install(TARGETS amgx_tests_launcher DESTINATION "lib/tests")
endif(NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")

