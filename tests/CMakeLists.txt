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
cmake_minimum_required (VERSION 3.13) 
ELSE (WIN32)
cmake_minimum_required (VERSION 3.13) 
ENDIF (WIN32)

set(AMGX_INCLUDES 
    ${THRUST_DIR} 
    ${CMAKE_SOURCE_DIR}/tests/include 
    ${CMAKE_SOURCE_DIR}/core/include 
    ${CMAKE_SOURCE_DIR}/base/include)

set(tests_all ${tests_all} src/testframework.cu src/test_utils.cu src/unit_test.cu)

add_library(amgx_tests_libs OBJECT ${tests_all})

target_include_directories(amgx_tests_libs PRIVATE ${AMGX_INCLUDES}) 

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  set_target_properties(amgx_tests_libs PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
endif()

add_library(amgx_tests_library STATIC $<TARGET_OBJECTS:amgx_tests_libs>)

add_dependencies(amgx_tests_library amgx)

target_link_libraries(amgx_tests_library amgx)

add_executable(amgx_tests_launcher src/utest.cu $<TARGET_OBJECTS:amgx_tests_libs>)

target_include_directories(amgx_tests_launcher PRIVATE ${AMGX_INCLUDES}) 

# create executables for host and device, scalar and block version

find_library(CUBLAS_LIB cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${CUSPARSE_LIB} ${CUBLAS_LIB})
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${CUSPARSE_LIB} ${CUBLAS_LIB} rt dl)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl)

add_dependencies(amgx_tests_launcher amgx_tests_library)

target_link_libraries(amgx_tests_launcher amgxsh amgx ${libs_all})

if(${AMGX_PUBLIC_RELEASE} MATCHES "FALSE")
  install(TARGETS amgx_tests_launcher DESTINATION "lib/tests")
endif(${AMGX_PUBLIC_RELEASE} MATCHES "FALSE")

