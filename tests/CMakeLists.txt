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

FIND_PACKAGE(CUDA)

CUDA_INCLUDE_DIRECTORIES(${THRUST_DIR}) 
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/tests/include)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/core/include)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/base/include)

set(tests_all ${tests_all} src/testframework.cu src/test_utils.cu src/unit_test.cu)

CUDA_ADD_LIBRARY(amgx_tests_library STATIC ${tests_all})

# workaround
get_target_property(src_tmp amgx_tests_library SOURCES)
set(tests_all ${tests_all} ${src_tmp})

set(tests_obj_all)
foreach( src ${tests_all} )
  if(${src} MATCHES ${CMAKE_BINARY_DIR})
    set(tests_obj_all ${tests_obj_all} ${src})
  endif(${src} MATCHES ${CMAKE_BINARY_DIR})
endforeach(src)
SET_SOURCE_FILES_PROPERTIES(${tests_obj_all} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
#file(WRITE "tests_obj_all.txt" "${tests_obj_all}") # debug, write out all objects for amgx

#CUDA_ADD_EXECUTABLE(amgx_tests_launcher ${tests_all} src/testframework.cu src/test_utils.cu src/unit_test.cu src/utest.cu)
CUDA_ADD_EXECUTABLE(amgx_tests_launcher src/utest.cu ${tests_obj_all})

# create executables for host and device, scalar and block version
# set(libs_all ${cusparse_library})

GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${cusparse_library})
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(libs_all ${cusparse_library} rt dl)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl)

add_dependencies(amgx_tests_launcher amgx_tests_library)
target_link_libraries(amgx_tests_launcher amgx ${libs_all})

if(${AMGX_PUBLIC_RELEASE} MATCHES "FALSE")
  install(TARGETS amgx_tests_launcher DESTINATION "lib/tests")
endif(${AMGX_PUBLIC_RELEASE} MATCHES "FALSE")
