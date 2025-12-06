# Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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

# IMPORTANT: CMAKE_CUDA_ARCHITECTURES must be set before project() enables CUDA
# Backward compatibility: translate old CUDA_ARCH to new CMAKE_CUDA_ARCHITECTURES
if(DEFINED CUDA_ARCH)
    message(WARNING "CUDA_ARCH is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead.")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}" CACHE STRING "Target CUDA architectures" FORCE)
endif()

# If CMAKE_CUDA_ARCHITECTURES is not set, provide a default
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to common architectures (will be validated later based on CUDA version)
    set(CMAKE_CUDA_ARCHITECTURES "90;100;120" CACHE STRING 
        "CUDA architectures to compile for (e.g., 90;100;120)")
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# the project
project (AMG LANGUAGES C CXX CUDA)

find_package(MPI)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

#disable in-place builds
if(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  MESSAGE(FATAL_ERROR "Error:  In-place builds are not supported. Please create a separate build directory")
endif(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})

# declare the supported configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile;RelWithTraces" CACHE STRING "Avaialble Configuration Types" FORCE)

# make sure a build type has been chosen!!!
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are Debug Release Profile RelWithTraces" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

find_package(CUDAToolkit 12.0 REQUIRED)

# update/define the compilation flags.
IF(WIN32)
  set(CMAKE_C_FLAGS                "/DWIN32 /D_WINDOWS /W3 /bigobj" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_DEBUG          "/D_DEBUG /Zl /Zi /Ob0 /Od /RTC1" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_MINSIZEREL     "/D_DEBUG /Zl /Zi /Ob0 /Od /RTC1" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_RELEASE        "/MT /O2 /Ob2 /D NDEBUG" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "/MT /O2 /Ob2 /D NDEBUG" CACHE STRING "" FORCE)

  set(CMAKE_CXX_FLAGS                "/DWIN32 /D_WINDOWS /DNOMINMAX /W3 /GR /EHsc /bigobj" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG          "/D_DEBUG /Zl /Zi /Ob0 /Od /RTC1" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "/D_DEBUG /Zl /Zi /Ob0 /Od /RTC1" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE        "/MT /O2 /Ob2 /DNDEBUG" CACHE STRING "Force flags. /Zl is used to remove deps" FORCE)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MT /O2 /Ob2 /DNDEBUG" CACHE STRING "Force flags. /Zl is used to remove deps" FORCE)
ENDIF(WIN32)

# disabling specific warnings
if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0)
    # we throw in ~Matrix(), we assume any error fatal so std::terminate call is ok
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-terminate)
  endif()
endif()

set(CMAKE_C_FLAGS_PROFILE "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "" FORCE)

set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "")

set(CMAKE_C_FLAGS_RELWITHTRACES "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHTRACES "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHTRACES "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHTRACES "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "" FORCE)

# install paths
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ".." CACHE PATH "Path where AMGX will be installed" FORCE)
endif()
# add CMAKE_INSTALL_PREFIX/lib to the RPATH to be used when installing,
# but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
     "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ignore rpath completely, if requested:
set(AMGX_NO_RPATH False CACHE BOOL "Don't build RPATH")
if (${AMGX_NO_RPATH})
  set(CMAKE_SKIP_RPATH TRUE)
endif (${AMGX_NO_RPATH})

if (DEFINED ENV{CRAY_MPICH2_DIR})
  set(MPI_C_LIBRARIES "$ENV{CRAY_MPICH2_DIR}/lib/libmpich.so")
  set(MPI_C_INCLUDE_PATH "$ENV{CRAY_MPICH2_DIR}/include")
endif(DEFINED ENV{CRAY_MPICH2_DIR})

set(AMGX_INCLUDE_EXTERNAL True CACHE BOOL "Include external 3rd party libraries")
if (AMGX_INCLUDE_EXTERNAL)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DRAPIDJSON_DEFINED)
  set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DRAPIDJSON_DEFINED)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/external/rapidjson/include")
endif (AMGX_INCLUDE_EXTERNAL)

set(CMAKE_NO_MPI false CACHE BOOL "Force non-MPI build")

if (CMAKE_NO_MPI)
  message("Non-MPI build has been forced")
  set(MPI_FOUND false)
else (CMAKE_NO_MPI)
  if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DAMGX_WITH_MPI)
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DAMGX_WITH_MPI)
    if(WIN32)
      set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DOMPI_IMPORTS)
    endif(WIN32)
  endif(MPI_FOUND)
endif (CMAKE_NO_MPI)

message ("This is a MPI build:" ${MPI_FOUND})

# Enable NVTX ranges on Linux
if(NOT WIN32)
  set(NVTXRANGE_FLAG -DNVTX_RANGES)
endif()

# Configuration specific nvcc flags
GET_FILENAME_COMPONENT(CMAKE_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_NAME MATCHES "clang")
  set(CUDA_NVCC_FLAGS_DEBUG -g -G)
  set(CUDA_NVCC_FLAGS_RELEASE -DNDEBUG)
  set(CUDA_NVCC_FLAGS_PROFILE -DPROFILE)
  set(CUDA_NVCC_FLAGS_RELWITHTRACES -g -DNDEBUG)
else()
  set(CUDA_NVCC_FLAGS_DEBUG -g -G)
  set(CUDA_NVCC_FLAGS_RELEASE -DNDEBUG)
  set(CUDA_NVCC_FLAGS_PROFILE -DPROFILE)
  set(CUDA_NVCC_FLAGS_RELWITHTRACES -g -DNDEBUG)
  if(WIN32)
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -DNOMINMAX)
  endif()
endif()

# Add the build-specific flags to the NVCC flags
string(TOUPPER ${CMAKE_BUILD_TYPE} UPP_BUILD_NAME)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_${UPP_BUILD_NAME}})

# Enable device lambdas
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --extended-lambda)

# Add errors for execution space warnings and enable NVTX ranges
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --Werror cross-execution-space-call ${NVTXRANGE_FLAG})

# Keep intermediate files
if (AMGX_keep_intermediate)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -keep)
endif(AMGX_keep_intermediate)

# Windows/linux specific settings for C
GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -rdynamic -fPIC -fvisibility=default)
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)

# VS: include object files in target property SOURCES
# otherwise a workaround for extracting ${obj_all} is necessary below
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

IF(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)

  SET(CUDA_ALLOW_ARCH "70;75;80;86;89;90;100;120")

ELSEIF(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0)

  SET(CUDA_ALLOW_ARCH "70;75;80;86;89;90;100;120")

ELSE()

  MESSAGE(FATAL_ERROR "Error: CUDA minimum supported version is 12.0.")

ENDIF()

# Validate CMAKE_CUDA_ARCHITECTURES against what this CUDA version supports
message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} supports architectures: ${CUDA_ALLOW_ARCH}")
message(STATUS "Building for architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Add the CXX flags to the host set of CUDA flags
string(REPLACE ";" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler ${CMAKE_CXX_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DTHRUST_CUB_WRAPPED_NAMESPACE=amgx)

add_library(amgx_libs OBJECT "")

add_subdirectory("src")

# CMAKE_CUDA_ARCHITECTURES automatically applies to all CUDA targets
# No need to set per-target properties when set as cache variable above

target_compile_options(amgx_libs PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS} >)

# build amgx
add_library(amgx STATIC $<TARGET_OBJECTS:amgx_libs>)  # static lib
target_link_libraries(amgx amgx_libs)

add_library(amgxsh SHARED $<TARGET_OBJECTS:amgx_libs>)  # shared lib
target_link_libraries(amgxsh amgx_libs)

set_target_properties(amgx PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(amgxsh PROPERTIES LINKER_LANGUAGE CUDA)

target_compile_options(amgx PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS} >)
target_compile_options(amgxsh PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS} >)

IF (WIN32)
  target_link_libraries(amgx CUDA::cublas CUDA::cusparse CUDA::cusolver)
  target_link_libraries(amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver)
ELSE (WIN32)
  target_link_libraries(amgx CUDA::cublas CUDA::cusparse CUDA::cusolver m pthread)
  target_link_libraries(amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver m pthread)
ENDIF(WIN32)

if(MPI_FOUND)
    target_link_libraries(amgx   MPI::MPI_CXX)
    target_link_libraries(amgxsh MPI::MPI_CXX)
endif(MPI_FOUND)

# CMAKE_CUDA_ARCHITECTURES is inherited by all targets automatically

install(FILES
  include/amgx_config.h
  include/amgx_c.h
  DESTINATION include)

install(TARGETS amgx   DESTINATION "lib")
install(TARGETS amgxsh DESTINATION "lib")

#export(TARGETS amgxsh FILE ${CMAKE_CURRENT_SOURCE_DIR}/amgxsh.cmake)

# build examples
add_subdirectory(examples)

