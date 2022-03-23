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

# the project
project (AMG LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

#disable in-place builds
if(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  MESSAGE(FATAL_ERROR "Error:  In-place builds are not supported. Please create a separate build directory")
endif(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})

# declare the supported configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile;RelWithTraces" CACHE STRING "Avaialble Configuration Types" FORCE)

# make sure a build type has been chosen!!!
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are Debug Release Profile RelWithTraces" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

find_package(CUDAToolkit 10.0 REQUIRED)

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-terminate")
  endif()
endif()


# min compiler requirements
if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.2)
    message(
      FATAL_ERROR
      "AmgX requires GCC >= 4.8.2 but GCC version is \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc")
  endif()
endif()

FIND_PACKAGE(OpenMP)

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

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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


# Thrust:
if (DEFINED ENV{THRUST_ROOT})
  set (THRUST_DIR $ENV{THRUST_ROOT})
endif(DEFINED ENV{THRUST_ROOT})

set(AMGX_INCLUDE_EXTERNAL True CACHE BOOL "Include external 3rd party libraries")
if (${AMGX_INCLUDE_EXTERNAL})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRAPIDJSON_DEFINED")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DRAPIDJSON_DEFINED")
  include_directories("${CMAKE_SOURCE_DIR}/external/rapidjson/include")
endif (${AMGX_INCLUDE_EXTERNAL})

set(CMAKE_NO_MPI false CACHE BOOL "Force non-MPI build")

FIND_PACKAGE(MPI)
  if (${CMAKE_NO_MPI})
    message("Non-MPI build has been forced")
    set(MPI_FOUND false)
  else (${CMAKE_NO_MPI})
    if(${MPI_FOUND})
      include_directories(${MPI_INCLUDE_PATH})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAMGX_WITH_MPI")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAMGX_WITH_MPI")
      if(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMPI_IMPORTS")
      endif(WIN32)
    endif(${MPI_FOUND})
  endif (${CMAKE_NO_MPI})

message ("This is a MPI build:" ${MPI_FOUND})

# Enable NVTX ranges on Linux
if(NOT WIN32)
  set(NVTXRANGE_FLAG "-DNVTX_RANGES;")
endif()

# Configuration specific nvcc flags
GET_FILENAME_COMPONENT(CMAKE_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_NAME MATCHES "clang")
  set(CUDA_NVCC_FLAGS_DEBUG "-g;-G" CACHE STRING "Debug compiler flags")
  set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG" CACHE STRING "Release compiler flags")
  set(CUDA_NVCC_FLAGS_PROFILE "-O3;-DPROFILE" CACHE STRING "Profile compiler flags")
  set(CUDA_NVCC_FLAGS_RELWITHTRACES "-O3;-DNDEBUG;-DAMGX_USE_CPU_PROFILER" CACHE STRING "RelWithTraces compiler flags")
else()
  set(CUDA_NVCC_FLAGS_DEBUG "-g;-G" CACHE STRING "Debug compiler flags")
  set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG" CACHE STRING "Release compiler flags")
  set(CUDA_NVCC_FLAGS_PROFILE "-O3;-DPROFILE" CACHE STRING "Profile compiler flags")
  set(CUDA_NVCC_FLAGS_RELWITHTRACES "-O3;-DNDEBUG;-DAMGX_USE_CPU_PROFILER" CACHE STRING "RelWithTraces compiler flags")
  if(WIN32)
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DNOMINMAX")
  endif()
endif()

# Add the build-specific flags to the NVCC flags
string(TOUPPER ${CMAKE_BUILD_TYPE} UPP_BUILD_NAME)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_${UPP_BUILD_NAME}}")

# Add errors for execution space warnings and enable NVTX ranges
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --Werror cross-execution-space-call ${NVTXRANGE_FLAG})

# Keep intermediate files
if (AMGX_keep_intermediate)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler=-keep")
endif(AMGX_keep_intermediate)

# Windows/linux specific settings for C
GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" CACHE STRING "nvcc flags")
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler=-rdynamic;-Xcompiler=-fPIC;-Xcompiler=-fvisibility=default" CACHE STRING "nvcc flags")
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)

# VS: include object files in target property SOURCES
# otherwise a workaround for extracting ${obj_all} is necessary below
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

if(MPI_FOUND)
  set(VAMPIR_TRACE_INCLUDE_PATH "${MPI_INCLUDE_PATH}/vampirtrace")
  set(VAMPIR_TRACE_LIB_PATH "${MPI_INCLUDE_PATH}/../lib")
  if (DEFINED ENV{VAMPIR_TRACE_PATH})
    message("Found VampirTrace: $ENV{VAMPIR_TRACE_PATH}")
    set(VAMPIR_TRACE_PATH "$ENV{VAMPIR_TRACE_PATH}")
    set(VAMPIR_TRACE_LIB_PATH "${VAMPIR_TRACE_PATH}/lib")
    set(VAMPIR_TRACE_INCLUDE_PATH "${VAMPIR_TRACE_PATH}/include/vampirtrace")
  endif (DEFINED ENV{VAMPIR_TRACE_PATH})
  set(CMAKE_CXX_FLAGS_RELWITHTRACES "${CMAKE_CXX_FLAGS_RELWITHTRACES} -DAMGX_WITH_MPI;-DAMGX_USE_VAMPIR_TRACE;-DVTRACE;-I${VAMPIR_TRACE_INCLUDE_PATH}")
  set(CUDA_NVCC_FLAGS_RELWITHTRACES "${CUDA_NVCC_FLAGS_RELWITHTRACES} -DAMGX_WITH_MPI;-DAMGX_USE_VAMPIR_TRACE;-DVTRACE;-I${VAMPIR_TRACE_INCLUDE_PATH}")
  set(CMAKE_EXE_LINKER_FLAGS_RELWITHTRACES "-L${VAMPIR_TRACE_LIB_PATH} -lvt-mpi -L${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64 -lcupti ${CMAKE_EXE_LINKER_FLAGS_RELWITHTRACES}")
  set(CMAKE_SHARED_LINKER_FLAGS_RELWITHTRACES "-L${VAMPIR_TRACE_LIB_PATH} -lvt-mpi -L${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64 -lcupti ${CMAKE_SHARED_LINKER_FLAGS_RELWITHTRACES}")
  if (DEFINED ENV{VAMPIR_TRACE_PATH})
    message("Will use the following include and lib dirs with RelWithTraces: ${VAMPIR_TRACE_INCLUDE_PATH} and ${VAMPIR_TRACE_LIB_PATH}")
    set(MPI_C_LIBRARIES "${VAMPIR_TRACE_LIB_PATH}/libvt-mpi.so;${MPI_C_LIBRARIES}")
    #set(MPI_C_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/libcupti.so;${VAMPIR_TRACE_LIB_PATH}/libvt-mpi.so;${MPI_C_LIBRARIES}") // currently needed by mvapich in ubuntuh
  endif (DEFINED ENV{VAMPIR_TRACE_PATH})
endif(MPI_FOUND)

#if compiling against CUDA Toolkit 11.x + 
IF(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0.0)

  SET(CUDA_ALLOW_ARCH "60;70;80")

  # Use the generic cuSPARSE interfaces available from 10.1 on Linux, cusparseSpGEMM from 11.0
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
      -DDISABLE_MIXED_PRECISION 
      -DCUSPARSE_GENERIC_INTERFACES 
      -DCUSPARSE_USE_GENERIC_SPGEMM)

#if compiling against CUDA Toolkit 10.x
ELSEIF(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0.0)

  SET(CUDA_ALLOW_ARCH "35;52;60;70")

  IF(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.1.0)

      # Disable mixed precision for CUDA 10.1+
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
          -DDISABLE_MIXED_PRECISION 
          -DCUSPARSE_GENERIC_INTERFACES)

  ENDIF()
ENDIF()

if(DEFINED CUDA_ARCH)
    # User passed a CUDA_ARCH so check it matches
    # Error if incorrect CUDA_ARCH passed
    FOREACH(ARCH IN LISTS CUDA_ARCH)
        message(STATUS "Checking if arch " ${ARCH} " is supported...")
        IF(NOT ${ARCH} IN_LIST CUDA_ALLOW_ARCH)
            message(STATUS
                "Chosen CUDA_ARCH ${ARCH} not expected for current CUDA version. " 
                "Please choose one or more of ${CUDA_ALLOW_ARCH}.")
        ENDIF()
    ENDFOREACH()
ELSE()

    # Set a default
    SET(CUDA_ARCH "${CUDA_ALLOW_ARCH}" CACHE STRING "Target Architectures (SM60 SM70 SM80), multiple are allowed")

ENDIF()

# Add the CXX flags to the host set of CUDA flags
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler ${CMAKE_CXX_FLAGS})

# write version
set(versionFile "ReleaseVersion.txt")
set(versionCode "core/src/version.cu")
if(${versionFile} IS_NEWER_THAN ${versionCode})
  file(STRINGS ${versionFile} versionString)
  file(WRITE ${versionCode} "#include <version.h>\n")
  file(APPEND ${versionCode} "namespace amgx{\n")
  file(APPEND ${versionCode} "const char __AMGX_BUILD_DATE__ [] = __DATE__;\n")
  file(APPEND ${versionCode} "const char __AMGX_BUILD_TIME__ [] = __TIME__;\n")
  file(APPEND ${versionCode} "const char __AMGX_BUILD_ID__ [] = " ${versionString} ";" \n}\n)
endif(${versionFile} IS_NEWER_THAN ${versionCode})

# plugin management
set(pluginConf "PluginConfig.txt")                     # plugin configuration
set(pluginCode "${CMAKE_SOURCE_DIR}/plugin_config.cu") # plugin code
file(STRINGS ${pluginConf} pluginList)                 # plugin list
set(libList base core ${pluginList})                   # lib list

# create initialization and finalization for plugins
if(${pluginConf} IS_NEWER_THAN ${pluginCode})
  file(WRITE ${pluginCode} "#include <error.h>\n")
  file(APPEND ${pluginCode} "#include <misc.h>\n")
  file(APPEND ${pluginCode} "namespace amgx {\n")
  foreach( plugin ${pluginList} )
    file(APPEND ${pluginCode} "namespace ${plugin} { extern AMGX_ERROR initialize(); extern void finalize(); }\n")
  endforeach(plugin)
  file(APPEND ${pluginCode} "\nAMGX_ERROR initializePlugins() {\n")
  foreach( plugin ${pluginList} )
    file(APPEND ${pluginCode} "    if (AMGX_OK != ${plugin}::initialize()) return AMGX_ERR_PLUGIN;\n")
  endforeach(plugin)
  file(APPEND ${pluginCode} "  return AMGX_OK;\n")
  file(APPEND ${pluginCode} "}\n")
  file(APPEND ${pluginCode} "\nvoid finalizePlugins() {\n")
  foreach( plugin ${pluginList} )
    file(APPEND ${pluginCode} "  ${plugin}::finalize();\n")
  endforeach(plugin)
  file(APPEND ${pluginCode} "}\n")
  file(APPEND ${pluginCode} "} // namespace amgx\n")
endif(${pluginConf} IS_NEWER_THAN ${pluginCode})

# WARNING: the placement of the following two if statements is important, they have to come after
# FIND_PACKAGE(CUDA) and before create list of all sources/objects in amgx. Since the scripts used
# to compile CUDA are generated at configure time. This means that the scripts are generated based
# on the values of variables at the point of call (the two phases of cmake are configure then generate).

#include magma
if (MAGMA_ROOT_DIR)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DAMGX_USE_MAGMA")
  CUDA_INCLUDE_DIRECTORIES(${MAGMA_ROOT_DIR}/include)
  set(magma_libs ${MAGMA_ROOT_DIR}/lib/libmagma.a)
endif(MAGMA_ROOT_DIR)
#include lapack (MKL)
if (MKL_ROOT_DIR)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DAMGX_USE_LAPACK")
  CUDA_INCLUDE_DIRECTORIES(${MKL_ROOT_DIR}/include)
  #WARNING: MKL link line flags vary significantly (please double check http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor)
  set(mkl_libs -Wl,--start-group ${MKL_ROOT_DIR}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT_DIR}/lib/intel64/libmkl_gnu_thread.a ${MKL_ROOT_DIR}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -fopenmp)
endif(MKL_ROOT_DIR)

add_library(amgx_libs OBJECT "")

#create a list of all sources in amgx
foreach( lib ${libList} )
  add_subdirectory(${lib})
  FILE(GLOB_RECURSE TESTS "${CMAKE_SOURCE_DIR}/${lib}/tests/*.cu")
endforeach(lib)

# set arch for main libs target
set_target_properties(amgx_libs PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

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
  target_link_libraries( amgx CUDA::cublas CUDA::cusparse CUDA::cusolver)
  target_link_libraries( amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver)
ELSE (WIN32)
  target_link_libraries( amgx CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread)
  target_link_libraries( amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread)
ENDIF(WIN32)

if(MPI_FOUND)
    target_link_libraries( amgx   ${MPI_C_LIBRARIES})
    target_link_libraries( amgxsh ${MPI_C_LIBRARIES})
endif(MPI_FOUND)

#link magma
if (MAGMA_ROOT_DIR)
  target_link_libraries(amgx   ${magma_libs})
  target_link_libraries(amgxsh ${magma_libs})
endif(MAGMA_ROOT_DIR)
#link lapack (MKL)
if (MKL_ROOT_DIR)
  target_link_libraries(amgx   ${mkl_libs} )
  target_link_libraries(amgxsh ${mkl_libs})
endif(MKL_ROOT_DIR)

# set arch for main libs
set_target_properties(amgx PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(amgxsh PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

install(TARGETS amgx   DESTINATION "lib")
install(TARGETS amgxsh DESTINATION "lib")

#export(TARGETS amgxsh FILE ${CMAKE_SOURCE_DIR}/amgxsh.cmake)

# build examples
add_subdirectory(examples)
if(EXISTS ${CMAKE_SOURCE_DIR}/eigen_examples)
  add_subdirectory(eigen_examples)
endif(EXISTS ${CMAKE_SOURCE_DIR}/eigen_examples)

# build unit tests
if(EXISTS ${CMAKE_SOURCE_DIR}/tests)
   add_subdirectory(tests)
endif(EXISTS ${CMAKE_SOURCE_DIR}/tests)
