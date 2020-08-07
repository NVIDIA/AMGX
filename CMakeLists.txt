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

# the project
project (AMG)


#disable in-place builds
if(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  MESSAGE(FATAL_ERROR "Error:  In-place builds are not supported. Please create a separate build directory")
endif(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})

# set the path to extra modules.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/cuda" ${CMAKE_MODULE_PATH})

# declare the supported configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile;RelWithTraces" CACHE STRING "Avaialble Configuration Types" FORCE)

# make sure a build type has been chosen!!!
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are Debug Release Profile RelWithTraces" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

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
    message(FATAL_ERROR "Insufficient gcc version. 4.8.2 is required to compile AMGX")
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
set (THRUST_DIR "${PROJECT_SOURCE_DIR}/../../thrust")
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

# enable source level parallel builds in visual studio
#IF(CMAKE_GENERATOR MATCHES "Visual Studio")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /M2 /MP")
#ENDIF (CMAKE_GENERATOR MATCHES "Visual Studio")

# Enable NVTX ranges on Linux
if(NOT WIN32)
  set(NVTXRANGE_FLAG "-DNVTX_RANGES;")
endif()

#Configuration specific nvcc flags
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CUDA_NVCC_FLAGS_DEBUG "-g;-G;-std=c++14;--Werror cross-execution-space-call;${NVTXRANGE_FLAG}" CACHE STRING "Debug compiler flags")
  set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG;-std=c++14;--Werror cross-execution-space-call;${NVTXRANGE_FLAG}" CACHE STRING "Release compiler flags")
  set(CUDA_NVCC_FLAGS_PROFILE "-O3;-DPROFILE;-std=c++14;${NVTXRANGE_FLAG}" CACHE STRING "Profile compiler flags")
  set(CUDA_NVCC_FLAGS_RELWITHTRACES "-O3;-DNDEBUG;-DAMGX_USE_CPU_PROFILER;-std=c++14;${NVTXRANGE_FLAG}" CACHE STRING "RelWithTraces compiler flags")
else()
  set(CUDA_NVCC_FLAGS_DEBUG "-g;-G;--Werror cross-execution-space-call;${NVTXRANGE_FLAG}" CACHE STRING "Debug compiler flags")
  set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG;--Werror cross-execution-space-call;${NVTXRANGE_FLAG}" CACHE STRING "Release compiler flags")
  set(CUDA_NVCC_FLAGS_PROFILE "-O3;-DPROFILE;${NVTXRANGE_FLAG}" CACHE STRING "Profile compiler flags")
  set(CUDA_NVCC_FLAGS_RELWITHTRACES "-O3;-DNDEBUG;-DAMGX_USE_CPU_PROFILER;${NVTXRANGE_FLAG}" CACHE STRING "RelWithTraces compiler flags")
	if(WIN32)
		set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DNOMINMAX")
	endif()
endif()

#keep intermediate files
if (AMGX_keep_intermediate)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler=-keep")
endif(AMGX_keep_intermediate)

#windows/linux specific settings for C
GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
IF(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" CACHE STRING "nvcc flags")
ELSE(CMAKE_C_COMPILER_NAME MATCHES cl)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler=-rdynamic;-Xcompiler=-fPIC;-Xcompiler=-fvisibility=default" CACHE STRING "nvcc flags")
ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl)

# VS: include object files in target property SOURCES
# otherwise a workaround for extracting ${obj_all} is necessary below
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# load CUDA.
FIND_PACKAGE(CUDA)

message("Cuda libraries: " ${CUDA_LIBRARIES})

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

#if compiling against CUDA Toolkit 8.x
IF(CUDA_VERSION_MAJOR MATCHES 8)
  SET(CUDA_ARCH "35 37 52 60" CACHE STRING "Target Architectures (SM35 SM37 SM52 SM60), multiple are allowed")
ENDIF(CUDA_VERSION_MAJOR MATCHES 8)

#if compiling against CUDA Toolkit 9.x
IF(CUDA_VERSION_MAJOR MATCHES 9)
  SET(CUDA_ARCH "35 52 60 70" CACHE STRING "Target Architectures (SM35 SM52 SM60 SM70), multiple are allowed")
ENDIF(CUDA_VERSION_MAJOR MATCHES 9)

#if compiling against CUDA Toolkit 10.x
IF(CUDA_VERSION_MAJOR MATCHES 10)
  SET(CUDA_ARCH "35 52 60 70" CACHE STRING "Target Architectures (SM35 SM52 SM60 SM70), multiple are allowed")

  IF(CUDA_VERSION_MINOR GREATER 0)
    # Disable mixed precision for CUDA 10.1+
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDISABLE_MIXED_PRECISION)

    # Use the generic cuSPARSE interfaces available from 10.1 on Linux
    IF(NOT WIN32)
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DCUSPARSE_GENERIC_INTERFACES)
    ENDIF(NOT WIN32)
  ENDIF(CUDA_VERSION_MINOR GREATER 0)
ENDIF(CUDA_VERSION_MAJOR MATCHES 10)

#if compiling against CUDA Toolkit 11.x
IF(CUDA_VERSION_MAJOR MATCHES 11)
  SET(CUDA_ARCH "70 80" CACHE STRING "Target Architectures (SM70 SM80), multiple are allowed")

  # Disable mixed precision for CUDA 10.1+
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDISABLE_MIXED_PRECISION)

  # Use the generic cuSPARSE interfaces available from 10.1 on Linux
  IF(NOT WIN32)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DCUSPARSE_GENERIC_INTERFACES)
  ENDIF(NOT WIN32)
ENDIF(CUDA_VERSION_MAJOR MATCHES 11)

#replace ' ' with ; to match the proper cmake format
STRING(REGEX REPLACE " " ";" CUDA_ARCH ${CUDA_ARCH})

#set the compiler flags for each NV target
FOREACH(target ${CUDA_ARCH})
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_${target},code=\\\"sm_${target},compute_${target}\\\")
ENDFOREACH(target ${CUDA_ARCH})

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
set(srcTarball "../srcTarball.txt")                    # dirs for source tarball


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

# create file with files and dirs for source tarball
#if(${pluginConf} IS_NEWER_THAN ${srcTarball})
  set(dirList "CMakeLists.txt" ${versionFile} ${pluginConf} "doc" "examples" ${libList})
  if (${AMGX_INCLUDE_EXTERNAL})
    set(dirList ${dirList} "external")
  endif (${AMGX_INCLUDE_EXTERNAL})
  set(dirList ${dirList} "tests")
  string(REPLACE "/" " " dirliststr ${CMAKE_SOURCE_DIR})
  string(REGEX MATCHALL "([^\ ]+\ |[^\ ]+$)" dirlist "${dirliststr}")
  list(GET dirlist -1 branch)
  #message (STATUS "working with branch = " ${branch})
  file(WRITE ${srcTarball} "")
  foreach( dir ${dirList} )
    file(TO_NATIVE_PATH "${branch}/${dir}" nativePath)
    file(APPEND ${srcTarball} "${nativePath}\n")
  endforeach(dir)
#endif(${pluginConf} IS_NEWER_THAN ${srcTarball})

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

#create a list of all sources in amgx
set(src_all)                              # collect all sources from libs
set(target_all)                           # collect all targets from libs
set(tests_all)
foreach( lib ${libList} )
  add_subdirectory(${lib})
  get_target_property(src amgx_${lib} SOURCES)
  set(src_all ${src_all} ${src})
  set(target_all ${target_all} amgx_${lib})
  FILE(GLOB_RECURSE TESTS "${CMAKE_SOURCE_DIR}/${lib}/tests/*.cu")
  set(tests_all ${tests_all} ${TESTS})
endforeach(lib)

# create a list of all objects in amgx
set(obj_all)
set(cpp_all)
foreach( src ${src_all} )
  if(${src} MATCHES ${CMAKE_BINARY_DIR})
    set(obj_all ${obj_all} ${src})
  else(${src} MATCHES ${CMAKE_BINARY_DIR})
    set(cpp_all ${cpp_all} ${src})
  endif(${src} MATCHES ${CMAKE_BINARY_DIR})
endforeach(src)

# build amgx
set_source_files_properties(${obj_all} PROPERTIES GENERATED TRUE)
CUDA_ADD_LIBRARY(amgx   STATIC ${obj_all} )  # static lib
CUDA_ADD_LIBRARY(amgxsh SHARED ${obj_all} )  # shared lib
if (target_all)
  add_dependencies(amgx  ${target_all})
  add_dependencies(amgxsh ${target_all})
endif (target_all)

find_library_local_first(cublas_library cublas "The cuBLAS CUDA library")
find_library_local_first(cusparse_library cusparse "The cusparse CUDA library")
find_library_local_first(cusolver_library cusolver "The cusolver CUDA library")

IF (WIN32)
  target_link_libraries( amgx ${cublas_library} ${cusparse_library} ${cusolver_library})
  target_link_libraries( amgxsh ${cublas_library} ${cusparse_library} ${cusolver_library})
ELSE (WIN32)
  find_library_local_first(nvtx nvToolsExt "nvtx CUDA library")
  target_link_libraries( amgx ${cublas_library} ${cusparse_library} ${cusolver_library} ${nvtx} m pthread)
  target_link_libraries( amgxsh ${cublas_library} ${cusparse_library} ${cusolver_library} ${nvtx} m pthread)
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

install(TARGETS amgx   DESTINATION "lib")
install(TARGETS amgxsh DESTINATION "lib")

export(TARGETS amgxsh FILE ${CMAKE_SOURCE_DIR}/amgxsh.cmake)

# build examples
add_subdirectory(examples)
if(EXISTS ${CMAKE_SOURCE_DIR}/eigen_examples)
  add_subdirectory(eigen_examples)
endif(EXISTS ${CMAKE_SOURCE_DIR}/eigen_examples)

# build unit tests
if(EXISTS ${CMAKE_SOURCE_DIR}/tests)
   add_subdirectory(tests)
endif(EXISTS ${CMAKE_SOURCE_DIR}/tests)
