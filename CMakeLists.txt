cmake_minimum_required(VERSION 3.13.0)

project(
  OGL
  LANGUAGES C CXX
  DESCRIPTION
    "A wrapper for Ginkgo solver to provide GPGPU capabilities to OpenFOAM")

if(NOT DEFINED ENV{FOAM_SRC})
  message(FATAL_ERROR "You must source OpenFOAM before building OGL")
endif()

# Configuration options

include(cmake/build_type_helpers.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules/")

include(CheckIncludeFileCXX)
option(OGL_USE_EXTERNAL_GINKGO "Use external ginkgo" FALSE)
option(GINKGO_BUILD_CUDA "Build ginkgo with cuda backend" FALSE)
option(GINKGO_BUILD_HIP "Build ginkgo with hip backend" FALSE)
option(GINKGO_BUILD_OMP "Build ginkgo with omp backend" FALSE)
option(GINKGO_BUILD_REFERENCE "Build ginkgo with reference backend" TRUE)
option(GINKGO_WITH_OGL_EXTENSIONS "Whether ginkgo was build with OGL extension"
       FALSE)

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

include(cmake/package_helpers.cmake)
include(cmake/build_helpers.cmake)
include(cmake/install_helpers.cmake)

ginkgo_find_package(Ginkgo "Ginkgo::ginkgo" FALSE 1.4.0)

add_subdirectory(third_party)

# If ginkgo wasn't found so far look for the third_party installation
if(NOT ${OGL_USE_EXTERNAL_GINKGO})
  find_package(
    Ginkgo
    CONFIG
    1.4.0
    REQUIRED
    PATHS
    $ENV{FOAM_USER_LIBBIN}/lib/cmake/Ginkgo
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/ginkgo/build/install/lib/cmake)
endif()

# To interface with ginkgo at least C++ 14 is needed
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} \
  -Wall -Wpedantic -Wextra -march=native -fopenmp -fPIC -Wno-undefined-var-template"
)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  message("Compiling with clang")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} \
    -fno-omit-frame-pointer \
    -fsanitize=address")

  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} \
    -fno-omit-frame-pointer \
    -Rpass=loop-vectorize \
    -ffast-math \
    "
      # -Rpass-analysis=loop-vectorize \ -Rpass-missed=loop-vectorize \
  )

elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  add_definitions(-DWITH_GNU)

  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} \
      --coverage\
      -fprofile-arcs \
      -fsanitize=address \
      -ggdb3 \
      -ftest-coverage")

  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} \
      -ffast-math \
      -march=native  \
      -ftree-vectorize \
      -rdynamic \
      ")
endif()

option(BUILD_SHARED_LIBS "Build shared library" ON)
add_library(OGL "")

add_compile_definitions(WM_LABEL_SIZE=32 WM_ARCH_OPTION=64 NoRepository WM_DP)

if(APPLE)
  message("Compiling on MacOS")
  target_link_options(OGL PUBLIC "LINKER:-undefined,dynamic_lookup")
  add_compile_definitions(APPLE arm64)
else()
  add_compile_definitions(linux64)
endif()

target_sources(
  OGL
  PRIVATE common/common.C
          StoppingCriterion/StoppingCriterion.C
          DevicePersistent/DevicePersistentBase/DevicePersistentBase.C
          DevicePersistent/DevicePersistentArray/DevicePersistentArray.C
          DevicePersistent/ExecutorHandler/ExecutorHandler.C
          DevicePersistent/CsrMatrixWrapper/CsrMatrixWrapper.C
          Preconditioner/Preconditioner.C
          DevicePersistent/IOGlobalIndex/gkoGlobalIndex.C
          BaseWrapper/lduBase/GKOlduBase.C
          BaseWrapper/CoupledLduBase/GKOCoupledLduBase.C
          HostMatrix/HostMatrix.C
          Solver/CG/GKOCG.C
          Solver/BiCGStab/GKOBiCGStab.C
          # Solver/IR/GKOIR.C Solver/Multigrid/GKOMultigrid.C
          Solver/GMRES/GKOGMRES.C
          LduMatrix/GKOACG/GKOACG.C
  PUBLIC common/common.H
         StoppingCriterion/StoppingCriterion.H
         lduLduBase/lduLduBase.H
         HostMatrix/HostMatrix.H
         DevicePersistent/DevicePersistentBase/DevicePersistentBase.H
         DevicePersistent/DevicePersistentArray/DevicePersistentArray.H
         DevicePersistent/ExecutorHandler/ExecutorHandler.H
         DevicePersistent/CsrMatrixWrapper/CsrMatrixWrapper.H
         Preconditioner/Preconditioner.H
         DevicePersistent/IOGlobalIndex/gkoGlobalIndex.H
         BaseWrapper/lduBase/GKOlduBase.H
         BaseWrapper/CoupledLduBase/GKOCoupledLduBase.H
         Solver/CG/GKOCG.H
         Solver/IR/GKOIR.H
         Solver/Multigrid/GKOMultigrid.H
         Solver/BiCGStab/GKOBiCGStab.H
         Solver/GMRES/GKOGMRES.H
         LduMatrix/GKOACG/GKOACG.H)

target_include_directories(
  OGL SYSTEM
  PRIVATE $ENV{FOAM_SRC}/finiteVolume/lnInclude
          $ENV{FOAM_SRC}/meshTools/lnInclude
          $ENV{FOAM_SRC}/OpenFOAM/lnInclude
          $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude
          common/
          BaseWrapper/lduBase/
          BaseWrapper/CoupledLduBase/
          DevicePersistent/DevicePersistentBase/
          DevicePersistent/ExecutorHandler/
          DevicePersistent/CsrMatrixWrapper/
          Preconditioner/
          DevicePersistent/IOGlobalIndex/)

target_link_libraries(OGL PUBLIC Ginkgo::ginkgo)

if(${GINKGO_WITH_OGL_EXTENSIONS})
  target_compile_definitions(OGL PRIVATE GINKGO_WITH_OGL_EXTENSIONS=1)
endif()

install(TARGETS OGL DESTINATION $ENV{FOAM_USER_LIBBIN})
