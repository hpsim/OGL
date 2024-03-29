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
option(OGL_DATA_VALIDATION "Check if resulting matrix data is valid" FALSE)
option(OGL_USE_FOAM_FOUNDATION_VERSION
       "Build OGL for the OpenFOAM Foundation version" FALSE)
option(OGL_BUILD_UNITTEST "Build OGL unittests" FALSE)
option(OGL_CUDA_ARCHITECTURES "Which cuda architecture to build for" "native")
option(OGL_DP_LABELS "Compile for 64bit label size" FALSE)
option(GINKGO_BUILD_CUDA "Build Ginkgo with cuda backend" FALSE)
option(GINKGO_BUILD_HIP "Build Ginkgo with hip backend" FALSE)
option(GINKGO_BUILD_SYCL "Build Ginkgo with sycl backend" FALSE)
option(GINKGO_BUILD_OMP "Build Ginkgo with omp backend" FALSE)
option(GINKGO_BUILD_REFERENCE "Build Ginkgo with reference backend" TRUE)
option(OGL_ALLOW_REFERENCE_ONLY
       "Enable builds of Ginkgo with reference backend only" FALSE)
option(GINKGO_FORCE_GPU_AWARE_MPI "Build Ginkgo using device aware MPI" TRUE)
option(GINKGO_WITH_OGL_EXTENSIONS "Whether ginkgo was build with OGL extension"
       FALSE)

if(NOT OGL_ALLOW_REFERENCE_ONLY)
  if((NOT GINKGO_BUILD_CUDA)
     AND (NOT GINKGO_BUILD_HIP)
     AND (NOT GINKGO_BUILD_SYCL))
    message(
      FATAL_ERROR
        "No GPU Backend was selected, set either:\n -DGINKGO_BUILD_CUDA=True\n -DGINKGO_BUILD_HIP=True\n -DGINKGO_BUILD_SYCL=True\n or set:\n -DOGL_ALLOW_REFERENCE_ONLY=True\n to turn off this check"
    )
  endif()
endif()

set(GINKGO_CHECKOUT_VERSION
    "fc86d48b78cebd2b2c5833a2dcf0fe40f615cf19"
    CACHE STRING "Use specific version of ginkgo")

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

include(cmake/package_helpers.cmake)
include(cmake/build_helpers.cmake)
include(cmake/install_helpers.cmake)

ginkgo_find_package(Ginkgo "Ginkgo::ginkgo" FALSE 1.7.0)

add_subdirectory(third_party)

# If ginkgo is not found at this point, look for the third_party installation
if(NOT ${OGL_USE_EXTERNAL_GINKGO})
  find_package(
    Ginkgo
    CONFIG
    1.7.0
    REQUIRED
    PATHS
    $ENV{FOAM_USER_LIBBIN}/cmake/Ginkgo
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/ginkgo/build/install/lib/cmake)
endif()

# C++ 14 standard or later is required to interface with Ginkgo OGL itself uses
# some c++ 17 features

set(CMAKE_CXX_STANDARD 17)
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
      # * Rpass-analysis=loop-vectorize  -Rpass-missed=loop-vectorize
  )

elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  add_definitions(-DWITH_GNU)

  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} \
      --coverage\
      -fprofile-arcs \
      -ggdb3")
  # * fsanitize = address \ - ftest - coverage \

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

if(OGL_DP_LABELS)
  target_compile_definitions(OGL PUBLIC WM_LABEL_SIZE=64 WM_ARCH_OPTION=64
                                        NoRepository WM_DP)
else()
  target_compile_definitions(OGL PUBLIC WM_LABEL_SIZE=32 WM_ARCH_OPTION=64
                                        NoRepository WM_DP)
endif()

if(APPLE)
  message("Compiling on MacOS")
  target_link_options(OGL PUBLIC "LINKER:-undefined,dynamic_lookup")
  target_compile_definitions(OGL PRIVATE APPLE arm64)
else()
  target_compile_definitions(OGL PRIVATE linux64)
endif()

# Add a custom command that produces version.cpp, plus a dummy output that's not
# actually produced, in order to always force re-run version.cmake before the
# build
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.C
         ${CMAKE_CURRENT_BINARY_DIR}/_version.C
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
          ${GINKGO_CHECKOUT_VERSION})

target_sources(
  OGL
  PRIVATE common/common.C
          ${CMAKE_CURRENT_BINARY_DIR}/version.C
          lduLduBase/lduLduBase.C
          StoppingCriterion/StoppingCriterion.C
          DevicePersistent/Base/Base.C
          DevicePersistent/Partition/Partition.C
          DevicePersistent/Array/Array.C
          DevicePersistent/Vector/Vector.C
          DevicePersistent/DeviceIdGuard/DeviceIdGuard.C
          DevicePersistent/ExecutorHandler/ExecutorHandler.C
          DevicePersistent/CsrMatrixWrapper/CsrMatrixWrapper.C
          Preconditioner/Preconditioner.C
          BaseWrapper/lduBase/GKOlduBase.C
          BaseWrapper/CoupledLduBase/GKOCoupledLduBase.C
          HostMatrix/HostMatrix.C
          HostMatrix/HostMatrixFreeFunctions.C
          Solver/CG/GKOCG.C
          Solver/BiCGStab/GKOBiCGStab.C
          # Solver / IR / GKOIR.C Solver / Multigrid / GKOMultigrid.C
          Solver/GMRES/GKOGMRES.C
  # LduMatrix / GKOACG / GKOACG.C
  PUBLIC common/common.H
         StoppingCriterion/StoppingCriterion.H
         lduLduBase/lduLduBase.H
         HostMatrix/HostMatrix.H
         DevicePersistent/Base/Base.H
         DevicePersistent/Partition/Partition.H
         DevicePersistent/Array/Array.H
         DevicePersistent/Vector/Vector.H
         DevicePersistent/ExecutorHandler/ExecutorHandler.H
         DevicePersistent/DeviceIdGuard/DeviceIdGuard.H
         DevicePersistent/CsrMatrixWrapper/CsrMatrixWrapper.H
         Preconditioner/Preconditioner.H
         BaseWrapper/lduBase/GKOlduBase.H
         BaseWrapper/CoupledLduBase/GKOCoupledLduBase.H
         Solver/CG/GKOCG.H
         Solver/IR/GKOIR.H
         # Solver / Multigrid / GKOMultigrid.H
         Solver/BiCGStab/GKOBiCGStab.H
         Solver/GMRES/GKOGMRES.H
  # LduMatrix / GKOACG / GKOACG.H
)

target_include_directories(
  OGL SYSTEM
  PUBLIC $ENV{FOAM_SRC}/finiteVolume/lnInclude
         $ENV{FOAM_SRC}/meshTools/lnInclude $ENV{FOAM_SRC}/OpenFOAM/lnInclude
         $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  OGL
  PRIVATE $ENV{FOAM_LIBBIN}/libOpenFOAM.so $ENV{FOAM_LIBBIN}/libfiniteVolume.so
          $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.so
  PUBLIC Ginkgo::ginkgo stdc++fs)

if(${GINKGO_WITH_OGL_EXTENSIONS})
  target_compile_definitions(OGL PRIVATE GINKGO_WITH_OGL_EXTENSIONS=1)
endif()
if(${GINKGO_BUILD_CUDA})
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(OGL PRIVATE GINKGO_BUILD_CUDA=1)
  target_link_libraries(OGL PUBLIC CUDA::nvToolsExt)
endif()

if(OGL_DATA_VALIDATION)
  target_compile_definitions(OGL PRIVATE DATA_VALIDATION=1)
endif()

if(EXISTS $ENV{WM_PROJECT_DIR}/META-INFO)
  message("Compile against ESI version of OpenFOAM")
  target_compile_definitions(OGL PRIVATE WITH_ESI_VERSION=1)
else()
  message("Compile against FoamFoundation version of OpenFOAM")
endif()

if(${OGL_BUILD_UNITTEST})
  enable_testing()
  add_subdirectory(unitTests)
endif()

add_custom_target(
  cmake-format
  COMMAND find . -name CMakeLists.txt -exec cmake-format -i {} +
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

install(TARGETS OGL DESTINATION $ENV{FOAM_USER_LIBBIN})
