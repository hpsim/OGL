# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.13.0)

project(
  OGL
  LANGUAGES C CXX
  DESCRIPTION
    "A wrapper for Ginkgo solver to provide GPGPU capabilities to OpenFOAM")

if(NOT DEFINED ENV{FOAM_SRC})
  message(FATAL_ERROR "You must source OpenFOAM before building OGL")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS ON)

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
option(GINKGO_BUILD_REFERENCE "Build Ginkgo with reference backend" TRUE)
option(OGL_ALLOW_REFERENCE_ONLY
       "Enable builds of Ginkgo with reference backend only" FALSE)
option(OGL_ENABLE_SANITIZE_ADDRESS "Enable address sanitizer" FALSE)
option(OGL_ENABLE_SANITIZE_LEAK "Enable leak sanitizer" FALSE)
option(OGL_ENABLE_SANITIZE_UB "Enable undefined behaviour sanitizer" FALSE)
option(OGL_ENABLE_SANITIZE_THREAD "Enable thread sanitizer" FALSE)
option(OGL_ENABLE_SANITIZE_MEMORY "Enable memory sanitizer" FALSE)
option(OGL_DEVEL_TOOLS "Add development tools to the build system" FALSE)

option(GINKGO_WITH_OGL_EXTENSIONS "Whether ginkgo was build with OGL extension"
       FALSE)
option(GINKGO_BUILD_REFERENCE "Build Ginkgo with reference backend" TRUE)

option(GINKGO_FORCE_GPU_AWARE_MPI "Build Ginkgo using device aware MPI" TRUE)
option(GINKGO_BUILD_CUDA "Build Ginkgo with cuda backend" FALSE)
option(GINKGO_BUILD_HIP "Build Ginkgo with hip backend" FALSE)
option(GINKGO_BUILD_SYCL "Build Ginkgo with sycl backend" FALSE)
option(GINKGO_BUILD_OMP "Build Ginkgo with omp backend" FALSE)

set(GINKGO_BUILD_MPI
    ON
    CACHE INTERNAL "")
set(GINKGO_BUILD_TESTS
    OFF
    CACHE INTERNAL "")
set(GINKGO_BUILD_EXAMPLES
    OFF
    CACHE INTERNAL "")
set(GINKGO_BUILD_BENCHMARKS
    OFF
    CACHE INTERNAL "")

if((NOT OGL_ALLOW_REFERENCE_ONLY) AND (NOT OGL_USE_EXTERNAL_GINKGO))
  if((NOT GINKGO_BUILD_CUDA)
     AND (NOT GINKGO_BUILD_HIP)
     AND (NOT GINKGO_BUILD_SYCL))
    message(
      FATAL_ERROR
        "No GPU Backend was selected, set either:\n\
-DGINKGO_BUILD_CUDA=True\n\
-DGINKGO_BUILD_HIP=True\n\
-DGINKGO_BUILD_SYCL=True\n\
or set:\n\
-DOGL_ALLOW_REFERENCE_ONLY=True\n\
to turn off this check")
  endif()
endif()

set(GINKGO_CHECKOUT_VERSION
    "ogl_rebase_1.8.0"
    CACHE STRING "Use specific version of ginkgo")

if(OGL_DEVEL_TOOLS)
  find_program(PRE_COMMIT pre-commit)
  if(NOT PRE_COMMIT)
    message(
      FATAL_ERROR
        "The pre-commit command was not found. "
        "It is necessary if you want to commit changes to OGL. "
        "If that is not the case, set OGL_DEVEL_TOOLS=OFF. "
        "Otherwise install pre-commit via pipx (or pip) using:\n"
        "    pipx install pre-commit")
  endif()

  execute_process(
    COMMAND "${PRE_COMMIT}" "install"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE pre-commit-result
    OUTPUT_VARIABLE pre-commit-output
    ERROR_VARIABLE pre-commit-error)
  if(pre-commit-result)
    message(
      FATAL_ERROR
        "Failed to install the git hooks via pre-commit."
        "Please check the error message:\n"
        "${pre-commit-output}\n${pre-commit-error}")
  endif()
  if(pre-commit-output MATCHES "^Running in migration mode with existing hooks")
    message(
      WARNING
        "An existing git hook was encountered during `pre-commit install`."
        "The old git hook will also be executed."
        "Consider removing it with `pre-commit install -f`")
  elseif(NOT pre-commit-output MATCHES "^pre-commit installed at")
    message(WARNING "`pre-commit install` did not exit normally."
                    "Please check the output message:\n" "${pre-commit-output}")
  endif()

  add_custom_target(
    pre-commit
    COMMENT "execute precommit"
    COMMAND bash -c "${PRE_COMMIT} run"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM)
endif()

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

include(cmake/package_helpers.cmake)
include(cmake/build_helpers.cmake)
include(cmake/install_helpers.cmake)
include(cmake/Sanitizer.cmake)
include(cmake/OpenFOAM.cmake)

ginkgo_find_package(Ginkgo "Ginkgo::ginkgo" FALSE 1.8.0)
include(cmake/ginkgo.cmake)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -ggdb")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
  -Wall -Wpedantic -Wextra -march=native -fopenmp -Wno-undefined-var-template")

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  message("Compiling with clang")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
    -fno-omit-frame-pointer")

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
      --fopenmp \
      -fprofile-arcs \
      -ggdb3 \
      ")

  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} \
      -ffast-math \
      -march=native  \
      -ftree-vectorize \
      -rdynamic \
      ")
endif()

add_library(OGL "")

target_compile_definitions(OGL PUBLIC NoRepository)
target_compile_definitions(OGL PUBLIC WM_LABEL_SIZE=$ENV{WM_LABEL_SIZE}
                                      WM_$ENV{WM_PRECISION_OPTION})

if(APPLE)
  message("Compiling on MacOS")
  target_link_options(OGL PUBLIC "LINKER:-undefined,dynamic_lookup")
  target_compile_definitions(OGL PRIVATE APPLE arm64)
else()
  target_compile_definitions(OGL PRIVATE linux64)
  target_link_libraries(OGL PRIVATE stdc++fs)
endif()

# Add a custom command that produces version.cpp, plus a dummy output that's not
# actually produced, in order to always force re-run version.cmake before the
# build
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.C
         ${CMAKE_CURRENT_BINARY_DIR}/_version.C
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
          ${GINKGO_CHECKOUT_VERSION}
  COMMENT "Create a version file")

add_subdirectory(include)

target_sources(
  OGL
  PRIVATE src/common.C
          src/CommunicationPattern.C
          src/Preconditioner.C
          src/lduLduBase.C
          src/StoppingCriterion.C
          src/BaseWrapper/lduBase/GKOlduBase.C
          src/BaseWrapper/CoupledLduBase/GKOCoupledLduBase.C
          ${CMAKE_CURRENT_BINARY_DIR}/version.C
          src/DevicePersistent/Base/Base.C
          src/DevicePersistent/Partition/Partition.C
          src/DevicePersistent/Vector/Vector.C
          src/DevicePersistent/DeviceIdGuard/DeviceIdGuard.C
          src/DevicePersistent/ExecutorHandler/ExecutorHandler.C
          src/MatrixWrapper/HostMatrix.C
          src/MatrixWrapper/SparsityPattern.C
          src/Repartitioner.C
	  # src/Solver/CG/GKOCG.C
	  # src/Solver/BiCGStab/GKOBiCGStab.C
	  # src/Solver/GMRES/GKOGMRES.C
	  )

enable_sanitizers(
  OGL ${OGL_ENABLE_SANITIZE_ADDRESS} ${OGL_ENABLE_SANITIZE_LEAK}
  ${OGL_ENABLE_SANITIZE_UB} ${OGL_ENABLE_SANITIZE_THREAD}
  ${OGL_ENABLE_SANITIZE_MEMORY})

target_link_libraries(
  OGL
  PUBLIC OGL_public_api
  PRIVATE OpenFOAM Ginkgo::ginkgo)

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
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "a target to format cmake files")

install(
  TARGETS OGL
          ginkgo
          ginkgo_device
          ginkgo_hip
          ginkgo_cuda
          ginkgo_omp
          ginkgo_dpcpp
          ginkgo_reference
  DESTINATION $ENV{FOAM_USER_LIBBIN})
