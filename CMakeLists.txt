cmake_minimum_required(VERSION 3.9)

project(OGL LANGUAGES C CXX DESCRIPTION "A wrapper for Ginkgo solver to provide GPGPU capabilities to OpenFOAM")

if(NOT DEFINED ENV{FOAM_SRC})
    message(FATAL_ERROR "You must source OpenFOAM before building OGL")
endif()

# Configuration options

include(cmake/build_type_helpers.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules/")

include(CheckIncludeFileCXX)
option(OGL_USE_EXTERNAL_GINKGO "Use external ginkgo" FALSE)

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

include(cmake/package_helpers.cmake)
include(cmake/build_helpers.cmake)
include(cmake/install_helpers.cmake)
ginkgo_find_package(Ginkgo "Ginkgo::ginkgo" FALSE 1.4.0)

add_subdirectory(third_party)

# If ginkgo wasn't found so far look for the third_party installation
if (NOT Ginkgo_FOUND)
  find_package(Ginkgo 1.4.0 REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/third_party/ginkgo/build/install/lib/cmake)
endif()

# To interface with ginkgo at least C++ 14 is needed
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
  -Wall -Wpedantic -Wextra -march=native -fopenmp -fPIC"
  )

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
      -fno-omit-frame-pointer \
      -fsanitize=address"
    )

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
    -fno-omit-frame-pointer \
    -Rpass=loop-vectorize \
      -ffast-math \
      "
      #-Rpass-analysis=loop-vectorize \
      #-Rpass-missed=loop-vectorize \
      )

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

    add_definitions(-DWITH_GNU)

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
      --coverage\
      -fprofile-arcs \
      -fsanitize=address \
      -ggdb3 \
      -ftest-coverage"
)

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -ffast-math \
        -march=native  \
        -ftree-vectorize \
        -rdynamic \
        "
        # -fopt-info-vec-missed"
        )
endif()

option(BUILD_SHARED_LIBS "Build shared library" ON)
add_library(OGL  ""  )

add_compile_definitions(
  WM_LABEL_SIZE=32
  WM_ARCH_OPTION=64
  NoRepository
  linux64
  WM_DP
  )

target_sources(OGL
  PRIVATE
common/common.C
IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.C
IOHandler/IOExecutorHandler/IOExecutorHandler.C
lduMatrix/GKOlduBase/GKOlduBase.C
lduMatrix/GKOCG/GKOCG.C
lduMatrix/GKOIR/GKOIR.C
lduMatrix/GKOBiCGStab/GKOBiCGStab.C
LduMatrix/GKOLduBase/GKOLduBase.C
LduMatrix/GKOACG/GKOACG.C
  PUBLIC
common/common.H
IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H
IOHandler/IOExecutorHandler/IOExecutorHandler.H
lduMatrix/GKOlduBase/GKOlduBase.H
lduMatrix/GKOCG/GKOCG.H
lduMatrix/GKOIR/GKOIR.H
lduMatrix/GKOBiCGStab/GKOBiCGStab.H
LduMatrix/GKOLduBase/GKOLduBase.H
LduMatrix/GKOACG/GKOACG.H
  )


target_include_directories(OGL
  PUBLIC
  $ENV{FOAM_SRC}/finiteVolume/lnInclude
  $ENV{FOAM_SRC}/meshTools/lnInclude
  $ENV{FOAM_SRC}/OpenFOAM/lnInclude
  $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude
    IOHandler/IOExecutorHandler/
    IOHandler/IOSortingIdxHandler/
    common/
    lduMatrix/GKOlduBase
    LduMatrix/GKOLduBase
  )


target_link_libraries(OGL
  PUBLIC
    Ginkgo::ginkgo
  )

install(TARGETS OGL 
    DESTINATION $ENV{FOAM_USER_LIBBIN}
    )

if(NOT ${OGL_USE_EXTERNAL_GINKGO})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party/ginkgo/build/install/lib/
    DESTINATION $ENV{FOAM_USER_LIBBIN}
    )
endif()

