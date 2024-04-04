# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

option(
  INSTALL_GTEST
  "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)"
  OFF)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(matrixConversion "test_HostMatrix.C")

find_package(MPI REQUIRED)

target_include_directories(
    matrixConversion SYSTEM
  PUBLIC $ENV{FOAM_SRC}/finiteVolume/lnInclude
         $ENV{FOAM_SRC}/meshTools/lnInclude $ENV{FOAM_SRC}/OpenFOAM/lnInclude
         $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(matrixConversion PUBLIC WM_LABEL_SIZE=32 WM_ARCH_OPTION=64
                              NoRepository WM_DP)

target_link_libraries(
  matrixConversion
  PUBLIC GTest::gtest_main
         $ENV{FOAM_LIBBIN}/libOpenFOAM.so
         $ENV{FOAM_LIBBIN}/libfiniteVolume.so
         $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.so
         OGL
         Ginkgo::ginkgo
         # MPI::MPI_CXX
         ${CMAKE_DL_LIBS})

include(GoogleTest)
gtest_discover_tests(matrixConversion)
