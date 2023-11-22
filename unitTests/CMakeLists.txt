# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
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

target_link_libraries(
  matrixConversion
  PUBLIC GTest::gtest_main
         OGL
         $ENV{FOAM_LIBBIN}/libOpenFOAM.so
         $ENV{FOAM_LIBBIN}/libfiniteVolume.so
         $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.so
         $ENV{FOAM_LIBBIN}/libOSspecific.o
         ${CMAKE_DL_LIBS})

include(GoogleTest)
gtest_discover_tests(matrixConversion)
