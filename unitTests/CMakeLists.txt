# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

option(INSTALL_GTEST "Enable installation of googletest.\
(Projects embedding googletest may want to turn this OFF.)" OFF)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
find_package(MPI REQUIRED)

macro(ogl_unit_test TEST FILE)
  add_executable(${TEST} ${FILE})
  target_link_libraries(
    ${TEST}
    OpenFOAM
    GTest::gtest_main
    OGL
    Ginkgo::ginkgo
    ginkgo_hip
    ${CMAKE_DL_LIBS})

  add_test(
    NAME ${TEST}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/unitTests/case
    COMMAND mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/${TEST} -parallel)
endmacro()

ogl_unit_test(CommunicationPattern "CommunicationPattern.C")

add_subdirectory(MatrixWrapper)
