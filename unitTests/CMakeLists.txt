# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

option(INSTALL_GTEST "Enable installation of googletest.\
(Projects embedding googletest may want to turn this OFF.)" OFF)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
find_package(MPI REQUIRED)

# generate unit test based on a test case
macro(ogl_unit_test_case TEST CASE)
  add_executable(${TEST}_${CASE} ${TEST}_${CASE}.cpp)
  target_link_libraries(
    ${TEST}_${CASE}
    OpenFOAM
    GTest::gtest_main
    GTest::gtest
    GTest::gmock
    OGL
    Ginkgo::ginkgo
    ginkgo_device
    ginkgo_hip
    ginkgo_reference
    ${CMAKE_DL_LIBS})

  add_test(
    NAME ${TEST}_${CASE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/unitTests/${CASE}
    COMMAND mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/${TEST}_${CASE} -parallel)
endmacro()

# generate default unit test
macro(ogl_unit_test TEST)
  add_executable(${TEST} ${TEST}.cpp)
  target_link_libraries(
    ${TEST}
    OpenFOAM
    GTest::gtest_main
    GTest::gtest
    GTest::gmock
    OGL
    Ginkgo::ginkgo
    ginkgo_device
    ginkgo_hip
    ginkgo_reference
    ${CMAKE_DL_LIBS})

  add_test(
    NAME ${TEST}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST})
endmacro()

ogl_unit_test_case(CommunicationPattern l2d)
ogl_unit_test_case(Repartitioner1D l2d)
ogl_unit_test_case(Repartitioner2D l2d)

add_subdirectory(MatrixWrapper)
