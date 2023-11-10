# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

include_directories(
    ${googletest_SOURCE_DIR}/googletest/include
)

add_executable(
  core_test
  "test_hello_world.cpp"
)

# Your-external "mylib", add GLOBAL if the imported library is located in directories above the current.
add_library( OpenFOAM SHARED IMPORTED )
# You can define two import-locations: one for debug and one for release.
set_target_properties( OpenFOAM PROPERTIES IMPORTED_LOCATION $ENV{FOAM_LIBBIN}/libOpenFOAM.so)


target_link_libraries(
  core_test
  GTest::gtest_main OGL OpenFOAM
)

include(GoogleTest)
gtest_discover_tests(core_test)

