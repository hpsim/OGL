# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)
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
   matrixConversion
   "test_HostMatrix.C"
)

# Your-external "mylib", add GLOBAL if the imported library is located in directories above the current.
add_library( libOpenFOAM SHARED IMPORTED )

# You can define two import-locations: one for debug and one for release.
set_target_properties(libOpenFOAM
    PROPERTIES IMPORTED_LOCATION_RELEASE $ENV{FOAM_LIBBIN}/libOpenFOAM.so
    )

add_library(libfiniteVolume SHARED IMPORTED )
set_target_properties(libfiniteVolume
    PROPERTIES IMPORTED_LOCATION_RELEASE $ENV{FOAM_LIBBIN}/libfiniteVolume.so
    )

target_link_libraries(
    matrixConversion
  PUBLIC
      GTest::gtest_main 
      OGL
      libOpenFOAM
      libfiniteVolume
)


include(GoogleTest)
gtest_discover_tests(matrixConversion)
