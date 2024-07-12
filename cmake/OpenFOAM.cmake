# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(OpenFOAM::core SHARED IMPORTED)
add_library(OpenFOAM::meshtools SHARED IMPORTED)
add_library(OpenFOAM::finiteVolume SHARED IMPORTED)
add_library(OpenFOAM::Pstream SHARED IMPORTED)

target_include_directories(
  OpenFOAM::core
  PUBLIC
  INTERFACE $ENV{FOAM_SRC}/finiteVolume/lnInclude
            $ENV{FOAM_SRC}/meshTools/lnInclude
            $ENV{FOAM_SRC}/OpenFOAM/lnInclude
            $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude)
if(APPLE)
  set_target_properties(
    OpenFOAM::core PROPERTIES IMPORTED_LOCATION
                              $ENV{FOAM_LIBBIN}/libOpenFOAM.dylib)
  set_target_properties(
    OpenFOAM::finiteVolume PROPERTIES IMPORTED_LOCATION
                                      $ENV{FOAM_LIBBIN}/libfiniteVolume.dylib)
  set_target_properties(
    OpenFOAM::Pstream
    PROPERTIES IMPORTED_LOCATION
               $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.dylib)
else()
  set_target_properties(
    OpenFOAM::core PROPERTIES IMPORTED_LOCATION
                              $ENV{FOAM_LIBBIN}/libOpenFOAM.so)
  set_target_properties(
    OpenFOAM::finiteVolume PROPERTIES IMPORTED_LOCATION
                                      $ENV{FOAM_LIBBIN}/libfiniteVolume.so)
  set_target_properties(
    OpenFOAM::Pstream PROPERTIES IMPORTED_LOCATION
                                 $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.so)
  set_target_properties(
    OpenFOAM::meshtools PROPERTIES IMPORTED_LOCATION
                                   $ENV{FOAM_LIBBIN}/libmeshTools.so)
endif()

add_library(OpenFOAM INTERFACE)
target_link_libraries(
  OpenFOAM
  PUBLIC
  INTERFACE OpenFOAM::core OpenFOAM::finiteVolume OpenFOAM::Pstream
            OpenFOAM::meshtools)
