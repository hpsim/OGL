add_library(OpenFOAM SHARED IMPORTED)

target_include_directories(
  OpenFOAM
  INTERFACE $ENV{FOAM_SRC}/finiteVolume/lnInclude
            $ENV{FOAM_SRC}/meshTools/lnInclude
            $ENV{FOAM_SRC}/OpenFOAM/lnInclude
            $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude)
if(APPLE)
  set_target_properties(
    OpenFOAM
    PROPERTIES IMPORTED_LOCATION $ENV{FOAM_LIBBIN}/libOpenFOAM.dylib
               $ENV{FOAM_LIBBIN}/libfiniteVolume.dylib
               $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.dylib)
else()
  set_target_properties(
    OpenFOAM
    PROPERTIES IMPORTED_LOCATION $ENV{FOAM_LIBBIN}/libOpenFOAM.so
               $ENV{FOAM_LIBBIN}/libfiniteVolume.so
               $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}/libPstream.so)
endif()
