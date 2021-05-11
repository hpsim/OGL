# TODO pass flags from main CMAKE file
ginkgo_load_git_package(ginkgo_external
  "https://github.com/ginkgo-project/ginkgo.git"
  "develop"
  "-DGINKGO_BUILD_CUDA=off" "-DGINKGO_BUILD_HIP=off"
  "-DGINKGO_BUILD_OMP=off" "-DGINKGO_BUILD_TESTS=off"
  "-DGINKGO_BUILD_EXAMPLES=off" "-DGINKGO_BUILD_BENCHMARKS=off"
  "-DGINKGO_BUILD_REFERENCE=on" "-DCMAKE_BUILD_TYPE=Release"
  "-DGINKGO_BUILD_HWLOC=off"
  "-DCMAKE_INSTALL_PREFIX=$ENV{FOAM_USER_LIBBIN}/.." "-DBUILD_SHARED_LIBS=on"
  )
