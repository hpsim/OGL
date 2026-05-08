# SPDX-FileCopyrightText: 2026 OGL authors
#
# SPDX-License-Identifier: Unlicense

# set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/cmake_packages)

include(cmake/Versions.cmake)
include(cmake/CPM.cmake)

if(NOT DEFINED OGL_GINKGO_DIR)
    set(OGL_GINKGO_CHECKOUT_VERSION
        "ogl_0600_gko190"
        CACHE STRING "Use specific version of ginkgo")
    message(STATUS "Using CPM to get Ginkgo ${GINKGO_CHECKOUT_VERSION}")
    set(OGL_GINKGO_VIA_CPM ON)
else()
  message(STATUS "using OGL_GINKGO_DIR ${OGL_GINKGO_DIR}")
  add_subdirectory(${OGL_GINKGO_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Ginkgo)
  unset(OGL_GINKGO_CHECKOUT_VERSION CACHE)
endif()

if(OGL_GINKGO_VIA_CPM)
  find_package(Ginkgo QUIET)
  if(Ginkgo_FOUND)
    message(STATUS "Using system-installed Ginkgo (version: ${Ginkgo_VERSION})")
    set(OGL_GINKGO_VERSION ${Ginkgo_VERSION})
  else()
    message(STATUS "System Ginkgo not found — fetching from GitHub via CPM.cmake...")
    cpmaddpackage(
      NAME
      Ginkgo
      GITHUB_REPOSITORY
      ginkgo-project/ginkgo
      GIT_TAG
      ${OGL_GINKGO_CHECKOUT_VERSION}
      SYSTEM
      YES
      OPTIONS
      "GINKGO_BUILD_TESTS OFF"
      "GINKGO_BUILD_BENCHMARKS OFF"
      "GINKGO_BUILD_EXAMPLES OFF"
      "GINKGO_BUILD_OMP OFF"
      "GINKGO_ENABLE_HALF OFF"
      "GINKGO_BUILD_MPI ON"
      "GINKGO_BUILD_PAPI_SDE OFF"
      "GINKGO_BUILD_CUDA ${OGL_BUILD_CUDA}"
      "GINKGO_BUILD_HIP ${OGL_BUILD_HIP}"
      "GINKGO_BUILD_SYCL ${OGL_BUILD_SYCL}"
      )
  endif()
endif()
