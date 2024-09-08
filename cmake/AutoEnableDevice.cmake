# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(CheckLanguage)

if(NOT DEFINED GINKGO_ENABLE_CUDA)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    set(GINKGO_BUILD_CUDA
        ON
        CACHE INTERNAL "")
  else()
    set(GINKGO_BUILD_CUDA
        OFF
        CACHE INTERNAL "")
  endif()
endif()

if(NOT DEFINED GINKGO_ENABLE_HIP)
  check_language(HIP)

  if(CMAKE_HIP_COMPILER)
    set(GINKGO_BUILD_HIP
        ON
        CACHE INTERNAL "")
  else()
    set(GINKGO_BUILD_HIP
        OFF
        CACHE INTERNAL "")
  endif()
endif()
