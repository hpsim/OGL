# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
# SPDX-FileCopyrightText: 2026 OGL authors

message(STATUS "Auto detecting accelerator devices")
include(CheckLanguage)

if(NOT DEFINED OGL_BUILD_CUDA)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    set(OGL_BUILD_CUDA
        ON
        CACHE INTERNAL "")
  else()
    set(OGL_BUILD_CUDA
        OFF
        CACHE INTERNAL "")
  endif()
else()
  message(STATUS "Skip CUDA detection OGL_BUILD_CUDA=${OGL_BUILD_CUDA}")
endif()

if(NOT DEFINED OGL_BUILD_HIP)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    set(OGL_BUILD_HIP
        ON
        CACHE INTERNAL "")
  else()
    set(OGL_BUILD_HIP
        OFF
        CACHE INTERNAL "")
  endif()
else()
  message(STATUS "Skip HIP detection OGL_BUILD_HIP=${OGL_BUILD_HIP}")
endif()
