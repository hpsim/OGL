# SPDX-License-Identifier: Unlicense
#
# SPDX-FileCopyrightText: 2024 OGL authors

add_library(OGL_public_api INTERFACE) # dummy target
set_property(
  TARGET OGL_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set_property(
  TARGET OGL_public_api APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                        "${CMAKE_CURRENT_BINARY_DIR}")
