# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git describe --tags --dirty --long --always
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(detailed_log "${PROJECT_BINARY_DIR}/detailed.log")
set(minimal_log "${PROJECT_BINARY_DIR}/minimal.log")
file(REMOVE ${detailed_log} ${minimal_log})

macro(_both)
  # Write to both log files:
  file(APPEND ${detailed_log} "${ARGN}")
  file(APPEND ${minimal_log} "${ARGN}")
endmacro()

macro(_detailed)
  # Only write to detailed.log:
  file(APPEND ${detailed_log} "${ARGN}")
endmacro()

macro(_minimal)
  # Only write to minimal.log:
  file(APPEND ${minimal_log} "${ARGN}")
endmacro()

macro(add_separator log)
  set(sep "\n+------------------------------------------------------------------------------+")
  string(APPEND ${log} ${sep})
endmacro()

macro(dump_cmake_variables regex log)
  get_cmake_property(_variableNames VARIABLES)
  list(SORT _variableNames)
  foreach(_variableName ${_variableNames})
    unset(MATCHED)

    # case sensitive match
    #
    # case insenstitive match
    string(TOLOWER "${regex}" ARGV0_lower)
    string(TOLOWER "${_variableName}" _variableName_lower)
    # string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
    string(REGEX MATCH ${ARGV0_lower} MATCHED ${_variableName_lower})

    if(NOT MATCHED)
      continue()
    endif()
    string(APPEND ${log} "\n")
    string(APPEND ${log} " -- \t${_variableName} = ${${_variableName}}")
  endforeach()
endmacro()

function(ogl_print_banner)
  set(banner
      "\n 
	    ███████      █████████  █████      
	  ███▒▒▒▒▒███   ███▒▒▒▒▒███▒▒███       
	 ███     ▒▒███ ███     ▒▒▒  ▒███       
	▒███      ▒███▒███          ▒███       
	▒███      ▒███▒███    █████ ▒███       
	▒▒███     ███ ▒▒███  ▒▒███  ▒███      █
	 ▒▒▒███████▒   ▒▒█████████  ███████████
	   ▒▒▒▒▒▒▒      ▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒ 
       -     OpenFOAM     Ginkgo     Layer
                                       
         Version: ${CMAKE_PROJECT_VERSION} Git commit: ${GIT_HASH}
")
  add_separator(log)
  string(APPEND log ${banner})
  add_separator(log)
  string(APPEND log "\n Build information:")
  string(APPEND log "\n--\tCMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
  string(APPEND log
         "\n--\tCMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER_ID}@${CMAKE_CXX_COMPILER_VERSION}")
  add_separator(log)
  string(APPEND log "\n Third-Party dependencies:")
  dump_cmake_variables("^OGL_GINKGO" log)
  add_separator(log)
  string(APPEND log "\n OGL components:")
  dump_cmake_variables("^OGL_BUILD" log)
  add_separator(log)
  string(APPEND log "\n OGL definitions:")
  dump_cmake_variables("^OGL_DEFINE" log)
  add_separator(log)
  message(STATUS ${log})
  write_file("${CMAKE_CURRENT_BINARY_DIR}/ogl_build.log" ${log})
endfunction()

ogl_print_banner()
