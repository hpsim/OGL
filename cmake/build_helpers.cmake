# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function(ginkgo_default_includes name)
    # set include path depending on used interface
    target_include_directories("${name}"
        PUBLIC
            $<BUILD_INTERFACE:${NLA4HPC_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${NLA4HPC_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${NLA4HPC_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
endfunction()

function(ginkgo_compile_features name)
    target_compile_features("${name}" PUBLIC cxx_std_14)
    # Set an appropriate SONAME
    set_property(TARGET "${name}" PROPERTY
        SOVERSION "${NLA4HPC_VERSION}")
    set_target_properties("${name}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

macro(ginkgo_modify_flags name)
    # add escape before "
    # the result var is ${name}_MODIFY
    string(REPLACE "\"" "\\\"" ${name}_MODIFY "${${name}}")
endmacro()

# Extract the clang version from a clang executable path
function(ginkgo_extract_clang_version CLANG_COMPILER GINKGO_CLANG_VERSION)
    set(CLANG_VERSION_PROG "#include <cstdio>\n"
        "int main() {printf(\"%d.%d.%d\", __clang_major__, __clang_minor__, __clang_patchlevel__)\;"
        "return 0\;}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver.cpp" ${CLANG_VERSION_PROG})
    execute_process(COMMAND ${CLANG_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver
        ERROR_VARIABLE CLANG_EXTRACT_VER_ERROR)
    execute_process(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver
        OUTPUT_VARIABLE FOUND_CLANG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        )

    set (${GINKGO_CLANG_VERSION} "${FOUND_CLANG_VERSION}" PARENT_SCOPE)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver.cpp)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver)
endfunction()
