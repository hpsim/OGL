function(ginkgo_default_includes name)
    # set include path depending on used interface
    target_include_directories("${name}"
        PUBLIC
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
