# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function(ginkgo_create_test test_name)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PUBLIC Ginkgo::ginkgo GTest::Main GTest::GTest ${ARGN})
  add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_test)

function(ginkgo_create_gbench bench_name)
  find_package(Threads REQUIRED)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${bench_name}")
  add_executable(${TEST_TARGET_NAME} ${bench_name}.cpp)
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${bench_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PUBLIC Ginkgo::ginkgo GTest::Main GTest::GTest GBenchmark::GBenchmark Threads::Threads ${ARGN} ${CUDA_LIBRARIES})
  add_test(NAME ${REL_BINARY_DIR}/${bench_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_gbench)

function(ginkgo_create_thread_test test_name)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo GTest::Main GTest::GTest Threads::Threads ${ARGN})
  add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_thread_test)

function(ginkgo_create_test_cpp_cuda_header test_name)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    "${CUDA_INCLUDE_DIRS}"
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo GTest::Main GTest::GTest ${ARGN})
  add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_test_cpp_cuda_header)

function(ginkgo_create_cuda_test test_name)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cu)
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    )
 cas_target_cuda_architectures(${TEST_TARGET_NAME}
    ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
    UNSUPPORTED "20" "21")
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})

  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo GTest::Main GTest::GTest ${ARGN})
  add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_cuda_test)

function(ginkgo_create_mpi_test test_name num_mpi_procs)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    ${MPI_INCLUDE_PATH}
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  if("${GINKGO_MPI_EXEC_SUFFIX}" MATCHES ".openmpi" AND MPI_RUN_AS_ROOT)
    set(OPENMPI_RUN_AS_ROOT_FLAG "--allow-run-as-root")
  else()
    set(OPENMPI_RUN_AS_ROOT_FLAG "")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo GTest::Main GTest::GTest ${ARGN})
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(test_param ${MPIEXEC_NUMPROC_FLAG} ${num_mpi_procs} ${OPENMPI_RUN_AS_ROOT_FLAG} ${CMAKE_BINARY_DIR}/${REL_BINARY_DIR}/${test_name})
  add_test(NAME ${REL_BINARY_DIR}/${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${test_param} )
endfunction(ginkgo_create_mpi_test)

function(ginkgo_create_hip_test_special_linkage test_name)
  # use gcc to compile but use hip to link
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
  # Fix the missing metadata when building static library.
  if(GINKGO_HIP_PLATFORM MATCHES "hcc" AND NOT BUILD_SHARED_LIBS)
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
  endif()
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo GTest::Main GTest::GTest ${ARGN})
  add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_hip_test_special_linkage)

function(ginkgo_create_hip_test test_name)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")

  set_source_files_properties(${test_name}.hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

  if (HIP_VERSION GREATER_EQUAL "3.5")
    hip_add_executable(${TEST_TARGET_NAME} ${test_name}.hip.cpp
      HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
      NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
      HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS}
      CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS})
  else()
    hip_add_executable(${TEST_TARGET_NAME} ${test_name}.hip.cpp
      HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
      NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
      HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS})
  endif()
  target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)

  # Let's really not use nvcc for linking here
  if (GINKGO_HIP_PLATFORM MATCHES "nvcc")
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif()

  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    # Only `math` requires it so far, but it's much easier
    # to put these this way.
    ${GINKGO_HIP_THRUST_PATH}
    # Only `exception_helpers` requires this so far, but it's much easier
    # to put these this way.
    ${HIPBLAS_INCLUDE_DIRS}
    ${HIPSPARSE_INCLUDE_DIRS}
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})

  # Pass in the `--amdgpu-target` flags if asked
  if(GINKGO_HIP_AMDGPU AND GINKGO_HIP_PLATFORM MATCHES "hcc")
    foreach(target ${GINKGO_HIP_AMDGPU})
      target_link_libraries(${TEST_TARGET_NAME} PRIVATE --amdgpu-target=${target})
    endforeach()
  endif()

  # GINKGO_RPATH_FOR_HIP needs to be populated before calling this for the linker to include
  # our libraries path into the executable's runpath.
  if(BUILD_SHARED_LIBS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_RPATH_FOR_HIP}")

    if (GINKGO_CHECK_CIRCULAR_DEPS)
      target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
  endif()

  target_link_libraries(${TEST_TARGET_NAME} PRIVATE Ginkgo::ginkgo GTest::Main GTest::GTest ${ARGN})
  add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_hip_test)
