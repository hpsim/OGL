# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Copy the data files to the execution directory
configure_file(data/A_sparse.mtx data/A_sparse.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/L.mtx data/L.mtx COPYONLY)
configure_file(data/D.mtx data/D.mtx COPYONLY)
configure_file(data/U.mtx data/U.mtx COPYONLY)

ogl_unit_test(matrixConversion "HostMatrix.C")
ogl_unit_test(combination "Combination.C")

# Set working directory for the unit test suite of Combination Matrix
set_tests_properties(combination PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
