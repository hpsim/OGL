# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

ogl_unit_test_case(HostMatrix l2d)
ogl_unit_test_case(HostMatrix p2d)
ogl_unit_test_case(HostMatrix channels1x2x2)
ogl_unit_test_case(HostMatrix p2ds2x2)

ogl_unit_test_case(Distributed l2d)
#
# ogl_unit_test_case(Distributed p2d)
#
# ogl_unit_test_case(Distributed p2ds2x2)

# Copy the data files to the execution directory
configure_file(data/A_sparse.mtx data/A_sparse.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/L.mtx data/L.mtx COPYONLY)
configure_file(data/D.mtx data/D.mtx COPYONLY)
configure_file(data/U.mtx data/U.mtx COPYONLY)
ogl_unit_test(Combination)
