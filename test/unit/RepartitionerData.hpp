// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <vector>

/* 1d case
 * The mesh has the following structure
 * ranks:    0     1     2     3
 * cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
 * cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- local row ids
 */

std::map<std::string, int> exp_size{{"1d", 9}, {"2d", 8}};
