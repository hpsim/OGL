// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include <map>
#include <vector>

/*
 * The mesh has the following structure
 *         global ids
 *        [24 25 26|33 34 35]
 *        [21 22 23|30 31 32]
 *   2    [18 19 20|27 28 29]  3
 *        ---------+---------
 *        [ 6  7  8|15 16 17]
 *        [ 3  4  5|12 13 14]
 *   0    [ 0  1  2| 9 10 11]  1
 *   */
std::map < std::string,
    std::map<label, vec> exp_local_size{
        {"l2d", {1, {9, 9, 9, 9}}, {2, {18, 0, 18, 0}}, {4, {36, 0, 0, 0}}}};
