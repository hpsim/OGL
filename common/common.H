/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OGL is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.


Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    common.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_COMMON_H
#define OGL_COMMON_H

#include "fvCFD.H"
#include "regIOobject.H"


// #include "../IOGKOMatrixHandler/IOGKOMatrixHandler.H"
// #include "../IOGlobalIndex/gkoGlobalIndex.H"
// #include "../IOHandler/IOPreconditioner/IOPreconditioner.H"
// #include "../IOHandler/IOSortingIdxHandler/IOSortingIdxHandler.H"
// #include "../common/StoppingCriterion.H"

#include <ginkgo/ginkgo.hpp>

namespace Foam {
using mtx = gko::matrix::Csr<scalar>;
using vec = gko::matrix::Dense<scalar>;

#define SIMPLE_LOG(VERBOSE, MSG)                                          \
    if (VERBOSE) {                                                        \
        if (Pstream::parRun()) {                                          \
            std::cout << "[OGL LOG][Proc: " << Pstream::myProcNo() << "]" \
                      << MSG << std::endl;                                \
        } else {                                                          \
            std::cout << "[OGL LOG] " << MSG << std::endl;                \
        }                                                                 \
    }

#define SIMPLE_TIME(VERBOSE, NAME, F)                                     \
    auto start_##NAME = std::chrono::steady_clock::now();                 \
    F auto end_##NAME = std::chrono::steady_clock::now();                 \
    if (VERBOSE) {                                                        \
        if (Pstream::parRun()) {                                          \
            std::cout                                                     \
                << "[OGL LOG][Proc: " << Pstream::myProcNo()              \
                << "]" #NAME " : "                                        \
                                                                          \
                << std::chrono::duration_cast<std::chrono::microseconds>( \
                       end_##NAME - start_##NAME)                         \
                       .count()                                           \
                << " mu s\n";                                             \
        } else {                                                          \
            std::cout                                                     \
                << "[OGL LOG] " #NAME " : "                               \
                << std::chrono::duration_cast<std::chrono::microseconds>( \
                       end_##NAME - start_##NAME)                         \
                       .count()                                           \
                << " mu s\n";                                             \
        }                                                                 \
    }


void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b, const word time);

void export_vec(const word fieldName, const vec *x, const word time);

void set_solve_prev_iters(const word sys_matrix_name, const objectRegistry &db,
                          label prev_solve_iters);

label get_solve_prev_iters(const word sys_matrix_name,
                           const objectRegistry &db);
}  // namespace Foam

#endif
