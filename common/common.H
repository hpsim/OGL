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

#include <string.h>
#include <ginkgo/ginkgo.hpp>

namespace Foam {
using mtx = gko::matrix::Csr<scalar>;
using vec = gko::matrix::Dense<scalar>;


#define __FILENAME__ \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define SIMPLE_LOG(VERBOSE, PRIO, MSG)                                     \
    if (VERBOSE > PRIO) {                                                  \
        if (Pstream::parRun()) {                                           \
            std::cout << "[OGL LOG][Proc: " << Pstream::myProcNo() << "][" \
                      << __FILENAME__ << ":" << __LINE__ << "] " << MSG    \
                      << std::endl;                                        \
        } else {                                                           \
            std::cout << "[OGL LOG][" << __FILENAME__ << "] " << MSG       \
                      << std::endl;                                        \
        }                                                                  \
    }

#define LOG_0(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 0, MSG)
#define LOG_1(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 1, MSG)
#define LOG_2(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 2, MSG)

#define SIMPLE_TIME(VERBOSE, NAME, F)                                     \
    auto start_##NAME = std::chrono::steady_clock::now();                 \
    F auto end_##NAME = std::chrono::steady_clock::now();                 \
    if (VERBOSE > 0) {                                                    \
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
