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

#ifdef GINKGO_BUILD_CUDA
#include "nvToolsExt.h"
#endif

#include <string.h>
#include <ginkgo/ginkgo.hpp>

namespace Foam {


#define __FILENAME__ \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define SIMPLE_LOG(VERBOSE, PRIO, MSG, MASTER_ONLY)                         \
    if (VERBOSE > PRIO) {                                                   \
        if (Pstream::parRun()) {                                            \
            if (MASTER_ONLY == 1) {                                         \
                if (Pstream::myProcNo() == 0) {                             \
                    std::cout << "[OGL LOG][" << __FILENAME__ << ":"        \
                              << __LINE__ << "] " << MSG << std::endl;      \
                }                                                           \
            } else {                                                        \
                std::cout << "[OGL LOG]" << Pstream::myProcNo() << "]["     \
                          << __FILENAME__ << ":" << __LINE__ << "] " << MSG \
                          << std::endl;                                     \
            }                                                               \
        } else {                                                            \
            std::cout << "[OGL LOG][" << __FILENAME__ << "] " << MSG        \
                      << std::endl;                                         \
        }                                                                   \
    }


#define IN_USE_DO_NOT_REMOVE(FUNC)                                 \
    std::cout << FUNC << " in " << __FILENAME__ << ":" << __LINE__ \
              << " is in use do not remove" << std::endl

#define LOG_0(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 0, MSG, 0)
#define LOG_1(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 1, MSG, 0)
#define LOG_2(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 2, MSG, 0)
#define MLOG_0(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 0, MSG, 1)
#define MLOG_1(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 1, MSG, 1)
#define MLOG_2(VERBOSE, MSG) SIMPLE_LOG(VERBOSE, 2, MSG, 1)

#ifdef GINKGO_BUILD_CUDA
#define START_ANNOTATE(NAME) nvtxRangePushA(#NAME);
#else
#define START_ANNOTATE(NAME)
#endif

#ifdef GINKGO_BUILD_CUDA
#define END_ANNOTATE(NAME) nvtxRangePop();
#else
#define END_ANNOTATE(NAME)
#endif


#define TIME_WITH_FIELDNAME(VERBOSE, NAME, FIELD, F)                          \
    START_ANNOTATE(NAME);                                                     \
    auto start_##NAME = std::chrono::steady_clock::now();                     \
    F auto end_##NAME = std::chrono::steady_clock::now();                     \
    END_ANNOTATE(NAME);                                                       \
    auto delta_t_##NAME =                                                     \
        std::chrono::duration_cast<std::chrono::microseconds>(end_##NAME -    \
                                                              start_##NAME)   \
            .count();                                                         \
    if (VERBOSE > 0) {                                                        \
        if (Pstream::parRun()) {                                              \
            if (Pstream::myProcNo() == 0 || VERBOSE > 1) {                    \
                std::cout << "[OGL LOG][Proc: " << Pstream::myProcNo() << "]" \
                          << FIELD << ": " #NAME ": "                         \
                          << delta_t_##NAME / 1000.0 << " [ms]\n";            \
            }                                                                 \
        } else {                                                              \
            std::cout << "[OGL LOG] " << FIELD << " " #NAME ": "              \
                      << delta_t_##NAME / 1000.0 << " [ms]\n";                \
        }                                                                     \
    }

#define SIMPLE_TIME(VERBOSE, NAME, F) TIME_WITH_FIELDNAME(VERBOSE, NAME, "", F)

#define UNUSED(x) (void)(x)

#define OGL_NOT_IMPLEMENTED GKO_NOT_IMPLEMENTED

std::ostream &operator<<(
    std::ostream &os,
    const std::shared_ptr<gko::matrix::Dense<scalar>> array_in);

void export_system(const word fieldName, const gko::matrix::Csr<scalar> *A,
                   const gko::matrix::Dense<scalar> *x,
                   const gko::matrix::Dense<scalar> *b, const word time);

void export_vec(const word fieldName, const gko::matrix::Dense<scalar> *x,
                const word time);

void export_mtx(const word fieldName, std::shared_ptr<const gko::LinOp> A,
                const word local, const objectRegistry &db,
                const word matrixFormat);

void set_solve_prev_iters(const word sys_matrix_name, const objectRegistry &db,
                          label prev_solve_iters, const bool is_final);

label get_solve_prev_iters(const word sys_matrix_name, const objectRegistry &db,
                           const bool is_final);

void set_solve_prev_rel_res_cost(const word sys_matrix_name,
                                 const objectRegistry &db,
                                 scalar prev_prev_rel_res_cost);

scalar get_solve_prev_rel_res_cost(const word sys_matrix_name,
                                   const objectRegistry &db);

void set_next_caching(word sys_matrix_name, const objectRegistry &db,
                      label caching);

label get_next_caching(word sys_matrix_name, const objectRegistry &db);
}  // namespace Foam

#endif
