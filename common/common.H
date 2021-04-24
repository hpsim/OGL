/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_COMMON_H
#define OGL_COMMON_H

#include "fvCFD.H"
#include "regIOobject.H"

#include "../IOGKOMatrixHandler/IOGKOMatrixHandler.H"

#include <ginkgo/ginkgo.hpp>
#include <map>

#define SIMPLE_TIME(NAME, F)                                            \
    auto start_##NAME = std::chrono::steady_clock::now();               \
    F auto end_##NAME = std::chrono::steady_clock::now();               \
    std::cout << "Gingko " #NAME " : "                                  \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     end_##NAME - start_##NAME)                         \
                     .count()                                           \
              << " ms\n";

namespace Foam {

using vec = gko::matrix::Dense<scalar>;
using mtx = gko::matrix::Coo<scalar>;
using idx_array = gko::Array<label>;

void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b);


typedef IOPtr<idx_array> GKOIDXIOPtr;
typedef IOPtr<gko::matrix::Coo<scalar>> GKOCOOIOPtr;

// Logs the number of iteration executed
struct IterationLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp *,
                               const gko::size_type &num_iterations,
                               const gko::LinOp *residual,
                               const gko::LinOp *res_norm,
                               const gko::LinOp *) const override
    {
        this->num_iters = num_iterations;
    }

    IterationLogger(std::shared_ptr<const gko::Executor> exec)
        : exec(exec),
          gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask)
    {}

    gko::size_type get_iters() { return num_iters; }

private:
    std::shared_ptr<const gko::Executor> exec;
    mutable gko::size_type num_iters{0};
};


class lduLduCommonBase {
    const objectRegistry &db_;

    const label maxIter_;

    const label minIter_;

    const scalar tolerance_;

    const scalar relTol_;

    const word update_sysMatrix_;

    const label nCells_;

    const label nNeighbours_;

    const label nElems_;

    const bool export_;

    const bool verbose_;


public:
    lduLduCommonBase(const objectRegistry &db, label nCells, label nNeighbours,
                     const dictionary &controlDict)
        : db_(db),
          maxIter_(controlDict.lookupOrDefault("maxIter", label(1000))),
          minIter_(controlDict.lookupOrDefault("minIter", label(0))),
          tolerance_(controlDict.lookupOrDefault("tolerance", scalar(1e-6))),
          relTol_(controlDict.lookupOrDefault("relTol", scalar(1e-6))),
          update_sysMatrix_(
              controlDict.lookupOrDefault("updateSysMatrix", word("yes"))),
          nCells_(nCells),
          nNeighbours_(nNeighbours),
          nElems_(nCells_ + 2 * nNeighbours_),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          verbose_(controlDict.lookupOrDefault<Switch>("verbose", false)){};

    const objectRegistry &get_db() { return db_; };

    scalar tolerance() const { return tolerance_; }

    label minIter() const { return minIter_; }

    label maxIter() const { return maxIter_; }

    label nCells() const { return nCells_; };

    label nElems() const { return nElems_; };

    label nNeighbours() const { return nNeighbours_; };

    //
    bool get_update_sys_matrix() const { return (update_sysMatrix_ == "yes"); }

    bool get_export() const { return export_; }
};
}  // namespace Foam

#endif
