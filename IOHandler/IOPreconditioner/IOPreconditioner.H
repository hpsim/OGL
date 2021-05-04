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
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::IOSortingIdxHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    IOSortingIdxHandler.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOPreconditioner_INCLUDED_H
#define OGL_IOPreconditioner_INCLUDED_H

#include <ginkgo/ginkgo.hpp>
#include "fvCFD.H"
#include "regIOobject.H"

#include "../IOPtr/IOPtr.H"


namespace Foam {
class IOPreconditioner {
    using bj = gko::preconditioner::Jacobi<>;

private:
    const word name_;

    const label maxBlockSize_;

    const word preconditioner_db_name_;

    const bool preconditioner_stored_;

    mutable GKOBJIOPtr *io_precond_ptr_ = NULL;

public:
    IOPreconditioner(const objectRegistry &db, const dictionary &controlDict,
                     const word fieldName)
        : name_(controlDict.lookupOrDefault("preconditioner", word("none"))),
          maxBlockSize_(controlDict.lookupOrDefault("maxBlockSize", label(16))),
          preconditioner_db_name_("preconditioner_" + fieldName),
          preconditioner_stored_(
              db.foundObject<regIOobject>(preconditioner_db_name_)){};

    void init_preconditioner(const objectRegistry &db,
                             std::shared_ptr<mtx> gkomatrix,
                             std::shared_ptr<gko::Executor> device_exec) const
    {
        if (preconditioner_stored_) {
            std::cout << " found preconditioner " << std::endl;
            io_precond_ptr_ =
                &db.lookupObjectRef<GKOBJIOPtr>(preconditioner_db_name_);
            return;
        }
        if (name_ == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            auto bj_factory = bj::build()
                                  .with_skip_sorting(true)
                                  .with_max_block_size(maxBlockSize_)
                                  .on(device_exec);
            auto bj_generated = gko::share(bj_factory->generate(gkomatrix));
            const fileName path = preconditioner_db_name_;
            io_precond_ptr_ = new GKOBJIOPtr(IOobject(path, db), bj_generated);
        }
    };

    std::shared_ptr<bj> get_preconditioner() const
    {
        if (io_precond_ptr_ == NULL) {
            std::cout << " get_precond == NULL" << std::endl;
            return std::shared_ptr<bj>{};
        }
        return io_precond_ptr_->get_ptr();
    }
};
}  // namespace Foam

#endif
