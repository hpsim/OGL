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
    using ic = gko::preconditioner::Ic<>;

private:
    const word name_;

    const label maxBlockSize_;

    const word preconditioner_db_name_;

    const bool preconditioner_stored_;

    mutable GKOLinOPIOPtr *io_precond_ptr_ = NULL;

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
        // TODO get field specific preconditioner
        if (preconditioner_stored_) {
            io_precond_ptr_ =
                &db.lookupObjectRef<GKOLinOPIOPtr>(preconditioner_db_name_);
            return;
        }
        const fileName path = preconditioner_db_name_;
        if (name_ == "BJ") {
            // TODO for non constant system matrix reuse block pointers
            auto pre_factory = bj::build()
                                   .with_skip_sorting(true)
                                   .with_max_block_size(maxBlockSize_)
                                   .on(device_exec);
            auto pre_generated = gko::share(pre_factory->generate(gkomatrix));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
        if (name_ == "ILU") {
            auto factorizaton_factory =
                gko::factorization::Ilu<scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto par_ilu =
                gko::share(factorizaton_factory->generate(gkomatrix));

            auto ilu_pre_factory =
                gko::preconditioner::Ilu<gko::solver::LowerTrs<scalar, label>,
                                         gko::solver::UpperTrs<scalar, label>,
                                         false>::build()
                    .on(device_exec);

            // Use incomplete factors to generate ILU preconditioner auto
            auto pre_generated =
                gko::share(ilu_pre_factory->generate(gko::share(par_ilu)));

            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
        if (name_ == "IC") {
            auto factorizaton_factory =
                gko::factorization::Ic<scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto ic = gko::share(factorizaton_factory->generate(gkomatrix));
            auto pre_factory =
                gko::preconditioner::Ic<>::build().on(device_exec);

            auto pre_generated = gko::share(pre_factory->generate(ic));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
        if (name_ == "ISAI") {
            auto pre_factory =
                gko::preconditioner::Isai<gko::preconditioner::isai_type::spd,
                                          scalar, label>::build()
                    .with_skip_sorting(true)
                    .on(device_exec);

            auto pre_generated = gko::share(pre_factory->generate(gkomatrix));
            io_precond_ptr_ =
                new GKOLinOPIOPtr(IOobject(path, db), pre_generated);
        }
    };

    std::shared_ptr<gko::LinOp> get_preconditioner() const
    {
        if (io_precond_ptr_ == NULL) {
            return std::shared_ptr<gko::LinOp>{};
        }
        return io_precond_ptr_->get_ptr();
    }
};
}  // namespace Foam

#endif
