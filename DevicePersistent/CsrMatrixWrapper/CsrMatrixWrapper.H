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

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOGKOMatrixHandler_INCLUDED_H
#define OGL_IOGKOMatrixHandler_INCLUDED_H

#include "fvCFD.H"

#include "../DevicePersistentArray/DevicePersistentArray.H"
#include "../DevicePersistentVector/DevicePersistentVector.H"
#include "../ExecutorHandler/ExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

namespace Foam {


struct CsrInitFunctor {
    const gkoGlobalIndex &global_index_;

    using mtx = gko::matrix::Csr<scalar>;

    const PersistentArray<label> &col_idxs_;

    const PersistentArray<label> &row_idxs_;

    const PersistentArray<scalar> &values_;

    const label verbose_;

    CsrInitFunctor(const gkoGlobalIndex &global_index,
                   const PersistentArray<label> &col_idxs,
                   const PersistentArray<label> &row_idxs,
                   const PersistentArray<scalar> &values, const label verbose)
        : global_index_(global_index),
          col_idxs_(col_idxs),
          row_idxs_(row_idxs),
          values_(values),
          verbose_(verbose)
    {}

    void update(std::shared_ptr<mtx> &csr_matrix) const;

    std::shared_ptr<mtx> init() const;
};


class CsrMatrixWrapper {
private:
    using mtx = gko::matrix::Csr<scalar>;

    const label verbose_;

    const bool export_;

    mutable PersistentBase<mtx, CsrInitFunctor> gkomatrix_;

    mutable label prev_solve_iters_ = 0;


public:
    CsrMatrixWrapper(const objectRegistry &db, const PersistentExecutor &exec,
                     const PersistentArray<label> &col_idxs,
                     const PersistentArray<label> &row_idxs,
                     const PersistentArray<scalar> &values,
                     const dictionary &controlDict, const word sys_matrix_name,
                     const gkoGlobalIndex &globalIndex, const label verbose)
        : verbose_(verbose),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          gkomatrix_{
              sys_matrix_name + "csr", db,
              CsrInitFunctor(globalIndex, col_idxs, row_idxs, values, verbose_),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    PersistentBase<mtx, CsrInitFunctor> &get_global_gkomatrix() const
    {
        return gkomatrix_;
    }

    PersistentBase<mtx, CsrInitFunctor> &get_local_gkomatrix() const
    {
        return gkomatrix_;
    }


    bool get_export() const { return export_; }
};

}  // namespace Foam

#endif
