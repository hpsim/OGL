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
#include "regIOobject.H"

#include "../../HostMatrix/HostMatrix.H"
#include "../DevicePersistentBase/DevicePersistentBase.H"
#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

namespace Foam {


struct CsrInitFunctor {
    const gkoGlobalIndex &global_index_;

    using mtx = gko::matrix::Csr<scalar>;

    using coo_mtx = gko::matrix::Coo<scalar>;

    using val_array = gko::Array<scalar>;

    const PersistentArray<label> &col_idxs_;

    const PersistentArray<label> &row_idxs_;

    const PersistentArray<scalar> &values_;

    const bool verbose_;

    CsrInitFunctor(const gkoGlobalIndex &global_index,
                   const PersistentArray<label> &col_idxs,
                   const PersistentArray<label> &row_idxs,
                   const PersistentArray<scalar> &values, const bool verbose)
        : global_index_(global_index),
          col_idxs_(col_idxs),
          row_idxs_(row_idxs),
          values_(values),
          verbose_(verbose)
    {}

    void update(std::shared_ptr<mtx> &csr_matrix) const
    {
        if (Pstream::parRun()) {
            if (Pstream::master()) {
                auto values_view = val_array::view(values_.get_device_exec(),
                                                   values_.get_global_size(),
                                                   csr_matrix->get_values());
                values_view = *values_.get_array().get();
            }
        } else {
            auto values_view = val_array::view(values_.get_device_exec(),
                                               values_.get_global_size(),
                                               csr_matrix->get_values());
            // copy values to device
            values_view = *values_.get_array().get();
        }
    }

    std::shared_ptr<mtx> init() const
    {
        if (Pstream::parRun()) {
            const auto device_exec = values_.get_device_exec();
            const label nCells = global_index_.size();
            word msg{"init global csr matrix of size " +
                     std::to_string(nCells)};
            SIMPLE_LOG(verbose_, msg)

            auto values = values_.get_global_array();
            auto cols = col_idxs_.get_global_array();
            auto rows = row_idxs_.get_global_array();

            if (Pstream::master()) {
                auto coo_mtx = gko::share(
                    coo_mtx::create(device_exec, gko::dim<2>(nCells, nCells),
                                    val_array(device_exec, *values.get()),
                                    *cols.get(), *rows.get()));

                auto gkomatrix = gko::share(
                    mtx::create(device_exec, gko::dim<2>(nCells, nCells)));
                SIMPLE_TIME(verbose_, convert_coo_to_csr,
                            coo_mtx->convert_to(gkomatrix.get());)
                return gkomatrix;
            } else {
                return {};
            }
        } else {
            const auto device_exec = values_.get_device_exec();
            const label nCells = global_index_.size();
            word msg{"init csr matrix of size " + std::to_string(nCells)};
            SIMPLE_LOG(verbose_, msg)

            auto coo_mtx = gko::share(coo_mtx::create(
                device_exec, gko::dim<2>(nCells, nCells),
                val_array(device_exec, *values_.get_array().get()),
                *col_idxs_.get_array().get(), *row_idxs_.get_array().get()));

            auto gkomatrix = gko::share(
                mtx::create(device_exec, gko::dim<2>(nCells, nCells)));

            SIMPLE_TIME(verbose_, convert_coo_to_csr,
                        coo_mtx->convert_to(gkomatrix.get());)
            return gkomatrix;
        }
    }
};


class CsrMatrixWrapper {
private:
    using mtx = gko::matrix::Csr<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;

    const bool verbose_;

    const bool export_;

    mutable PersistentBase<mtx, CsrInitFunctor> gkomatrix_;

    mutable label prev_solve_iters_ = 0;


public:
    CsrMatrixWrapper(const objectRegistry &db, const PersistentExecutor &exec,
                     const PersistentArray<label> &col_idxs,
                     const PersistentArray<label> &row_idxs,
                     const PersistentArray<scalar> &values,
                     const dictionary &controlDict, const word sys_matrix_name,
                     const gkoGlobalIndex &globalIndex, const bool verbose)
        : verbose_(verbose),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          gkomatrix_{
              sys_matrix_name + "csr",
              db,
              exec,
              CsrInitFunctor(globalIndex, col_idxs, row_idxs, values, verbose_),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    PersistentBase<mtx, CsrInitFunctor> &get_global_gkomatrix() const
    {
        return gkomatrix_;
    }

    PersistentBase<mtx, CsrInitFunctor> &get_gkomatrix() const
    {
        return gkomatrix_;
    }


    bool get_export() const { return export_; }
};

}  // namespace Foam

#endif
