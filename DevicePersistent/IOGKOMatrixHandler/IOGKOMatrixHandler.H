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

    const label size_;
    const bool verbose_;

    CsrInitFunctor(const gkoGlobalIndex &global_index,
                   const PersistentArray<label> &col_idxs,
                   const PersistentArray<label> &row_idxs,
                   const PersistentArray<scalar> &values, const label size,
                   const bool verbose)
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

            auto coo_mtx = gko::share(
                coo_mtx::create(device_exec, gko::dim<2>(nCells, nCells),
                                val_array(device_exec, *values.get()),
                                *cols.get(), *rows.get()));

            auto gkomatrix = gko::share(
                mtx::create(device_exec, gko::dim<2>(nCells, nCells)));

            if (Pstream::master()) {
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

    // const word sys_matrix_name_;

    const bool verbose_;

    const bool export_;

    mutable PersistentBase<mtx, CsrInitFunctor> gkomatrix_;

    // const bool sys_matrix_stored_;

    // const word update_sysMatrix_;

    // const word sparsity_pattern_name_cols_;

    // const word sparsity_pattern_name_rows_;

    // const bool sparsity_pattern_stored_;

    // const word init_guess_vector_name_;

    // const bool init_guess_vector_stored_;

    // const bool update_init_guess_vector_;

    mutable label prev_solve_iters_ = 0;

    // mutable GKOCSRIOPtr *gkomatrix_ptr_ = NULL;

    // mutable GKOIDXIOPtr *io_col_idxs_ptr_ = NULL;

    // mutable GKOIDXIOPtr *io_row_idxs_ptr_ = NULL;

    // mutable std::shared_ptr<vec> init_guess_ = NULL;

    // mutable std::vector<GKOVECIOPtr *> io_init_guess_ptrs_ = {};

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

    // bool get_sys_matrix_stored() const { return sys_matrix_stored_; }

    // bool get_sparsity_pattern_stored() const
    // {
    //     return sparsity_pattern_stored_;
    // }

    // void init_device_matrix(const objectRegistry &db,
    //                         const gkoGlobalIndex &globalIndex,
    //                         std::shared_ptr<val_array> values_host,
    //                         std::shared_ptr<idx_array> col_idxs_host,
    //                         std::shared_ptr<idx_array> row_idxs_host,
    //                         const label nElems, const label nCells,
    //                         const bool update) const;
    PersistentBase<mtx, CsrInitFunctor> &get_global_gkomatrix() const
    {
        return gkomatrix_;
    }

    PersistentBase<mtx, CsrInitFunctor> &get_gkomatrix() const
    {
        return gkomatrix_;
    }

    // initialize the local initial guess
    // void init_initial_guess_vector(const Field<vector> &psi,
    //                                const objectRegistry &db,
    //                                const gkoGlobalIndex &globalIndex,
    //                                const label nCells) const
    // {
    //     std::vector<word> postFixes{"", "y", "z"};
    //     for (int i = 0; i < 3; i++) {
    //         init_initial_guess(&psi[0][i], db, globalIndex, nCells,
    //                            postFixes[i]);
    //     }
    // }


    // void init_initial_guess_vector(const Field<scalar> &psi,
    //                                const objectRegistry &db,
    //                                const gkoGlobalIndex &globalIndex,
    //                                const label nCells) const
    // {
    //     init_initial_guess(&psi[0], db, globalIndex, nCells, "");
    // }

    // void init_initial_guess(const scalar *psi, const objectRegistry &db,
    //                         const gkoGlobalIndex &globalIndex,
    //                         const label nCells, const word postFix) const;

    // void get_initial_guess(std::vector<std::shared_ptr<vec>> &init_guess)
    // const
    // {
    //     if (io_init_guess_ptrs_.size() == 0) {
    //         init_guess.push_back(init_guess_);
    //     }
    //     for (int i = 0; i < io_init_guess_ptrs_.size(); i++) {
    //         init_guess.push_back(io_init_guess_ptrs_[i]->get_ptr());
    //     }
    // }
    // const PersistentExecutor &exec;

    // TODO clean up with scalarField version
    // template <class OFField>
    // void copy_result_back(const OFField &psi, const label nCells) const
    // {
    //     std::vector<std::shared_ptr<vec>> device_xs_ptr{};
    //     get_initial_guess(device_xs_ptr);

    //     for (int i = 0; i < 3; i++) {
    //         auto host_x = vec::create(ref_exec(), gko::dim<2>(nCells, 1));
    //         host_x->copy_from(gko::lend(device_xs_ptr[i]));
    //         auto host_x_view =
    //             val_array::view(ref_exec(), nCells, host_x->get_values());
    //         auto psi_view = val_array::view(ref_exec(), nCells,
    //                                         const_cast<scalar
    //                                         *>(&psi[0][i]));
    //         psi_view = host_x_view;
    //     }
    // }

    // // TODO clean up with scalarField version
    // void copy_result_back(const gkoGlobalIndex &globalAddr,
    //                       const scalarField &local_psi) const
    // {
    //     Info << "copy result back parallel" << endl;
    //     const label globalNCells{globalAddr.size()};
    //     const label nCells{globalAddr.localSize()};

    //     // TODO can this be avoided on non master
    //     // ranks, for example by passing null ptrs?
    //     auto global_host_x = val_array(ref_exec(), globalNCells);

    //     if (Pstream::master()) {
    //         std::vector<std::shared_ptr<vec>> device_xs_ptr{};
    //         get_initial_guess(device_xs_ptr);

    //         auto device_xs_view = val_array::view(
    //             app_exec(), globalNCells, device_xs_ptr[0]->get_values());

    //         global_host_x = device_xs_view;
    //     }

    //     auto local_host_x = val_array(ref_exec(), nCells);

    //     globalAddr.scatter(global_host_x, local_host_x);

    //     auto psi_view = val_array::view(ref_exec(), nCells,
    //                                     const_cast<scalar *>(&local_psi[0]));

    //     psi_view = local_host_x;
    // }

    // void copy_result_back(const scalarField &psi, const label nCells) const;

    // bool get_update_sys_matrix() const { return (update_sysMatrix_ == "yes");
    // }

    bool get_export() const { return export_; }
};

}  // namespace Foam

#endif
