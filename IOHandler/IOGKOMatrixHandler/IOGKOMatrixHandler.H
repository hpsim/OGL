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
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOGKOMatrixHandler_INCLUDED_H
#define OGL_IOGKOMatrixHandler_INCLUDED_H

#include "fvCFD.H"
#include "regIOobject.H"

#include "../IOExecutorHandler/IOExecutorHandler.H"
#include "../IOPtr/IOPtr.H"

namespace Foam {

using coo_mtx = gko::matrix::Coo<scalar>;
using mtx = gko::matrix::Csr<scalar>;
using vec = gko::matrix::Dense<scalar>;
using val_array = gko::Array<scalar>;
using idx_array = gko::Array<label>;

class IOGKOMatrixHandler : public IOExecutorHandler {
private:
    const word sys_matrix_name_;

    const bool sys_matrix_stored_;

    const word update_sysMatrix_;

    const word sparsity_pattern_name_cols_;

    const word sparsity_pattern_name_rows_;

    const bool sparsity_pattern_stored_;

    const word init_guess_vector_name_;

    const bool init_guess_vector_stored_;

    const bool update_init_guess_vector_;

    const bool export_;

    mutable label prev_solve_iters_ = 0;

    mutable std::shared_ptr<mtx> gkomatrix_ = NULL;

    mutable GKOCSRIOPtr *gkomatrix_ptr_ = NULL;

    mutable GKOIDXIOPtr *io_col_idxs_ptr_ = NULL;

    mutable GKOIDXIOPtr *io_row_idxs_ptr_ = NULL;

    mutable std::shared_ptr<vec> init_guess_ = NULL;

    mutable std::vector<GKOVECIOPtr *> io_init_guess_ptrs_ = {};

public:
    IOGKOMatrixHandler(const objectRegistry &db, const dictionary &controlDict,
                       const word sys_matrix_name)
        : IOExecutorHandler(db, controlDict),
          sys_matrix_name_(sys_matrix_name + "_gko"),
          sys_matrix_stored_(db.foundObject<regIOobject>(sys_matrix_name_)),
          update_sysMatrix_(
              controlDict.lookupOrDefault("updateSysMatrix", word("yes"))),
          sparsity_pattern_name_cols_("gko_sparsity_pattern_cols"),
          sparsity_pattern_name_rows_("gko_sparsity_pattern_rows"),
          sparsity_pattern_stored_(
              db.foundObject<regIOobject>(sparsity_pattern_name_cols_)),
          init_guess_vector_name_(sys_matrix_name_ + "_init_guess"),
          init_guess_vector_stored_(
              db.foundObject<regIOobject>(init_guess_vector_name_)),
          update_init_guess_vector_(
              controlDict.lookupOrDefault<Switch>("updateInitVector", false)),

          export_(controlDict.lookupOrDefault<Switch>("export", false)){};

    bool get_sys_matrix_stored() const { return sys_matrix_stored_; };

    bool get_sparsity_pattern_stored() const
    {
        return sparsity_pattern_stored_;
    };

    void init_device_matrix(const objectRegistry &db,
                            std::vector<scalar> &values_host,
                            std::vector<label> &col_idxs_host,
                            std::vector<label> &row_idxs_host,
                            const label nElems, const label nCells,
                            const bool store) const;

    std::shared_ptr<mtx> get_gkomatrix() const
    {
        if (gkomatrix_ptr_ == NULL) {
            return gkomatrix_;
        }
        return gkomatrix_ptr_->get_ptr();
    };

    void init_initial_guess_vector(const Field<vector> &psi,
                                   const objectRegistry &db,
                                   const label nCells) const
    {
        std::vector<word> postFixes{"", "y", "z"};
        for (int i = 0; i < 3; i++) {
            init_initial_guess(&psi[0][i], db, nCells, postFixes[i]);
        }
    }

    void init_initial_guess_vector(const Field<scalar> &psi,
                                   const objectRegistry &db,
                                   const label nCells) const
    {
        init_initial_guess(&psi[0], db, nCells, "");
    }

    void init_initial_guess(const scalar *psi, const objectRegistry &db,
                            const label nCells, const word postFix) const
    {
        std::shared_ptr<gko::Executor> device_exec = get_device_executor();

        if (init_guess_vector_stored_ && !update_init_guess_vector_) {
            io_init_guess_ptrs_.push_back(&db.lookupObjectRef<GKOVECIOPtr>(
                init_guess_vector_name_ + postFix));
            return;
        }

        auto psi_view = val_array::view(gko::ReferenceExecutor::create(),
                                        nCells, const_cast<scalar *>(psi));

        auto x = gko::share(
            vec::create(device_exec, gko::dim<2>(nCells, 1), psi_view, 1));

        const fileName path_init_guess = init_guess_vector_name_ + postFix;
        io_init_guess_ptrs_.push_back(
            new GKOVECIOPtr(IOobject(path_init_guess, db), x));
    }


    void get_initial_guess(std::vector<std::shared_ptr<vec>> &init_guess) const
    {
        if (io_init_guess_ptrs_.size() == 0) {
            init_guess.push_back(init_guess_);
        }
        for (int i = 0; i < io_init_guess_ptrs_.size(); i++) {
            init_guess.push_back(io_init_guess_ptrs_[i]->get_ptr());
        }
    }

    template <class OFField>
    void copy_result_back(const OFField &psi, const label nCells) const
    {
        std::vector<std::shared_ptr<vec>> device_xs_ptr{};
        get_initial_guess(device_xs_ptr);

        for (int i = 0; i < 3; i++) {
            auto host_x = vec::create(ref_exec(), gko::dim<2>(nCells, 1));
            host_x->copy_from(gko::lend(device_xs_ptr[i]));
            auto host_x_view =
                val_array::view(ref_exec(), nCells, host_x->get_values());
            auto psi_view = val_array::view(ref_exec(), nCells,
                                            const_cast<scalar *>(&psi[0][i]));
            psi_view = host_x_view;
        }
    }

    void copy_result_back(const scalarField &psi, const label nCells) const
    {
        auto device_x = vec::create(ref_exec(), gko::dim<2>(nCells, 1));

        std::vector<std::shared_ptr<vec>> device_xs_ptr{};
        get_initial_guess(device_xs_ptr);
        device_x->copy_from(gko::lend(device_xs_ptr[0]));

        auto x_view =
            val_array::view(ref_exec(), nCells, device_x->get_values());
        // for (label i = 0; i < nCells; i++) {
        //     std::cout << x_view[i] << std::endl;
        // }

        // move frome device
        auto psi_view =
            val_array::view(ref_exec(), nCells, const_cast<scalar *>(&psi[0]));

        psi_view = x_view;
    }

    bool get_update_sys_matrix() const { return (update_sysMatrix_ == "yes"); }

    bool get_export() const { return export_; }
};
}  // namespace Foam

#endif
