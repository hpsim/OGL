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

using mtx = gko::matrix::Coo<scalar>;
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

    mutable std::shared_ptr<mtx> gkomatrix_ = NULL;

    mutable GKOCOOIOPtr *gkomatrix_ptr_ = NULL;

    mutable GKOIDXIOPtr *io_col_idxs_ptr_ = NULL;

    mutable GKOIDXIOPtr *io_row_idxs_ptr_ = NULL;

    mutable std::shared_ptr<vec> init_guess_ = NULL;

    mutable GKOVECIOPtr *io_init_guess_ptr_ = NULL;

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

    void init_initial_guess(const scalarField &psi, const objectRegistry &db,
                            const label nCells) const
    {
        std::shared_ptr<gko::Executor> device_exec = get_device_executor();

        if (init_guess_vector_stored_ && !update_init_guess_vector_) {
            std::cout << " init_initial_guess do nothing" << std::endl;
            io_init_guess_ptr_ =
                &db.lookupObjectRef<GKOVECIOPtr>(init_guess_vector_name_);
            return;
        }

        std::cout << " init_initial_guess create psi_view" << std::endl;
        auto psi_view = val_array::view(gko::ReferenceExecutor::create(),
                                        nCells, const_cast<scalar *>(&psi[0]));

        std::cout << " init_initial_guess create x" << std::endl;
        auto x = gko::share(
            vec::create(device_exec, gko::dim<2>(nCells, 1), psi_view, 1));

        const fileName path_init_guess = init_guess_vector_name_;
        std::cout << " init_initial_guess create io_init_guess_ptr_"
                  << std::endl;
        io_init_guess_ptr_ = new GKOVECIOPtr(IOobject(path_init_guess, db), x);
    }


    std::shared_ptr<vec> get_initial_guess() const
    {
        std::cout << "get_initial_guess" << std::endl;
        if (io_init_guess_ptr_ == NULL) {
            std::cout << "is Null" << std::endl;
            return init_guess_;
        }
        return io_init_guess_ptr_->get_ptr();
    }

    void copy_result_back(const scalarField &psi, const label nCells) const
    {
        std::cout << "copy back" << std::endl;
        auto device_x = vec::create(ref_exec(), nCells);
        std::cout << "copy back 2" << std::endl;
        device_x->copy_from(gko::lend(get_initial_guess()));

        std::cout << "create x_view back" << std::endl;
        auto x_view =
            val_array::view(ref_exec(), nCells, device_x->get_values());
        // for (label i = 0; i < nCells; i++) {
        //     std::cout << x_view[i] << std::endl;
        // }

        // move frome device
        std::cout << "create psi_view " << std::endl;
        auto psi_view =
            val_array::view(ref_exec(), nCells, const_cast<scalar *>(&psi[0]));

        std::cout << "copy x_view to psi_view " << std::endl;

        psi_view = x_view;
    }

    bool get_update_sys_matrix() const { return (update_sysMatrix_ == "yes"); }

    bool get_export() const { return export_; }
};
}  // namespace Foam

#endif
