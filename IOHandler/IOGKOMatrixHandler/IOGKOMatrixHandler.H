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
using val_array = gko::Array<scalar>;
using idx_array = gko::Array<label>;

class IOGKOMatrixHandler : public IOExecutorHandler {
private:
    const word sys_matrix_name_;

    const bool sys_matrix_stored_;

    const word sparsity_pattern_name_cols_ = "gko_sparsity_pattern_cols";

    const word sparsity_pattern_name_rows_ = "gko_sparsity_pattern_rows";

    const word update_sysMatrix_;

    const bool export_;

    mutable GKOCOOIOPtr *gkomatrix_ptr_ = NULL;

    mutable GKOIDXIOPtr *io_col_idxs_ptr_ = NULL;

    mutable GKOIDXIOPtr *io_row_idxs_ptr_ = NULL;

public:
    IOGKOMatrixHandler(const objectRegistry &db, const dictionary &controlDict)
        : IOExecutorHandler(db, controlDict),
          sys_matrix_name_("gko_sys_matrix"),
          sys_matrix_stored_(db.foundObject<regIOobject>(sys_matrix_name_)),
          update_sysMatrix_(
              controlDict.lookupOrDefault("updateSysMatrix", word("yes"))),
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
        return gkomatrix_ptr_->get_ptr();
    };

    bool get_update_sys_matrix() const { return (update_sysMatrix_ == "yes"); }

    bool get_export() const { return export_; }
};
}  // namespace Foam

#endif
