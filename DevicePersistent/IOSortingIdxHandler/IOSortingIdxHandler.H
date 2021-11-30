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
    Foam::IOSortingIdxHandler

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    IOSortingIdxHandler.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOSortingIdxHandler_INCLUDED_H
#define OGL_IOSortingIdxHandler_INCLUDED_H
#include <ginkgo/ginkgo.hpp>

#include <vector>
#include "fvCFD.H"
#include "regIOobject.H"


namespace Foam {
class IOSortingIdxHandler {
private:
    using val_array = gko::Array<scalar>;
    using idx_array = gko::Array<label>;

    const label nElems_;

    mutable bool is_sorted_;

    mutable IOField<label> *sorting_idxs_ = NULL;

public:
    // TODO handle sorting idxs per rank
    IOSortingIdxHandler(const objectRegistry &db, const label nElems);

    void init_sorting_idxs();

    void compute_sorting_idxs(const std::shared_ptr<idx_array> row_idxs,
                              const std::shared_ptr<idx_array> col_idxs,
                              const label nCells);

    bool get_sort() const { return true; }

    void set_is_sorted(bool is_sorted) const { is_sorted_ = is_sorted; }

    bool get_is_sorted() const { return is_sorted_; }

    IOField<label> *get_sorting_idxs() const { return sorting_idxs_; }

    const IOField<label> *get_const_sorting_idxs() const
    {
        return sorting_idxs_;
    }
};
}  // namespace Foam

#endif
