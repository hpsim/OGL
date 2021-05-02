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
#ifndef OGL_IOSortingIdxHandler_INCLUDED_H
#define OGL_IOSortingIdxHandler_INCLUDED_H

#include <vector>
#include "fvCFD.H"
#include "regIOobject.H"


namespace Foam {
class IOSortingIdxHandler {
private:
    const label nElems_;

    // if sorting_idxs_ was found in object registry it does not
    // need to be resorted
    mutable bool is_sorted_;

    // flag to set wheter the idxs for creating the gko coo matrix
    // should be sorted
    const bool sort_;

    IOField<label> *sorting_idxs_ = NULL;

public:
    IOSortingIdxHandler(const objectRegistry &db, const label nElems,
                        const bool sort);

    void init_sorting_idxs();

    void compute_sorting_idxs(const std::vector<label> &row_idxs,
                              const std::vector<label> &col_idxs,
                              const label nCells);

    bool get_sort() const { return sort_; }

    void set_is_sorted(bool is_sorted) const { is_sorted_ = is_sorted; }

    bool get_is_sorted() const { return is_sorted_; }

    const IOField<label> *get_sorting_idxs() const { return sorting_idxs_; }
};
}  // namespace Foam

#endif
