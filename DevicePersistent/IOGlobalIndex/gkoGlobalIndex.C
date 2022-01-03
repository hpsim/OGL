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

\*---------------------------------------------------------------------------*/

#include "gkoGlobalIndex.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

namespace Foam {
gkoGlobalIndex::gkoGlobalIndex(const label localSize) : gkoGlobalIndex()
{
    init(localSize);
}

gkoGlobalIndex::gkoGlobalIndex(const label localSize, const int tag,
                               const label comm, const bool parallel)
    : gkoGlobalIndex()
{
    init(localSize, tag, comm, parallel);
}


gkoGlobalIndex::gkoGlobalIndex(const labelUList &offsets) : offsets_(offsets) {}


gkoGlobalIndex::gkoGlobalIndex(labelList &&offsets)
    : offsets_(std::move(offsets))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool gkoGlobalIndex::empty() const
{
    return offsets_.empty() || offsets_.last() == 0;
}


const labelList &gkoGlobalIndex::offsets() const { return offsets_; }


labelList &gkoGlobalIndex::offsets() { return offsets_; }


label gkoGlobalIndex::size() const
{
    return offsets_.empty() ? 0 : offsets_.last();
}


void gkoGlobalIndex::init(const label localSize)
{
    init(localSize, Pstream::msgType(), UPstream::worldComm, Pstream::parRun());
}


label gkoGlobalIndex::offset(const label proci) const
{
    return offsets_[proci];
}


label gkoGlobalIndex::localStart(const label proci) const
{
    return offsets_[proci];
}


label gkoGlobalIndex::localStart() const
{
    return localStart(Pstream::myProcNo());
}


label gkoGlobalIndex::localSize(const label proci) const
{
    return offsets_[proci + 1] - offsets_[proci];
}


label gkoGlobalIndex::localSize() const
{
    return localSize(Pstream::myProcNo());
}


bool gkoGlobalIndex::isLocal(const label proci, const label i) const
{
    return i >= offsets_[proci] && i < offsets_[proci + 1];
}


bool gkoGlobalIndex::isLocal(const label i) const
{
    return isLocal(Pstream::myProcNo(), i);
}


label gkoGlobalIndex::toGlobal(const label proci, const label i) const
{
    if (!Pstream::parRun()) {
        return i;
    }
    return i + offsets_[proci];
}


label gkoGlobalIndex::toGlobal(const label i) const
{
    return toGlobal(Pstream::myProcNo(), i);
}


labelList gkoGlobalIndex::toGlobal(const label proci,
                                   const labelUList &labels) const
{
    labelList result(labels);
    inplaceToGlobal(proci, result);

    return result;
}


labelList gkoGlobalIndex::toGlobal(const labelUList &labels) const
{
    return toGlobal(Pstream::myProcNo(), labels);
}


void gkoGlobalIndex::inplaceToGlobal(const label proci, labelList &labels) const
{
    const label off = offsets_[proci];

    for (label &val : labels) {
        val += off;
    }
}


void gkoGlobalIndex::inplaceToGlobal(labelList &labels) const
{
    inplaceToGlobal(Pstream::myProcNo(), labels);
}


label gkoGlobalIndex::toLocal(const label proci, const label i) const
{
    const label locali = i - offsets_[proci];

    if (locali < 0 || i >= offsets_[proci + 1]) {
        FatalErrorInFunction << "Global " << i
                             << " does not belong on processor " << proci << nl
                             << "Offsets:" << offsets_ << abort(FatalError);
    }
    return locali;
}


label gkoGlobalIndex::toLocal(const label i) const
{
    return toLocal(Pstream::myProcNo(), i);
}


label gkoGlobalIndex::whichProcID(const label i) const
{
    return findLower(offsets_, i + 1);
}


void gkoGlobalIndex::init(const label localSize, const int tag,
                          const label comm, const bool parallel)
{
    const label nProcs = Pstream::nProcs(comm);
    offsets_.resize(nProcs + 1);

    labelList localSizes(nProcs, Zero);
    localSizes[Pstream::myProcNo(comm)] = localSize;

    if (parallel) {
        Pstream::gatherList(localSizes, tag, comm);
        Pstream::scatterList(localSizes, tag, comm);
    }

    label offset = 0;
    offsets_[0] = 0;
    for (int proci = 0; proci < nProcs; proci++) {
        const label oldOffset = offset;
        offset += localSizes[proci];

        if (offset < oldOffset) {
            FatalErrorInFunction
                << "Overflow : sum of sizes " << localSizes
                << " exceeds capability of label (" << labelMax
                << "). Please recompile with larger datatype for label."
                << exit(FatalError);
        }
        offsets_[proci + 1] = offset;
    }
}


labelList gkoGlobalIndex::sizes() const
{
    labelList values;

    const label len = (offsets_.size() - 1);

    if (len < 1) {
        return values;
    }

    values.resize(len);

    for (label proci = 0; proci < len; ++proci) {
        values[proci] = offsets_[proci + 1] - offsets_[proci];
    }

    return values;
}


// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //


}  // namespace Foam


// ************************************************************************* //
