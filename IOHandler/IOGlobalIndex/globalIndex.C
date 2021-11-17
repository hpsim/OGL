/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2018 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "ListOps.H"
#include "globalIndex.H"
// #include "labelRange.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

namespace Foam {
globalIndex::globalIndex(const label localSize) : globalIndex()
{
    reset(localSize);
}

globalIndex::globalIndex(const label localSize, const int tag, const label comm,
                         const bool parallel)
    : globalIndex()
{
    reset(localSize, tag, comm, parallel);
}


globalIndex::globalIndex(const labelUList &offsets) : offsets_(offsets) {}


globalIndex::globalIndex(labelList &&offsets) : offsets_(std::move(offsets)) {}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool globalIndex::empty() const
{
    return offsets_.empty() || offsets_.last() == 0;
}


const labelList &globalIndex::offsets() const { return offsets_; }


labelList &globalIndex::offsets() { return offsets_; }


label globalIndex::size() const
{
    return offsets_.empty() ? 0 : offsets_.last();
}


void globalIndex::reset(const label localSize)
{
    reset(localSize, Pstream::msgType(), UPstream::worldComm, true);
}


label globalIndex::offset(const label proci) const { return offsets_[proci]; }


label globalIndex::localStart(const label proci) const
{
    return offsets_[proci];
}


label globalIndex::localStart() const
{
    return localStart(Pstream::myProcNo());
}


label globalIndex::localSize(const label proci) const
{
    return offsets_[proci + 1] - offsets_[proci];
}


label globalIndex::localSize() const { return localSize(Pstream::myProcNo()); }


// labelRange globalIndex::range(const label proci) const
// {
//     return labelRange(offsets_[proci], offsets_[proci + 1] -
//     offsets_[proci]);
// }


// labelRange globalIndex::range() const { return range(Pstream::myProcNo()); }


bool globalIndex::isLocal(const label proci, const label i) const
{
    return i >= offsets_[proci] && i < offsets_[proci + 1];
}


bool globalIndex::isLocal(const label i) const
{
    return isLocal(Pstream::myProcNo(), i);
}


label globalIndex::toGlobal(const label proci, const label i) const
{
    return i + offsets_[proci];
}


label globalIndex::toGlobal(const label i) const
{
    return toGlobal(Pstream::myProcNo(), i);
}


labelList globalIndex::toGlobal(const label proci,
                                const labelUList &labels) const
{
    labelList result(labels);
    inplaceToGlobal(proci, result);

    return result;
}


labelList globalIndex::toGlobal(const labelUList &labels) const
{
    return toGlobal(Pstream::myProcNo(), labels);
}


void globalIndex::inplaceToGlobal(const label proci, labelList &labels) const
{
    const label off = offsets_[proci];

    for (label &val : labels) {
        val += off;
    }
}


void globalIndex::inplaceToGlobal(labelList &labels) const
{
    inplaceToGlobal(Pstream::myProcNo(), labels);
}


label globalIndex::toLocal(const label proci, const label i) const
{
    const label locali = i - offsets_[proci];

    if (locali < 0 || i >= offsets_[proci + 1]) {
        FatalErrorInFunction << "Global " << i
                             << " does not belong on processor " << proci << nl
                             << "Offsets:" << offsets_ << abort(FatalError);
    }
    return locali;
}


label globalIndex::toLocal(const label i) const
{
    return toLocal(Pstream::myProcNo(), i);
}


label globalIndex::whichProcID(const label i) const
{
    // if (i < 0 || i >= size()) {
    //     FatalErrorInFunction << "Global " << i
    //                          << " does not belong on any processor."
    //                          << " Offsets:" << offsets_ << abort(FatalError);
    // }

    return findLower(offsets_, i + 1);
}
template <class Container, class Type>
void globalIndex::gather(const labelUList &off, const label comm,
                         const Container &procIDs, const UList<Type> &fld,
                         List<Type> &allFld, const int tag,
                         const Pstream::commsTypes commsType)
{
    if (Pstream::myProcNo(comm) == procIDs[0]) {
        allFld.setSize(off.last());

        // Assign my local data
        SubList<Type>(allFld, fld.size(), 0) = fld;

        if (commsType == Pstream::commsTypes::scheduled ||
            commsType == Pstream::commsTypes::blocking) {
            for (label i = 1; i < procIDs.size(); ++i) {
                SubList<Type> procSlot(allFld, off[i + 1] - off[i], off[i]);

                // TODO
                // if (is_contiguous<Type>::value) {
                //     IPstream::read(commsType, procIDs[i],
                //                    reinterpret_cast<char *>(procSlot.data()),
                //                    procSlot.byteSize(), tag, comm);
                // } else {
                //     IPstream fromSlave(commsType, procIDs[i], 0, tag, comm);
                //     fromSlave >> procSlot;
                // }
            }
        } else {
            // nonBlocking

            // TODO
            // if (!is_contiguous<Type>::value) {
            //     FatalErrorInFunction
            //         << "nonBlocking not supported for non-contiguous data"
            //         << exit(FatalError);
            // }

            label startOfRequests = Pstream::nRequests();

            // Set up reads
            for (label i = 1; i < procIDs.size(); ++i) {
                SubList<Type> procSlot(allFld, off[i + 1] - off[i], off[i]);

                IPstream::read(commsType, procIDs[i],
                               reinterpret_cast<char *>(procSlot.data()),
                               procSlot.byteSize(), tag, comm);
            }

            // Wait for all to finish
            Pstream::waitRequests(startOfRequests);
        }
    } else {
        if (commsType == Pstream::commsTypes::scheduled ||
            commsType == Pstream::commsTypes::blocking) {
            // TODO
            // if (is_contiguous<Type>::value) {
            //     OPstream::write(commsType, procIDs[0],
            //                     reinterpret_cast<const char *>(fld.cdata()),
            //                     fld.byteSize(), tag, comm);
            // } else {
            //     OPstream toMaster(commsType, procIDs[0], 0, tag, comm);
            //     toMaster << fld;
            // }
        } else {
            // nonBlocking

            // TODO
            // if (!is_contiguous<Type>::value) {
            //     FatalErrorInFunction
            //         << "nonBlocking not supported for non-contiguous data"
            //         << exit(FatalError);
            // }

            label startOfRequests = Pstream::nRequests();

            // Set up write
            OPstream::write(commsType, procIDs[0],
                            reinterpret_cast<const char *>(fld.cdata()),
                            fld.byteSize(), tag, comm);

            // Wait for all to finish
            Pstream::waitRequests(startOfRequests);
        }
    }
}


template <class Type>
void globalIndex::gather(const UList<Type> &fld, List<Type> &allFld,
                         const int tag,
                         const Pstream::commsTypes commsType) const
{
    gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm), fld,
           allFld, tag, commsType);
}


template <class Type>
void globalIndex::gatherOp(const UList<Type> &fld, List<Type> &allFld,
                           const int tag, const Pstream::commsTypes commsType)
{
    globalIndex(fld.size()).gather(fld, allFld, tag, commsType);
}


template <class Container, class Type>
void globalIndex::gather(const labelUList &off, const label comm,
                         const Container &procIDs, List<Type> &fld,
                         const int tag, const Pstream::commsTypes commsType)
{
    List<Type> allFld;

    gather(off, comm, procIDs, fld, allFld, tag, commsType);

    if (Pstream::myProcNo(comm) == procIDs[0]) {
        fld.transfer(allFld);
    }
}


template <class Type>
void globalIndex::gather(List<Type> &fld, const int tag,
                         const Pstream::commsTypes commsType) const
{
    List<Type> allFld;

    gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm), fld,
           allFld, tag, commsType);

    if (Pstream::master(UPstream::worldComm)) {
        fld.transfer(allFld);
    } else {
        fld.clear();
    }
}


template <class Type>
void globalIndex::gatherOp(List<Type> &fld, const int tag,
                           const Pstream::commsTypes commsType)
{
    globalIndex(fld.size()).gather(fld, tag, commsType);
}


template <class Container, class Type>
void globalIndex::scatter(const labelUList &off, const label comm,
                          const Container &procIDs, const UList<Type> &allFld,
                          UList<Type> &fld, const int tag,
                          const Pstream::commsTypes commsType)
{
    if (Pstream::myProcNo(comm) == procIDs[0]) {
        fld.deepCopy(SubList<Type>(allFld, off[1] - off[0]));

        if (commsType == Pstream::commsTypes::scheduled ||
            commsType == Pstream::commsTypes::blocking) {
            for (label i = 1; i < procIDs.size(); ++i) {
                const SubList<Type> procSlot(allFld, off[i + 1] - off[i],
                                             off[i]);

                // TODO
                // if (is_contiguous<Type>::value) {
                //     OPstream::write(
                //         commsType, procIDs[i],
                //         reinterpret_cast<const char *>(procSlot.cdata()),
                //         procSlot.byteSize(), tag, comm);
                // } else {
                //     OPstream toSlave(commsType, procIDs[i], 0, tag, comm);
                //     toSlave << procSlot;
                // }
            }
        } else {
            // nonBlocking

            // TODO
            // if (!is_contiguous<Type>::value) {
            //     FatalErrorInFunction
            //         << "nonBlocking not supported for non-contiguous data"
            //         << exit(FatalError);
            // }

            label startOfRequests = Pstream::nRequests();

            // Set up writes
            for (label i = 1; i < procIDs.size(); ++i) {
                const SubList<Type> procSlot(allFld, off[i + 1] - off[i],
                                             off[i]);

                OPstream::write(
                    commsType, procIDs[i],
                    reinterpret_cast<const char *>(procSlot.cdata()),
                    procSlot.byteSize(), tag, comm);
            }

            // Wait for all to finish
            Pstream::waitRequests(startOfRequests);
        }
    } else {
        if (commsType == Pstream::commsTypes::scheduled ||
            commsType == Pstream::commsTypes::blocking) {
            // TODO
            // if (is_contiguous<Type>::value) {
            //     IPstream::read(commsType, procIDs[0],
            //                    reinterpret_cast<char *>(fld.data()),
            //                    fld.byteSize(), tag, comm);
            // } else {
            //     IPstream fromMaster(commsType, procIDs[0], 0, tag, comm);
            //     fromMaster >> fld;
            // }
        } else {
            // nonBlocking

            // TODO
            // if (!is_contiguous<Type>::value) {
            //     FatalErrorInFunction
            //         << "nonBlocking not supported for non-contiguous data"
            //         << exit(FatalError);
            // }

            label startOfRequests = Pstream::nRequests();

            // Set up read
            IPstream::read(commsType, procIDs[0],
                           reinterpret_cast<char *>(fld.data()), fld.byteSize(),
                           tag, comm);

            // Wait for all to finish
            Pstream::waitRequests(startOfRequests);
        }
    }
}


template <class Type>
void globalIndex::scatter(const UList<Type> &allFld, UList<Type> &fld,
                          const int tag,
                          const Pstream::commsTypes commsType) const
{
    scatter(offsets_, UPstream::worldComm,
            UPstream::procID(UPstream::worldComm), allFld, fld, tag, commsType);
}


template <class Type, class CombineOp>
void globalIndex::get(List<Type> &allFld, const labelUList &globalIds,
                      const CombineOp &cop, const label comm,
                      const int tag) const
{
    allFld.setSize(globalIds.size());
    if (globalIds.size()) {
        // Sort according to processor
        labelList order;
        // CompactListList<label> bins;
        // DynamicList<label> validBins(Pstream::nProcs());
        // bin(offsets(), globalIds, order, bins, validBins);

        // Send local indices to individual processors as local index
        PstreamBuffers sendBufs(Pstream::commsTypes::nonBlocking, tag, comm);

        // TODO
        // for (const auto proci : validBins) {
        //     // const labelUList &es = bins[proci];

        //     // labelList localIDs(es.size());
        //     // forAll(es, i) { localIDs[i] = toLocal(proci, es[i]); }

        //     // UOPstream os(proci, sendBufs);
        //     // os << localIDs;
        // }
        labelList recvSizes;
        sendBufs.finishedSends(recvSizes);


        PstreamBuffers returnBufs(Pstream::commsTypes::nonBlocking, tag, comm);

        forAll(recvSizes, proci)
        {
            if (recvSizes[proci]) {
                UIPstream is(proci, sendBufs);
                labelList localIDs(is);

                // Collect entries
                List<Type> fld(localIDs.size());
                cop(fld, localIDs);

                UOPstream os(proci, returnBufs);
                os << fld;
            }
        }
        returnBufs.finishedSends();

        // Slot back
        // for (const auto proci : validBins) {
        //     label start = bins.offsets()[proci];
        //     const SubList<label> es(order,
        //                             bins.offsets()[proci + 1] - start,  //
        //                             start start);
        //     UIPstream is(proci, returnBufs);
        //     List<Type> fld(is);

        //     UIndirectList<Type>(allFld, es) = fld;
        // }
    }
}


}  // namespace Foam


// ************************************************************************* //
