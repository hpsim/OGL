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


// labelRange gkoGlobalIndex::range(const label proci) const
// {
//     return labelRange(offsets_[proci], offsets_[proci + 1] -
//     offsets_[proci]);
// }


// labelRange gkoGlobalIndex::range() const { return range(Pstream::myProcNo());
// }


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
    // if (i < 0 || i >= size()) {
    //     FatalErrorInFunction << "Global " << i
    //                          << " does not belong on any processor."
    //                          << " Offsets:" << offsets_ << abort(FatalError);
    // }

    return findLower(offsets_, i + 1);
}
// template <class Container, class Type>
// void gkoGlobalIndex::gather(const labelUList &off, const label comm,
//                          const Container &procIDs, const UList<Type> &fld,
//                          List<Type> &allFld, const int tag,
//                          const Pstream::commsTypes commsType)
// {
//     // TODO why not using is master ?
//     if (Pstream::master()) {
//         allFld.setSize(off.last());

//         // Assign my local data
//         SubList<Type>(allFld, fld.size(), 0) = fld;

//         if (commsType == Pstream::commsTypes::scheduled ||
//             commsType == Pstream::commsTypes::blocking) {
//             for (label i = 1; i < procIDs.size(); ++i) {
//                 SubList<Type> procSlot(allFld, off[i + 1] - off[i], off[i]);

//                 // TODO
//                 // if (is_contiguous<Type>::value) {
//                 //     IPstream::read(commsType, procIDs[i],
//                 //                    reinterpret_cast<char
//                 *>(procSlot.data()),
//                 //                    procSlot.byteSize(), tag, comm);
//                 // } else {
//                 //     IPstream fromSlave(commsType, procIDs[i], 0, tag,
//                 comm);
//                 //     fromSlave >> procSlot;
//                 // }
//             }
//         } else {
//             // nonBlocking

//             // TODO
//             // if (!is_contiguous<Type>::value) {
//             //     FatalErrorInFunction
//             //         << "nonBlocking not supported for non-contiguous data"
//             //         << exit(FatalError);
//             // }

//             label startOfRequests = Pstream::nRequests();

//             // Set up reads
//             for (label i = 1; i < procIDs.size(); ++i) {
//                 SubList<Type> procSlot(allFld, off[i + 1] - off[i], off[i]);

//                 IPstream::read(commsType, procIDs[i],
//                                reinterpret_cast<char *>(procSlot.data()),
//                                procSlot.byteSize(), tag, comm);
//             }

//             // Wait for all to finish
//             Pstream::waitRequests(startOfRequests);
//         }
//     } else {
//         if (commsType == Pstream::commsTypes::scheduled ||
//             commsType == Pstream::commsTypes::blocking) {
//             // TODO
//             // if (is_contiguous<Type>::value) {
//             //     OPstream::write(commsType, procIDs[0],
//             //                     reinterpret_cast<const char
//             *>(fld.cdata()),
//             //                     fld.byteSize(), tag, comm);
//             // } else {
//             //     OPstream toMaster(commsType, procIDs[0], 0, tag, comm);
//             //     toMaster << fld;
//             // }
//         } else {
//             // nonBlocking

//             // TODO
//             // if (!is_contiguous<Type>::value) {
//             //     FatalErrorInFunction
//             //         << "nonBlocking not supported for non-contiguous data"
//             //         << exit(FatalError);
//             // }

//             label startOfRequests = Pstream::nRequests();

//             // Set up write
//             OPstream::write(commsType, procIDs[0],
//                             reinterpret_cast<const char *>(fld.cdata()),
//                             fld.byteSize(), tag, comm);

//             // Wait for all to finish
//             Pstream::waitRequests(startOfRequests);
//         }
//     }
// }


// template <class Type>
// void gkoGlobalIndex::gather(const UList<Type> &fld, List<Type> &allFld,
//                          const int tag,
//                          const Pstream::commsTypes commsType) const
// {
//     gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm), fld,
//            allFld, tag, commsType);
// }


// template <class Type>
// void gkoGlobalIndex::gatherOp(const UList<Type> &fld, List<Type> &allFld,
//                            const int tag, const Pstream::commsTypes
//                            commsType)
// {
//     gkoGlobalIndex(fld.size()).gather(fld, allFld, tag, commsType);
// }


// template <class Container, class Type>
// void gkoGlobalIndex::gather(const labelUList &off, const label comm,
//                          const Container &procIDs, List<Type> &fld,
//                          const int tag, const Pstream::commsTypes commsType)
// {
//     List<Type> allFld;

//     gather(off, comm, procIDs, fld, allFld, tag, commsType);

//     if (Pstream::myProcNo(comm) == procIDs[0]) {
//         fld.transfer(allFld);
//     }
// }


// template <class Type>
// void gkoGlobalIndex::gather(List<Type> &fld, const int tag,
//                          const Pstream::commsTypes commsType) const
// {
//     List<Type> allFld;

//     gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm), fld,
//            allFld, tag, commsType);

//     if (Pstream::master(UPstream::worldComm)) {
//         fld.transfer(allFld);
//     } else {
//         fld.clear();
//     }
// }


// template <class Type>
// void gkoGlobalIndex::gatherOp(List<Type> &fld, const int tag,
//                            const Pstream::commsTypes commsType)
// {
//     gkoGlobalIndex(fld.size()).gather(fld, tag, commsType);
// }


// template <class Container, class Type>
// void gkoGlobalIndex::scatter(const labelUList &off, const label comm,
//                           const Container &procIDs, const UList<Type>
//                           &allFld, UList<Type> &fld, const int tag, const
//                           Pstream::commsTypes commsType)
// {
//     if (Pstream::myProcNo(comm) == procIDs[0]) {
//         fld.deepCopy(SubList<Type>(allFld, off[1] - off[0]));

//         if (commsType == Pstream::commsTypes::scheduled ||
//             commsType == Pstream::commsTypes::blocking) {
//             for (label i = 1; i < procIDs.size(); ++i) {
//                 const SubList<Type> procSlot(allFld, off[i + 1] - off[i],
//                                              off[i]);

//                 // TODO
//                 // if (is_contiguous<Type>::value) {
//                 //     OPstream::write(
//                 //         commsType, procIDs[i],
//                 //         reinterpret_cast<const char *>(procSlot.cdata()),
//                 //         procSlot.byteSize(), tag, comm);
//                 // } else {
//                 //     OPstream toSlave(commsType, procIDs[i], 0, tag, comm);
//                 //     toSlave << procSlot;
//                 // }
//             }
//         } else {
//             // nonBlocking

//             // TODO
//             // if (!is_contiguous<Type>::value) {
//             //     FatalErrorInFunction
//             //         << "nonBlocking not supported for non-contiguous data"
//             //         << exit(FatalError);
//             // }

//             label startOfRequests = Pstream::nRequests();

//             // Set up writes
//             for (label i = 1; i < procIDs.size(); ++i) {
//                 const SubList<Type> procSlot(allFld, off[i + 1] - off[i],
//                                              off[i]);

//                 OPstream::write(
//                     commsType, procIDs[i],
//                     reinterpret_cast<const char *>(procSlot.cdata()),
//                     procSlot.byteSize(), tag, comm);
//             }

//             // Wait for all to finish
//             Pstream::waitRequests(startOfRequests);
//         }
//     } else {
//         if (commsType == Pstream::commsTypes::scheduled ||
//             commsType == Pstream::commsTypes::blocking) {
//             // TODO
//             // if (is_contiguous<Type>::value) {
//             //     IPstream::read(commsType, procIDs[0],
//             //                    reinterpret_cast<char *>(fld.data()),
//             //                    fld.byteSize(), tag, comm);
//             // } else {
//             //     IPstream fromMaster(commsType, procIDs[0], 0, tag, comm);
//             //     fromMaster >> fld;
//             // }
//         } else {
//             // nonBlocking

//             // TODO
//             // if (!is_contiguous<Type>::value) {
//             //     FatalErrorInFunction
//             //         << "nonBlocking not supported for non-contiguous data"
//             //         << exit(FatalError);
//             // }

//             label startOfRequests = Pstream::nRequests();

//             // Set up read
//             IPstream::read(commsType, procIDs[0],
//                            reinterpret_cast<char *>(fld.data()),
//                            fld.byteSize(), tag, comm);

//             // Wait for all to finish
//             Pstream::waitRequests(startOfRequests);
//         }
//     }
// }


// template <class Type>
// void gkoGlobalIndex::scatter(const UList<Type> &allFld, UList<Type> &fld,
//                           const int tag,
//                           const Pstream::commsTypes commsType) const
// {
//     scatter(offsets_, UPstream::worldComm,
//             UPstream::procID(UPstream::worldComm), allFld, fld, tag,
//             commsType);
// }


// template <class Type, class CombineOp>
// void gkoGlobalIndex::get(List<Type> &allFld, const labelUList &globalIds,
//                       const CombineOp &cop, const label comm,
//                       const int tag) const
// {
//     allFld.setSize(globalIds.size());
//     if (globalIds.size()) {
//         // Sort according to processor
//         labelList order;
//         // CompactListList<label> bins;
//         // DynamicList<label> validBins(Pstream::nProcs());
//         // bin(offsets(), globalIds, order, bins, validBins);

//         // Send local indices to individual processors as local index
//         PstreamBuffers sendBufs(Pstream::commsTypes::nonBlocking, tag, comm);

//         // TODO
//         // for (const auto proci : validBins) {
//         //     // const labelUList &es = bins[proci];

//         //     // labelList localIDs(es.size());
//         //     // forAll(es, i) { localIDs[i] = toLocal(proci, es[i]); }

//         //     // UOPstream os(proci, sendBufs);
//         //     // os << localIDs;
//         // }
//         labelList recvSizes;
//         sendBufs.finishedSends(recvSizes);


//         PstreamBuffers returnBufs(Pstream::commsTypes::nonBlocking, tag,
//         comm);

//         forAll(recvSizes, proci)
//         {
//             if (recvSizes[proci]) {
//                 UIPstream is(proci, sendBufs);
//                 labelList localIDs(is);

//                 // Collect entries
//                 List<Type> fld(localIDs.size());
//                 cop(fld, localIDs);

//                 UOPstream os(proci, returnBufs);
//                 os << fld;
//             }
//         }
//         returnBufs.finishedSends();

//         // Slot back
//         // for (const auto proci : validBins) {
//         //     label start = bins.offsets()[proci];
//         //     const SubList<label> es(order,
//         //                             bins.offsets()[proci + 1] - start,  //
//         //                             start start);
//         //     UIPstream is(proci, returnBufs);
//         //     List<Type> fld(is);

//         //     UIndirectList<Type>(allFld, es) = fld;
//         // }
//     }
// }
// gkoGlobalIndex::gkoGlobalIndex(Istream &is) { is >> offsets_; }


// // * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * *
// //

// void gkoGlobalIndex::bin(const labelUList &offsets, const labelUList
// &globalIds,
//                       labelList &order, CompactListList<label> &bins,
//                       DynamicList<label> &validBins)
// {
//     sortedOrder(globalIds, order);

//     bins.m() = UIndirectList<label>(globalIds, order);

//     labelList &binOffsets = bins.offsets();
//     binOffsets.setSize(offsets.size());
//     binOffsets = 0;

//     validBins.clear();

//     if (globalIds.size()) {
//         const label id = bins.m()[0];
//         label proci = findLower(offsets, id + 1);

//         validBins.append(proci);
//         label binSize = 1;

//         for (label i = 1; i < order.size(); i++) {
//             const label id = bins.m()[i];

//             if (id < offsets[proci + 1]) {
//                 binSize++;
//             } else {
//                 // Not local. Reset proci
//                 label oldProci = proci;
//                 proci = findLower(offsets, id + 1);

//                 // Set offsets
//                 for (label j = oldProci + 1; j < proci; ++j) {
//                     binOffsets[j] = binOffsets[oldProci] + binSize;
//                 }
//                 binOffsets[proci] = i;
//                 validBins.append(proci);
//                 binSize = 1;
//             }
//         }

//         for (label j = proci + 1; j < binOffsets.size(); ++j) {
//             binOffsets[j] = binOffsets[proci] + binSize;
//         }
//     }
// }


void gkoGlobalIndex::init(const label localSize, const int tag,
                          const label comm, const bool parallel)
{
    const label nProcs = Pstream::nProcs(comm);
    offsets_.resize(nProcs + 1);

    labelList localSizes(nProcs, Zero);
    localSizes[Pstream::myProcNo(comm)] = localSize;

    if (parallel) {
        // TODO check if this is meant to run only on master
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

// Istream &operator>>(Istream &is, gkoGlobalIndex &gi) { return is >>
// gi.offsets_;
// }


// Ostream &operator<<(Ostream &os, const gkoGlobalIndex &gi)
// {
//     return os << gi.offsets_;
// }


}  // namespace Foam


// ************************************************************************* //
