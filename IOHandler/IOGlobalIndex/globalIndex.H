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
Based on:
https://develop.openfoam.com/Development/openfoam/-/tree/OpenFOAM-v2012/src/OpenFOAM/meshes/polyMesh/globalMeshData

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_IOGlobalIndex_INCLUDED_H
#define OGL_IOGlobalIndex_INCLUDED_H


#include "labelList.H"

//#include "CompactListList.H"
#include "DynamicList.H"
#include "Pstream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

// Forward declarations
template <class T>
class CompactListList;


/*---------------------------------------------------------------------------*\
                           Class globalIndex Declaration
\*---------------------------------------------------------------------------*/

class globalIndex {
    // Private Member Functions

    // //- Sort and bin. validBins contains bins with non-zero size.
    // static void bin(const labelUList &offsets, const labelUList &globalIds,
    //                 labelList &order, CompactListList<label> &sortedElems,
    //                 DynamicList<label> &validBins);


    // Private data

    //- Start of proci. Size is nProcs()+1. (so like CompactListList)
    labelList offsets_;


public:
    // Constructors

    //- Construct null
    globalIndex() = default;

    //- Construct from local max size.
    //  Does communication with default communicator and message tag.
    explicit globalIndex(const label localSize);

    //- Construct from local max size.
    //  Does communication with given communicator and message tag
    globalIndex(const label localSize,
                const int tag,       //!< message tag
                const label comm,    //!< communicator
                const bool parallel  //!< use parallel comms
    );

    //- Copy construct from list of labels
    explicit globalIndex(const labelUList &offsets);

    //- Move construct from list of labels
    explicit globalIndex(labelList &&offsets);

    //- Construct from Istream
    explicit globalIndex(Istream &is);


    // Member Functions

    //- Check for null constructed or global sum == 0
    bool empty() const;

    //- Const-access to the offsets
    const labelList &offsets() const;

    //- Global sum of localSizes
    label size() const;

    //- The local sizes
    labelList sizes() const;


    // Edit

    //- Write-access to the offsets, for changing after construction
    labelList &offsets();

    /**
     * Initialize from local size.
     *
     * Gets msgType and worlComm from Pstream and call 4 arguments init
     */
    void init(const label localSize);

    //- Initialize from local size.
    //  Does communication with given communicator and message tag
    void init(const label localSize,
              const int tag,       //!< message tag
              const label comm,    //!< communicator
              const bool parallel  //!< use parallel comms
    );


    // Queries relating to my processor (using world communicator)

    //- My local start
    label localStart() const;

    //- My local size
    label localSize() const;

    //- Return start/size range of local processor data
    // std::pair<label, label> range() const;

    //- Is on local processor
    bool isLocal(const label i) const;

    //- From local to global index
    label toGlobal(const label i) const;

    //- From local to global index
    labelList toGlobal(const labelUList &labels) const;

    //- From local to global index (inplace)
    void inplaceToGlobal(labelList &labels) const;

    //- From global to local on current processor.
    //  FatalError if not on local processor.
    label toLocal(const label i) const;


    // Global queries

    //- Start of proci data
    label offset(const label proci) const;

    //- Start of proci data
    label localStart(const label proci) const;

    //- Size of proci data
    label localSize(const label proci) const;

    //- Return start/size range of proci data
    //  labelRange range(const label proci) const;

    //- Is on processor proci
    bool isLocal(const label proci, const label i) const;

    //- From local to global on proci
    label toGlobal(const label proci, const label i) const;

    //- From local to global on proci
    labelList toGlobal(const label proci, const labelUList &labels) const;

    //- From local to global index on proci (inplace)
    void inplaceToGlobal(const label proci, labelList &labels) const;


    //- From global to local on proci
    label toLocal(const label proci, const label i) const;

    //- Which processor does global come from? Binary search.
    label whichProcID(const label i) const;


    // Other

    //- Collect data in processor order on master (== procIDs[0]).
    //  Needs offsets only on master.
    template <class Container, class Type>
    static void gather(
        const labelUList &offsets, const label comm, const Container &procIDs,
        const UList<Type> &fld, List<Type> &allFld,
        const int tag = UPstream::msgType(),
        const Pstream::commsTypes commsType = Pstream::commsTypes::nonBlocking)
    {
        if (Pstream::master()) {
            allFld.setSize(offsets.last());

            // Assign my local data
            SubList<Type>(allFld, fld.size(), 0) = fld;

            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                for (label i = 1; i < procIDs.size(); ++i) {
                    SubList<Type> procSlot(allFld, offsets[i + 1] - offsets[i],
                                           offsets[i]);

                    // TODO
                    // if (is_contiguous<Type>::value) {
                    // For now assum contiguous data only
                    IPstream::read(commsType, procIDs[i],
                                   reinterpret_cast<char *>(procSlot.data()),
                                   procSlot.byteSize(), tag, comm);
                    // } else {
                    //     IPstream fromSlave(commsType, procIDs[i], 0, tag,
                    //     comm);
                    //     fromSlave >> procSlot;
                    // }
                }
            } else {
                // nonBlocking

                // TODO
                // if (!is_contiguous<Type>::value) {
                //     FatalErrorInFunction
                //         << "nonBlocking not supported for non-contiguous
                //         data"
                //         << exit(FatalError);
                // }

                label startOfRequests = Pstream::nRequests();

                // Set up reads
                for (label i = 1; i < procIDs.size(); ++i) {
                    SubList<Type> procSlot(allFld, offsets[i + 1] - offsets[i],
                                           offsets[i]);

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
                OPstream::write(commsType, procIDs[0],
                                reinterpret_cast<const char *>(fld.cdata()),
                                fld.byteSize(), tag, comm);
                // } else {
                //     OPstream toMaster(commsType, procIDs[0], 0, tag, comm);
                //     toMaster << fld;
                // }
            } else {
                // nonBlocking

                // TODO
                // if (!is_contiguous<Type>::value) {
                //     FatalErrorInFunction
                //         << "nonBlocking not supported for non-contiguous
                //         data"
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
    };

    //- Collect data in processor order on master (== procIDs[0]).
    //  Needs offsets only on master.
    template <class Container, class Type>
    void gather(const label comm, const Container &procIDs,
                const UList<Type> &fld, List<Type> &allFld,
                const int tag = UPstream::msgType(),
                const Pstream::commsTypes commsType =
                    Pstream::commsTypes::nonBlocking) const
    {
        gather(offsets_, comm, procIDs, fld, allFld, tag, commsType);
    }

    //- Collect data in processor order on master.
    //  Does communication with default communicator and message tag.
    template <class Type>
    void gather(const UList<Type> &fld, List<Type> &allFld,
                const int tag = UPstream::msgType(),
                const Pstream::commsTypes commsType =
                    Pstream::commsTypes::nonBlocking) const
    {
        gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm), fld,
               allFld, tag, commsType);
    }

    // //- Collect data in processor order on master.
    // //  Does communication with default communicator and message tag.
    // template <class Type>
    // static void gatherOp(
    //     const UList<Type> &fld, List<Type> &allFld,
    //     const int tag = UPstream::msgType(),
    //     const Pstream::commsTypes commsType =
    //     Pstream::commsTypes::nonBlocking);


    // //- Inplace collect in processor order on master (== procIDs[0]).
    // //- Needs offsets only on master.
    // template <class Container, class Type>
    // static void gather(
    //     const labelUList &offsets, const label comm, const Container
    //     &procIDs, List<Type> &fld, const int tag = UPstream::msgType(), const
    //     Pstream::commsTypes commsType = Pstream::commsTypes::nonBlocking);

    // //- Inplace collect in processor order on master (== procIDs[0]).
    // //- Needs offsets only on master.
    // template <class Container, class Type>
    // void gather(const label comm, const Container &procIDs, List<Type> &fld,
    //             const int tag = UPstream::msgType(),
    //             const Pstream::commsTypes commsType =
    //                 Pstream::commsTypes::nonBlocking) const
    // {
    //     gather(offsets_, comm, procIDs, fld, tag, commsType);
    // }

    // //- Inplace collect data in processor order on master
    // //  Does communication with default communicator and message tag.
    // //  After the gather, the field is zero-sized on the slaves.
    // template <class Type>
    // void gather(List<Type> &fld, const int tag = UPstream::msgType(),
    //             const Pstream::commsTypes commsType =
    //                 Pstream::commsTypes::nonBlocking) const;

    // //- Inplace collect data in processor order on master
    // //  Does communication with default communicator and message tag.
    // //  After the gather, the field is zero-sized on the slaves.
    // template <class Type>
    // static void gatherOp(
    //     List<Type> &fld, const int tag = UPstream::msgType(),
    //     const Pstream::commsTypes commsType =
    //     Pstream::commsTypes::nonBlocking);


    // //- Distribute data in processor order. Requires fld to be sized!
    // template <class Container, class Type>
    // static void scatter(
    //     const labelUList &offsets, const label comm, const Container
    //     &procIDs, const UList<Type> &allFld, UList<Type> &fld, const int tag
    //     = UPstream::msgType(), const Pstream::commsTypes commsType =
    //     Pstream::commsTypes::nonBlocking);

    // //- Distribute data in processor order. Requires fld to be sized!
    // template <class Container, class Type>
    // void scatter(const label comm, const Container &procIDs,
    //              const UList<Type> &allFld, UList<Type> &fld,
    //              const int tag = UPstream::msgType(),
    //              const Pstream::commsTypes commsType =
    //                  Pstream::commsTypes::nonBlocking) const
    // {
    //     scatter(offsets_, comm, procIDs, allFld, fld, tag, commsType);
    // }

    // //- Distribute data in processor order. Requires fld to be sized!
    // //  Does communication with default communicator and message tag.
    // template <class Type>
    // void scatter(const UList<Type> &allFld, UList<Type> &fld,
    //              const int tag = UPstream::msgType(),
    //              const Pstream::commsTypes commsType =
    //                  Pstream::commsTypes::nonBlocking) const;

    // //- Get (potentially remote) data. Elements required given as
    // //  global indices
    // template <class Type, class CombineOp>
    // void get(List<Type> &allFld, const labelUList &globalIds,
    //          const CombineOp &cop, const label comm = Pstream::worldComm,
    //          const int tag = UPstream::msgType()) const;


    // IOstream Operators

    // friend Istream &operator>>(Istream &is, globalIndex &gi);
    // friend Ostream &operator<<(Ostream &os, const globalIndex &gi);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
