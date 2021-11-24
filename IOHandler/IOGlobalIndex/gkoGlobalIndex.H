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

#include <ginkgo/ginkgo.hpp>
#include "labelList.H"

//#include "CompactListList.H"
#include <memory>
#include "DynamicList.H"
#include "Pstream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {
using val_array = gko::Array<scalar>;
using idx_array = gko::Array<label>;


/*---------------------------------------------------------------------------*\
                           Class gkoGlobalIndex Declaration
\*---------------------------------------------------------------------------*/

class gkoGlobalIndex {
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
    gkoGlobalIndex() = default;

    //- Construct from local max size.
    //  Does communication with default communicator and message tag.
    explicit gkoGlobalIndex(const label localSize);

    //- Construct from local max size.
    //  Does communication with given communicator and message tag
    gkoGlobalIndex(const label localSize,
                   const int tag,       //!< message tag
                   const label comm,    //!< communicator
                   const bool parallel  //!< use parallel comms
    );

    //- Copy construct from list of labels
    explicit gkoGlobalIndex(const labelUList &offsets);

    //- Move construct from list of labels
    explicit gkoGlobalIndex(labelList &&offsets);

    //- Construct from Istream
    explicit gkoGlobalIndex(Istream &is);


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
    template <class Container, class Value>
    static void gather(
        const labelUList &offsets, const label comm, const Container &procIDs,
        const gko::Array<Value> &localDataIn, gko::Array<Value> &globalDataOut,
        const int tag = UPstream::msgType(),
        const Pstream::commsTypes commsType = Pstream::commsTypes::nonBlocking)
    {
        if (Pstream::master()) {
            // Assign my local data
            std::cout << "create view  " << endl;
            gko::Array<Value>::view(gko::ReferenceExecutor::create(),
                                    localDataIn.get_num_elems(),
                                    globalDataOut.get_data()) = localDataIn;
            std::cout << "create view done  " << endl;

            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                for (label i = 1; i < procIDs.size(); ++i) {
                    label nSubElems = offsets[i + 1] - offsets[i];
                    IPstream::read(
                        commsType, procIDs[i],
                        reinterpret_cast<char *>(
                            &globalDataOut.get_data()[offsets[i + 1]]),
                        nSubElems * sizeof(Value), tag, comm);
                }
            } else {
                label startOfRequests = Pstream::nRequests();

                // Set up reads
                std::cout << "start reads  " << endl;
                for (label i = 1; i < procIDs.size(); ++i) {
                    label nSubElems = offsets[i + 1] - offsets[i];
                    auto sub_view = gko::Array<Value>::view(
                        gko::ReferenceExecutor::create(), nSubElems,
                        &globalDataOut.get_data()[offsets[i]]);
                    IPstream::read(
                        commsType, procIDs[i],
                        reinterpret_cast<char *>(sub_view.get_data()),
                        nSubElems * sizeof(Value), tag, comm);
                }
                std::cout << "start reads done " << endl;

                // Wait for all to finish
                Pstream::waitRequests(startOfRequests);
            }
        } else {
            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                OPstream::write(commsType, procIDs[0],
                                reinterpret_cast<const char *>(
                                    localDataIn.get_const_data()),
                                localDataIn.get_num_elems() * sizeof(Value),
                                tag, comm);
            } else {
                label startOfRequests = Pstream::nRequests();
                // Set up write
                std::cout << "start write  " << endl;
                OPstream::write(commsType, procIDs[0],
                                reinterpret_cast<const char *>(
                                    localDataIn.get_const_data()),
                                localDataIn.get_num_elems() * sizeof(Value),
                                tag, comm);
                std::cout << "start write done  " << endl;

                // Wait for all to finish
                Pstream::waitRequests(startOfRequests);
            }
        }
    };

    //- Collect data in processor order on master (== procIDs[0]).
    //  Needs offsets only on master.
    template <class Container, class Value>
    void gather(const label comm, const Container &procIDs,
                const gko::Array<Value> &localDataIn,
                gko::Array<Value> &globalDataOut,
                const int tag = UPstream::msgType(),
                const Pstream::commsTypes commsType =
                    Pstream::commsTypes::nonBlocking) const
    {
        gather(offsets_, comm, procIDs, localDataIn, globalDataOut, tag,
               commsType);
    }

    //- Collect data in processor order on master.
    //  Does communication with default communicator and message tag.
    template <class Value>
    void gather(const gko::Array<Value> &localDataIn,
                gko::Array<Value> &globalDataOut,
                const int tag = UPstream::msgType(),
                const Pstream::commsTypes commsType =
                    Pstream::commsTypes::nonBlocking) const
    {
        gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm),
               localDataIn, globalDataOut, tag, commsType);
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

    // friend Istream &operator>>(Istream &is, gkoGlobalIndex &gi);
    // friend Ostream &operator<<(Ostream &os, const gkoGlobalIndex &gi);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
