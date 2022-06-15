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
#include "fvCFD.H"
#include "labelList.H"

//#include "CompactListList.H"
#include <memory>
#include "DynamicList.H"
#include "Pstream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {
using val_array = gko::array<scalar>;
using idx_array = gko::array<label>;


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
        std::shared_ptr<const gko::array<Value>> localDataIn,
        std::shared_ptr<gko::array<Value>> globalDataOut,
        const int tag = UPstream::msgType(),
        const Pstream::commsTypes commsType = Pstream::commsTypes::nonBlocking)
    {
        if (Pstream::master()) {
            // Assign my local data
            gko::array<Value>::view(
                gko::ReferenceExecutor::create(), localDataIn->get_num_elems(),
                globalDataOut->get_data()) = *localDataIn.get();
            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                for (label i = 1; i < procIDs.size(); ++i) {
                    label nSubElems = offsets[i + 1] - offsets[i];
                    IPstream::read(commsType, procIDs[i],
                                   reinterpret_cast<char *>(
                                       &globalDataOut->get_data()[offsets[i]]),
                                   nSubElems * sizeof(Value), tag, comm);
                }
            } else {
                label startOfRequests = Pstream::nRequests();

                // Set up reads
                for (label i = 1; i < procIDs.size(); ++i) {
                    label nSubElems = offsets[i + 1] - offsets[i];
                    auto sub_view = gko::array<Value>::view(
                        gko::ReferenceExecutor::create(), nSubElems,
                        &globalDataOut->get_data()[offsets[i]]);
                    IPstream::read(
                        commsType, procIDs[i],
                        reinterpret_cast<char *>(sub_view.get_data()),
                        nSubElems * sizeof(Value), tag, comm);
                }

                // Wait for all to finish
                Pstream::waitRequests(startOfRequests);
            }
        } else {
            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                OPstream::write(commsType, procIDs[0],
                                reinterpret_cast<const char *>(
                                    localDataIn->get_const_data()),
                                localDataIn->get_num_elems() * sizeof(Value),
                                tag, comm);
            } else {
                label startOfRequests = Pstream::nRequests();
                // Set up write
                OPstream::write(commsType, procIDs[0],
                                reinterpret_cast<const char *>(
                                    localDataIn->get_const_data()),
                                localDataIn->get_num_elems() * sizeof(Value),
                                tag, comm);

                // Wait for all to finish
                Pstream::waitRequests(startOfRequests);
            }
        }
    }


    //- Collect data in processor order on master (== procIDs[0]).
    //  Needs offsets only on master.
    template <class Container, class Value>
    void gather(const label comm, const Container &procIDs,
                std::shared_ptr<const gko::array<Value>> localDataIn,
                std::shared_ptr<gko::array<Value>> globalDataOut,
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
    void gather(std::shared_ptr<const gko::array<Value>> localDataIn,
                std::shared_ptr<gko::array<Value>> globalDataOut,
                const int tag = UPstream::msgType(),
                const Pstream::commsTypes commsType =
                    Pstream::commsTypes::nonBlocking) const
    {
        gather(UPstream::worldComm, UPstream::procID(UPstream::worldComm),
               localDataIn, globalDataOut, tag, commsType);
    }


    //- Distribute data in processor order. Requires fld to be sized!
    template <class Container, class Value>
    static void scatter(
        const labelUList &offsets, const label comm, const Container &procIDs,
        const gko::array<Value> &globalDataIn, gko::array<Value> &localDataOut,
        const int tag = UPstream::msgType(),
        const Pstream::commsTypes commsType = Pstream::commsTypes::nonBlocking)
    {
        if (Pstream::master()) {
            // copy its own data to local array.
            localDataOut = gko::detail::array_const_cast(
                gko::array<Value>::const_view(gko::ReferenceExecutor::create(),
                                              localDataOut.get_num_elems(),
                                              globalDataIn.get_const_data()));
            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                for (label i = 1; i < procIDs.size(); ++i) {
                    label nSubElems = offsets[i + 1] - offsets[i];
                    OPstream::write(
                        commsType, procIDs[i],
                        reinterpret_cast<const char *>(
                            &globalDataIn.get_const_data()[offsets[i]]),
                        nSubElems * sizeof(Value), tag, comm);
                }
            } else {
                // nonBlocking
                label startOfRequests = Pstream::nRequests();

                // Set up writes
                for (label i = 1; i < procIDs.size(); ++i) {
                    label nSubElems = offsets[i + 1] - offsets[i];
                    OPstream::write(
                        commsType, procIDs[i],
                        reinterpret_cast<const char *>(
                            &globalDataIn.get_const_data()[offsets[i]]),
                        nSubElems * sizeof(Value), tag, comm);
                }

                // Wait for all to finish
                Pstream::waitRequests(startOfRequests);
            }
        } else {
            if (commsType == Pstream::commsTypes::scheduled ||
                commsType == Pstream::commsTypes::blocking) {
                IPstream::read(
                    commsType, procIDs[0],
                    reinterpret_cast<char *>(localDataOut.get_data()),
                    localDataOut.get_num_elems() * sizeof(Value), tag, comm);
            } else {
                label startOfRequests = Pstream::nRequests();
                IPstream::read(
                    commsType, procIDs[0],
                    reinterpret_cast<char *>(localDataOut.get_data()),
                    localDataOut.get_num_elems() * sizeof(Value), tag, comm);
                Pstream::waitRequests(startOfRequests);
            }
        }
    }


    //- Distribute data in processor order. Requires fld to be sized!
    template <class Container, class Value>
    void scatter(const label comm, const Container &procIDs,
                 const gko::array<Value> &globalDataIn,
                 gko::array<Value> &localDataOut,
                 const int tag = UPstream::msgType(),
                 const Pstream::commsTypes commsType =
                     Pstream::commsTypes::nonBlocking) const
    {
        scatter(offsets_, comm, procIDs, globalDataIn, localDataOut, tag,
                commsType);
    }


    //- Distribute data in processor order. Requires fld to be sized!
    //  Does communication with default communicator and message tag.
    template <class Value>
    void scatter(const gko::array<Value> &globalDataIn,
                 gko::array<Value> &localDataOut,
                 const int tag = UPstream::msgType(),
                 const Pstream::commsTypes commsType =
                     Pstream::commsTypes::nonBlocking) const
    {
        scatter(offsets_, UPstream::worldComm,
                UPstream::procID(UPstream::worldComm), globalDataIn,
                localDataOut, tag, commsType);
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// //

}  // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
