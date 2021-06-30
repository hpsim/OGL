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
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.


Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    common.C

\*---------------------------------------------------------------------------*/

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <type_traits>

#include "common.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b, const word time)
{
    std::string fn_mtx{time + "_" + fieldName + "_A.mtx"};
    std::ofstream stream{fn_mtx};
    std::cerr << "Writing " << fn_mtx << std::endl;
    gko::write(stream, A, gko::layout_type::coordinate);

    std::string fn_b{time + "_" + fieldName + "_b.mtx"};
    std::ofstream stream_b{fn_b};
    std::cerr << "Writing " << fn_b << std::endl;
    gko::write(stream_b, b);

    std::string fn_x{time + "_" + fieldName + "_x0.mtx"};
    std::ofstream stream_x{fn_x};
    std::cerr << "Writing " << fn_x << std::endl;
    gko::write(stream_x, x);
};

}  // namespace Foam
