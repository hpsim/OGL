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

void export_x(const std::string fn, const vec *x)
{
    std::ofstream stream_x{fn};
    LOG_1(1, "Writing " + fn)
    gko::write(stream_x, x);
};

void export_x(const std::string fn, const mtx *A)
{
    LOG_1(1, "Writing " + fn)
    std::ofstream stream{fn};
    gko::write(stream, A, gko::layout_type::coordinate);
};

void export_vec(const word fieldName, const vec *x, const word time)
{
    system("mkdir -p export/" + time);
    std::string fn_mtx{"export/" + time + "/" + fieldName + ".mtx"};
    export_x(fn_mtx, x);
};

void export_system(const word fieldName, const mtx *A, const vec *x,
                   const vec *b, const word time)
{
    system("mkdir -p export/" + time);
    std::string fn_mtx{"export/" + time + "/" + fieldName + "_A.mtx"};
    export_x(fn_mtx, A);

    std::string fn_b{"export/" + time + "/" + fieldName + "_b.mtx"};
    export_x(fn_b, b);

    std::string fn_x{"export/" + time + "/" + fieldName + "_x0.mtx"};
    export_x(fn_x, x);
};


void set_solve_prev_iters(word sys_matrix_name_, const objectRegistry &db,
                          label prev_solve_iters)
{
    const word solvPropsDict = sys_matrix_name_ + "gkoSolverProperties";
    if (db.foundObject<regIOobject>(solvPropsDict)) {
        const_cast<objectRegistry &>(db)
            .lookupObjectRef<IOdictionary>(solvPropsDict)
            .set<label>("prev_solve_iters", prev_solve_iters);
    } else {
        auto gkoSolverProperties =
            new IOdictionary(IOobject(solvPropsDict, fileName("None"), db,
                                      IOobject::NO_READ, IOobject::NO_WRITE));
        gkoSolverProperties->add("prev_solve_iters", prev_solve_iters, true);
    }
}

std::ostream &operator<<(
    std::ostream &os,
    const std::shared_ptr<gko::matrix::Dense<scalar>> array_in)
{
    auto array = array_in->clone();
    auto ref_exec = gko::ReferenceExecutor::create();
    array->set_executor(ref_exec);
    label size = array->get_size();
    os << size << " elements [";
    if (size > 40) {
        for (label i = 0; i < 9; i++) {
            os << array->at(i) << ", "
        }
        os << array->at(10) << " ... ";
        for (label i = size - 9; i < size - 1; i++) {
            os << array->at(i) << ", "
        }
        os << array->at(size - 1) << "]\n"
    } else {
        for (label i = 0; i < size; i++) {
            os << array->at(i) << ", "
        }
    }
}

label get_solve_prev_iters(word sys_matrix_name_, const objectRegistry &db)
{
    const word solvPropsDict = sys_matrix_name_ + "gkoSolverProperties";
    if (db.foundObject<regIOobject>(solvPropsDict)) {
        label pre_solve_iters =
            db.lookupObject<IOdictionary>(solvPropsDict)
                .lookupOrDefault<label>("prev_solve_iters", 0);
        return pre_solve_iters;
    }
    return 0;
}
}  // namespace Foam
