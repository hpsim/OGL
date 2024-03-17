// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ginkgo/ginkgo.hpp>
#include <iomanip>
#include <map>
#include <type_traits>

#include <filesystem>
#include "common.H"

#include "MatrixWrapper/GkoCombinationMatrix/GkoCombinationMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

void export_x(const std::string fn, const gko::matrix::Dense<scalar> *x)
{
    std::ofstream stream_x{fn};
    LOG_1(1, "Writing " + fn)
    gko::write(stream_x, x);
}

void export_x(const std::string fn, const gko::matrix::Csr<scalar> *A)
{
    LOG_1(1, "Writing " + fn)
    std::ofstream stream{fn};
    gko::write(stream, A, gko::layout_type::coordinate);
}

void export_vec(const word fieldName, const gko::matrix::Dense<scalar> *x,
                const objectRegistry &db)
{
    std::string folder{db.time().timePath()};
    std::string fn{folder + "/" + fieldName + "_b_.mtx"};
    export_x(fn, x);
}

template <typename Mtx>
void export_mtx(const word fieldName,
                std::vector<std::shared_ptr<const gko::LinOp>> &As,
                const objectRegistry &db)
{
    std::string folder{db.time().timePath()};
    std::filesystem::create_directories(folder);

    for (int i = 0; i < As.size(); i++) {
        std::string fn{folder + "/" + fieldName + "_A_" + std::to_string(i) +
                       ".mtx"};
        std::cout << "exporting " << fn << std::endl;
        std::ofstream stream{fn};
        stream << std::setprecision(15);

        gko::write(stream, gko::as<Mtx>(As[i]).get());
    }
}

template void export_mtx<gko::matrix::Coo<scalar, label>>(
    const word fieldName, std::vector<std::shared_ptr<const gko::LinOp>> &As,
    const objectRegistry &db);

template void export_mtx<gko::matrix::Csr<scalar, label>>(
    const word fieldName, std::vector<std::shared_ptr<const gko::LinOp>> &As,
    const objectRegistry &db);
template void export_mtx<gko::matrix::Ell<scalar, label>>(
    const word fieldName, std::vector<std::shared_ptr<const gko::LinOp>> &As,
    const objectRegistry &db);


void export_system(const word fieldName, const gko::matrix::Csr<scalar> *A,
                   const gko::matrix::Dense<scalar> *x,
                   const gko::matrix::Dense<scalar> *b, const word time)
{
    system("mkdir -p export/" + time);
    std::string fn_mtx{"export/" + time + "/" + fieldName + "_A.mtx"};
    export_x(fn_mtx, A);

    std::string fn_b{"export/" + time + "/" + fieldName + "_b.mtx"};
    export_x(fn_b, b);

    std::string fn_x{"export/" + time + "/" + fieldName + "_x0.mtx"};
    export_x(fn_x, x);
}

void set_gko_solver_property(word sys_matrix_name, const objectRegistry &db,
                             const word key, label value)
{
    const word solvPropsDict = sys_matrix_name + "_gkoSolverProperties";
    if (db.foundObject<regIOobject>(solvPropsDict)) {
        const_cast<objectRegistry &>(db)
            .lookupObjectRef<IOdictionary>(solvPropsDict)
            .set<label>(key, value);
    } else {
        auto gkoSolverProperties =
            new IOdictionary(IOobject(solvPropsDict, fileName("None"), db,
                                      IOobject::NO_READ, IOobject::NO_WRITE));
        gkoSolverProperties->add(key, value, true);
    }
}

template <typename T>
T get_gko_solver_property(word sys_matrix_name_, word key,
                          const objectRegistry &db, T in)
{
    const word solvPropsDict = sys_matrix_name_ + "_gkoSolverProperties";
    if (db.foundObject<regIOobject>(solvPropsDict)) {
        label pre_solve_iters = db.lookupObject<IOdictionary>(solvPropsDict)
                                    .lookupOrDefault<T>(key, in);
        return pre_solve_iters;
    }
    return in;
}

void set_next_caching(word sys_matrix_name, const objectRegistry &db,
                      label caching)
{
    set_gko_solver_property(sys_matrix_name, db, "preconditionerCaching",
                            caching);
}

label get_next_caching(word sys_matrix_name, const objectRegistry &db)
{
    return get_gko_solver_property(sys_matrix_name, "preconditionerCaching", db,
                                   label(0));
}

void set_solve_prev_rel_res_cost(const word sys_matrix_name,
                                 const objectRegistry &db,
                                 scalar prev_solve_rel_res_cost)
{
    set_gko_solver_property(sys_matrix_name, db, "_prev_solve",
                            prev_solve_rel_res_cost);
}

scalar get_solve_prev_rel_res_cost(const word sys_matrix_name,
                                   const objectRegistry &db)
{
    return get_gko_solver_property(sys_matrix_name, "_prev_solve", db,
                                   scalar(0.0));
}

void set_solve_prev_iters(word sys_matrix_name, const objectRegistry &db,
                          label prev_solve_iters, const bool is_final)
{
    const word iters_name =
        (is_final) ? "prevSolveIters_final" : "prevSolveIters";
    set_gko_solver_property(sys_matrix_name, db, iters_name, prev_solve_iters);
}

label get_solve_prev_iters(word sys_matrix_name, const objectRegistry &db,
                           const bool is_final)
{
    const word iters_name =
        (is_final) ? "prevSolveIters_final" : "prevSolveIters";
    return get_gko_solver_property(sys_matrix_name, iters_name, db, label(1));
}

std::ostream &operator<<(std::ostream &os,
                         const std::shared_ptr<gko::matrix::Dense<scalar>> in)
{
    auto ref_exec = gko::ReferenceExecutor::create();
    auto array = in->clone(ref_exec);
    label size = array->get_size()[0];
    os << size << " elements [";
    if (size > 100) {
        for (label i = 0; i < 9; i++) {
            os << array->at(i) << ", ";
        }
        os << array->at(10) << " ... ";
        for (label i = size - 9; i < size - 1; i++) {
            os << array->at(i) << ", ";
        }
        os << array->at(size - 1) << "]\n";
    } else {
        for (label i = 0; i < size - 1; i++) {
            os << "(" << i << ", " << array->at(i) << ") ";
        }
        os << "(" << size - 1 << ", " << array->at(size - 1) << ")]\n";
    }
    return os;
}
}  // namespace Foam
