// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "StoppingCriterion.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

void StoppingCriterion::OpenFOAMDistStoppingCriterion::compute_Axref_dist(
    size_t global_size, size_t local_size,
    std::shared_ptr<const gko::Executor> device_exec,
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<const dist_vec> x, std::shared_ptr<dist_vec> res) const
{
    auto xAvg =
        gko::initialize<gko::matrix::Dense<scalar>>(1, {0}, device_exec);
    x->compute_mean(xAvg.get());

    auto xAvg_host = gko::initialize<gko::matrix::Dense<scalar>>(
        1, {0}, device_exec->get_master());
    xAvg->move_to(xAvg_host);
    auto xAvg_vec = gko::share(dist_vec::create(
        device_exec, x->get_communicator(), gko::dim<2>{global_size, 1},
        gko::dim<2>{local_size, 1}));
    xAvg_vec->fill(xAvg_host->at(0));

    gkomatrix->apply(xAvg_vec.get(), res.get());
}

scalar
StoppingCriterion::OpenFOAMDistStoppingCriterion::compute_normfactor_dist(
    std::shared_ptr<const gko::Executor> device_exec, const dist_vec *r,
    std::shared_ptr<const gko::LinOp> gkomatrix,
    std::shared_ptr<const dist_vec> x, std::shared_ptr<const dist_vec> b) const
{
    // TODO store colA vector
    auto comm = x->get_communicator();

    gko::dim<2> local_size = x->get_local_vector()->get_size();
    gko::dim<2> global_size = x->get_size();

    auto Axref = gko::share(
        dist_vec::create(device_exec, comm, global_size, local_size));

    compute_Axref_dist(global_size[0], local_size[0], device_exec, gkomatrix, x,
                       Axref);

    auto unity =
        gko::initialize<gko::matrix::Dense<scalar>>(1, {1.0}, device_exec);

    auto b_sub_xstar = b->clone();
    b_sub_xstar->sub_scaled(unity.get(), Axref.get());

    auto norm_part2 = b_sub_xstar->compute_absolute();

    b_sub_xstar->sub_scaled(unity.get(), r);
    b_sub_xstar->compute_absolute_inplace();

    b_sub_xstar->add_scaled(unity.get(), norm_part2.get());
    auto res = vec::create(device_exec, gko::dim<2>{1});
    b_sub_xstar->compute_norm1(res.get());

    auto res_host = vec::create(device_exec->get_master(), gko::dim<2>{1});
    res_host->copy_from(res.get());

    return res_host->get_values()[0] + SMALL;
}

bool StoppingCriterion::OpenFOAMDistStoppingCriterion::check_impl(
    gko::uint8 stoppingId, bool setFinalized,
    gko::array<gko::stopping_status> *stop_status, bool *one_changed,
    const Criterion::Updater &updater)
{
    // Dont check residual norm before minIter is reached
    if (*(parameters_.iter) > 0 &&
        *(parameters_.iter) < parameters_.openfoam_minIter) {
        *(parameters_.iter) += 1;
        return false;
    }

    // Only check residual for every frequency iteration
    if (*(parameters_.iter) % parameters_.frequency != 0) {
        *(parameters_.iter) += 1;
        return false;
    }

    auto start_eval = std::chrono::steady_clock::now();
    const auto exec = this->get_executor();

    auto *dense_r = gko::as<dist_vec>(updater.residual_);
    auto norm1 = vec::create(exec, gko::dim<2>{1});
    dense_r->compute_norm1(norm1.get());
    auto norm1_host = vec::create(exec->get_master(), gko::dim<2>{1});
    norm1_host->copy_from(norm1.get());
    scalar residual_norm = norm1_host->at(0);

    bool result = false;

    // Store initial residual
    if (*(parameters_.iter) == 0) {
        //
        if (eval_norm_factor_) {
            norm_factor_ =
                compute_normfactor_dist(exec, dense_r, parameters_.gkomatrix,
                                        parameters_.x, parameters_.b);
        }

        *(parameters_.init_residual_norm) = residual_norm / norm_factor_;
    }

    residual_norm /= norm_factor_;

    if (parameters_.export_res) {
        parameters_.residual_norms->at(*(parameters_.iter)) = residual_norm;
    }

    *(parameters_.residual_norm) = residual_norm;

    scalar init_residual = *(parameters_.init_residual_norm);

    // stop if maximum number of iterations was reached
    if (*(parameters_.iter) >= parameters_.openfoam_maxIter) {
        result = true;
    }
    // check if absolute tolerance is hit
    if (residual_norm < parameters_.openfoam_absolute_tolerance) {
        result = true;
    }
    // check if relative tolerance is hit
    if (parameters_.openfoam_relative_tolerance > 0 &&
        residual_norm <
            parameters_.openfoam_relative_tolerance * init_residual) {
        result = true;
    }

    if (result) {
        this->set_all_statuses(stoppingId, setFinalized, stop_status);
        *one_changed = true;
    }

    *(parameters_.iter) += 1;

    auto end_eval = std::chrono::steady_clock::now();
    *(parameters_.time) = std::chrono::duration_cast<std::chrono::microseconds>(
                              end_eval - start_eval)
                              .count() /
                          1.0;
    return result;
}


}  // namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //


// ************************************************************************* //
