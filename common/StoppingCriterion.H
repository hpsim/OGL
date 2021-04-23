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

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/

#ifndef OGL_STOPPING_CRITERION_H
#define OGL_STOPPING_CRITERION_H

#include <ginkgo/ginkgo.hpp>
#include "fvCFD.H"

namespace Foam {

class OpenFOAMStoppingCriterion
    : public gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                          gko::stop::Criterion> {
    friend class gko::EnablePolymorphicObject<OpenFOAMStoppingCriterion,
                                              gko::stop::Criterion>;
    using Criterion = gko::stop::Criterion;
    using vec = gko::matrix::Dense<scalar>;

private:
    label iter_ = 0;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Boolean set by the user to stop the iteration process
         */
        // TODO check why GKO_FACTORY_PARAMETER_SCALAR does not work
        scalar GKO_FACTORY_PARAMETER(openfoam_absolute_tolerance, 1.0e-6);

        scalar GKO_FACTORY_PARAMETER(openfoam_relative_tolerance, 0.0);

        scalar GKO_FACTORY_PARAMETER(openfoam_norm_factor, 1.0);

        label GKO_FACTORY_PARAMETER(openfoam_minIter, 0);

        label GKO_FACTORY_PARAMETER(openfoam_maxIter, 0);
    };

    GKO_ENABLE_CRITERION_FACTORY(OpenFOAMStoppingCriterion, parameters,
                                 Factory);

    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                    gko::Array<gko::stopping_status> *stop_status,
                    bool *one_changed,
                    const Criterion::Updater &updater) override
    {
        iter_++;

        auto *dense_r = gko::as<vec>(updater.residual_);
        auto exec = dense_r->get_executor();
        auto res = vec::create(exec, gko::dim<2>{1, 1});
        auto vec_abs = gko::clone(dense_r);
        const label cells = dense_r->get_size()[0];
        vec_abs->compute_absolute_inplace();
        auto unit = vec::create(dense_r->get_executor(),
                                gko::dim<2>{(unsigned long)cells, 1});
        unit->fill(1.0);

        const auto unit_t = unit->transpose();

        unit_t->apply(gko::lend(vec_abs), gko::lend(res));

        auto res_host = vec::create(exec->get_master(), gko::dim<2>{1, 1});
        res_host->copy_from(gko::lend(res));

        bool result = false;
        scalar normalised_res_norm =
            res_host->at(0) / parameters_.openfoam_norm_factor;
        // printf("normalised res norm %f\n", normalised_res_norm);
        // printf("iter %d\n", iter_);

        if ((normalised_res_norm < parameters_.openfoam_absolute_tolerance) &&
            (iter_ >= parameters_.openfoam_minIter)) {
            result = true;
        }

        if (iter_ >= parameters_.openfoam_maxIter) {
            result = true;
        }

        if (result) {
            this->set_all_statuses(stoppingId, setFinalized, stop_status);
            *one_changed = true;
        }

        return result;
    }

    explicit OpenFOAMStoppingCriterion(
        std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<OpenFOAMStoppingCriterion, Criterion>(
              std::move(exec))
    {}

    explicit OpenFOAMStoppingCriterion(const Factory *factory,
                                       const gko::stop::CriterionArgs &args)

        : EnablePolymorphicObject<OpenFOAMStoppingCriterion, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}
};
}  // namespace Foam

#endif
