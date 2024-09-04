// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Combination.H"


std::shared_ptr<gko::matrix::Coo<scalar, label>>
detail::convert_combination_to_coo(std::shared_ptr<const gko::Executor> exec,
                                   std::shared_ptr<const gko::LinOp> in)
{
    auto out = gko::share(gko::matrix::Coo<scalar, label>::create(exec));
    gko::as<CombinationMatrix<gko::matrix::Coo<scalar, label>>>(in)->convert_to(
        out.get());

    return out;
}
