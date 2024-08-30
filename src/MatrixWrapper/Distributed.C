// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Distributed.H"

std::vector<std::shared_ptr<const gko::LinOp>> detail::generate_inner_linops(
    word matrix_format, bool fuse, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const SparsityPattern> sparsity)
{
    std::vector<std::shared_ptr<const gko::LinOp>> lin_ops;
    for (int i = 0; i < sparsity->spans.size(); i++) {
        auto [begin, end] = sparsity->spans[i];
        gko::array<scalar> coeffs(exec, end - begin);
        coeffs.fill(0.0);

        auto row_idxs_view =
            gko::array<label>::view(exec->get_master(), end - begin,
                                    sparsity->row_idxs.get_data() + begin);
        auto col_idxs_view =
            gko::array<label>::view(exec->get_master(), end - begin,
                                    sparsity->col_idxs.get_data() + begin);

        auto create_function = [&](word format) {
            return gko::share(gko::matrix::Coo<scalar, label>::create(
                exec, sparsity->dim, coeffs, col_idxs_view, row_idxs_view));
        };
        lin_ops.push_back(create_function(matrix_format));
    }
    return lin_ops;
}
