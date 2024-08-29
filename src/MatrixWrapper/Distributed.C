// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/MatrixWrapper/Distributed.H"

std::vector<std::unique_ptr<gko::LinOp>> detail::generate_inner_linops(
    word matrix_format, bool fuse, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const SparsityPattern> sparsity)
{
    std::vector<std::unique_ptr<gko::LinOp>> lin_ops;
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
            return gko::matrix::Coo<scalar, label>::create(
                exec, sparsity->dim, coeffs, col_idxs_view, row_idxs_view);
        };
        lin_ops.push_back(create_function(matrix_format));
    }
    return lin_ops;
}


// auto local_dim = local_sparsity->dim;
// auto non_local_dim = non_local_sparsity->dim;
// auto local_interfaces = local_sparsity->interface_spans;
// auto non_local_interfaces = non_local_sparsity->interface_spans;

// if (matrix_format == "Ell") {
//     using mtx_type = gko::matrix::Ell<scalar, label>;

//     return DistMtxType::create(
//         exec, comm,
//         CombinationMatrix<scalar, label, mtx_type>::create(
//             exec, local_dim, local_interfaces),
//         CombinationMatrix<scalar, label, mtx_type>::create(
//             exec, non_local_dim, non_local_interfaces));
// }
// if (matrix_format == "Csr") {
//     using mtx_type = gko::matrix::Csr<scalar, label>;

//     return DistMtxType::create(
//         exec, comm,
//         CombinationMatrix<scalar, label, mtx_type>::create(
//             exec, local_dim, local_interfaces),
//         CombinationMatrix<scalar, label, mtx_type>::create(
//             exec, non_local_dim, non_local_interfaces));
// }
// if (matrix_format == "Coo") {
//     using mtx_type = gko::matrix::Coo<scalar, label>;

//     return DistMtxType::create(
//         exec, comm,
//         CombinationMatrix<scalar, label, mtx_type>::create(
//             exec, local_dim, local_interfaces),
//         CombinationMatrix<scalar, label, mtx_type>::create(
//             exec, non_local_dim, non_local_interfaces));
// }

//     FatalErrorInFunction << "Matrix format " << matrix_format
//                          << " not supported! Supported formats: Csr, Ell,
//                          Coo"
//                          << abort(FatalError);
//     return {};
// }
