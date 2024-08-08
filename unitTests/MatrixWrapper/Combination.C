// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gtest/gtest.h>

#include "OGL/MatrixWrapper/Combination.H"

#include "fvCFD.H"

#include <ginkgo/ginkgo.hpp>


TEST(Combination, CanCreateEmptyCombination)
{
    auto exec = gko::share(gko::ReferenceExecutor::create());
    auto cmb = CombinationMatrix<gko::matrix::Coo<scalar, label>>::create(exec);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmb->get_coefficients().size(), 0);
    ASSERT_EQ(cmb->get_operators().size(), 0);
}


TEST(Combination, CanCreateCombinationWithLinOpVector)
{
    auto dim = gko::dim<2>{5, 5};
    gko::matrix_data<double, int> m1(dim, {{0, 0, 2}, {1, 1, 0}, {2, 3, 5}});

    auto exec = gko::share(gko::ReferenceExecutor::create());
    auto m1linop =
        gko::share(gko::matrix::Csr<scalar, label>::create(exec, dim));
    m1linop->read(m1);

    std::vector<std::shared_ptr<const gko::LinOp>> linops{m1linop, m1linop};

    auto cmb = CombinationMatrix<gko::matrix::Csr<scalar, label>>::create(
        exec, dim, linops);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);

    auto b = gko::matrix::Dense<scalar>::create(exec, gko::dim<2>{5, 1});
    b->fill(1.0);
    auto x = gko::matrix::Dense<scalar>::create(exec, gko::dim<2>{5, 1});
    x->fill(0.0);
    cmb->apply(b, x);

    ASSERT_EQ(x->at(0, 0), 4.0);
    ASSERT_EQ(x->at(1, 0), 0.0);
    ASSERT_EQ(x->at(2, 0), 10.0);
    ASSERT_EQ(x->at(3, 0), 0.0);
    ASSERT_EQ(x->at(4, 0), 0.0);
}

TEST(Combination, CanCreateConvertToCsr)
{
    auto dim = gko::dim<2>{5, 5};
    gko::matrix_data<double, int> m1(dim, {{0, 0, 2}, {1, 1, 0}, {2, 3, 5}});
    gko::matrix_data<double, int> m2(dim, {{3, 3, 1}, {4, 4, 2}});

    auto exec = gko::share(gko::ReferenceExecutor::create());
    auto m1linop =
        gko::share(gko::matrix::Csr<scalar, label>::create(exec, dim));
    m1linop->read(m1);

    auto m2linop =
        gko::share(gko::matrix::Csr<scalar, label>::create(exec, dim));
    m2linop->read(m2);

    std::vector<std::shared_ptr<const gko::LinOp>> linops{m1linop, m2linop};

    auto cmb = CombinationMatrix<gko::matrix::Csr<scalar, label>>::create(
        exec, dim, linops);

    auto out = CombinationMatrix<gko::matrix::Csr<scalar, label>>::create(
        exec);

    cmb->convert_to(out);
}
