// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gtest/gtest.h>

#include "OGL/MatrixWrapper/Combination.H"

#include "fvCFD.H"

#include <ginkgo/ginkgo.hpp>

#include <fstream>

#include <string>

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

TEST(Combination, CanCreateCombinationMatrixWithSparseRealMatrix)
{
    // Arrange
    using ValueType = scalar;
    using IndexType = label;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    
    int dim_size = 5;
    auto dim = gko::dim<2>{dim_size, dim_size};

    auto exec = gko::share(gko::ReferenceExecutor::create());

    auto m1linop = gko::share(
        gko::read<mtx>(std::ifstream("data/A_sparse.mtx"), exec));
    
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);

    std::vector<std::shared_ptr<const gko::LinOp>> linops{m1linop, m1linop};

    // Act
    auto cmb = CombinationMatrix<gko::matrix::Csr<scalar, label>>::create(
        exec, dim, linops);

    // Assert
    ASSERT_EQ(cmb->get_size(), gko::dim<2>(dim_size, dim_size));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);

    // Act
    auto x = gko::matrix::Dense<scalar>::create(exec, gko::dim<2>{dim_size, 1});
    x->fill(0.0);
    cmb->apply(b, x);

    // Assert
    ASSERT_EQ(x->at(0, 0), 38.0);
    ASSERT_EQ(x->at(1, 0), 0.0);
    ASSERT_EQ(x->at(2, 0), 2.0);
    ASSERT_EQ(x->at(3, 0), -1414.4);
    ASSERT_EQ(x->at(4, 0), 96.0);
}


TEST(Combination, CanCreateCombinationMatrixWithOpenFOAMLDUMatrix)
{
    // Arrange
    using ValueType = scalar;
    using IndexType = label;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    
    int dim_size = 5;
    auto dim = gko::dim<2>{dim_size, dim_size};

    auto exec = gko::share(gko::ReferenceExecutor::create());

    auto Llinop = gko::share(
        gko::read<mtx>(std::ifstream("data/L.mtx"), exec));

    auto Dlinop = gko::share(
        gko::read<mtx>(std::ifstream("data/D.mtx"), exec));
    
    auto Ulinop = gko::share(
        gko::read<mtx>(std::ifstream("data/U.mtx"), exec));

    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);

    std::vector<std::shared_ptr<const gko::LinOp>> linops{Llinop, Dlinop, Ulinop};

    // Act
    auto cmb = CombinationMatrix<gko::matrix::Csr<scalar, label>>::create(
        exec, dim, linops);

    // Assert
    ASSERT_EQ(cmb->get_size(), gko::dim<2>(dim_size, dim_size));
    ASSERT_EQ(cmb->get_coefficients().size(), 3);
    ASSERT_EQ(cmb->get_operators().size(), 3);

    // Act
    auto x = gko::matrix::Dense<scalar>::create(exec, gko::dim<2>{dim_size, 1});
    x->fill(0.0);
    cmb->apply(b, x);

    // Assert
    ASSERT_EQ(x->at(0, 0), 19.0);
    ASSERT_EQ(x->at(1, 0), 0.0);
    ASSERT_EQ(x->at(2, 0), 1.0);
    ASSERT_EQ(x->at(3, 0), -707.2);
    ASSERT_EQ(x->at(4, 0), 48.0);
}

TEST(Combination, CanConvertToCoo)
{
    // Arrange
    using ValueType = scalar;
    using IndexType = label;
    using InputFormat = gko::matrix::Csr<ValueType, IndexType>;
    using OutputFormat = gko::matrix::Coo<ValueType, IndexType>;

    int dim_size = 5;

    auto dim = gko::dim<2>{dim_size, dim_size};
    gko::matrix_data<double, int> m1(dim, {{0, 0, 2}, {1, 1, 0}, {2, 3, 5}});
    gko::matrix_data<double, int> m2(dim, {{3, 3, 1}, {4, 4, 2}});

    auto exec = gko::share(gko::ReferenceExecutor::create());
    auto m1linop =
        gko::share(InputFormat::create(exec, dim));
    m1linop->read(m1);

    auto m2linop =
        gko::share(InputFormat::create(exec, dim));
    m2linop->read(m2);

    std::vector<std::shared_ptr<const gko::LinOp>> linops{m1linop, m2linop};

    auto cmb = CombinationMatrix<InputFormat>::create(
        exec, dim, linops);

    auto out = OutputFormat::create(
        exec);

    // Act
    cmb->convert_to(out.get());

    // Assert
    ASSERT_EQ(out->get_values()[0], 2);
    ASSERT_EQ(out->get_values()[1], 0);
    ASSERT_EQ(out->get_values()[2], 5);
    ASSERT_EQ(out->get_values()[3], 1);
    ASSERT_EQ(out->get_values()[0], 2);
}
