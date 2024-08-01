// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gtest/gtest.h>

#include "OGL/MatrixWrapper/Combination.H"

#include "fvCFD.H"

TEST(Combination, CanCreateEmptyCombination)
{
    auto cmb = Combination<scalar, label, gko::matrix::Coo>::create(this->exec);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmb->get_coefficients().size(), 0);
    ASSERT_EQ(cmb->get_operators().size(), 0);
}

TEST(Combination, CanCreateCombinationWithRanges)
{
    std::vector<gko::span> ranges {{0, 5}, {5, 6}};
    auto cmb = Combination<scalar, label, gko::matrix::Coo>::create(this->exec, gko::dim<2>{5,5}, ranges);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);
}
