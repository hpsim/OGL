// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include "fvCFD.H"

#include <gtest/gtest.h>

#include "OGL/MatrixWrapper/Combination.H"



TEST(Combination, CanCreateCombination)
{
    auto cmb = Combination<double, int, gko::matrix::Coo>::create(this->exec);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmb->get_coefficients().size(), 0);
    ASSERT_EQ(cmb->get_operators().size(), 0);
}
