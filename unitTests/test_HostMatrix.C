#include "HostMatrix/HostMatrix.H"
#include "gtest/gtest.h"

TEST(HostMatrixConversion, symmetric_update)
{
    /* test a 5x5 symmetric matrix
     * A =
     * | 1  1  .  2  .  |
     * | 1  1  1  .  2  |
     * | .  1  1  1  .  |
     * | 2  .  1  1  1  |
     * | .  2  .  1  1  |
     */

    std::vector<scalar> d{1., 1., 1., 1., 1.};
    std::vector<scalar> u{1., 2., 1., 2., 1., 1.};
    std::vector<label> p{6, 0, 1, 0, 7, 2, 3, 2, 8, 4, 1, 4, 9, 5, 3, 5, 10};

    label total_nnz{17};
    label upper_nnz{6};

    std::vector<scalar> res{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<scalar> exp{1., 1., 2., 1., 1., 1., 2., 1., 1.,
                            1., 2., 1., 1., 1., 2., 1., 1.};

    Foam::symmetric_update(total_nnz, upper_nnz, p.data(), 1.0, d.data(),
                           u.data(), res.data());

    EXPECT_EQ(res, exp);
}

TEST(HostMatrixConversion, non_symmetric_update)
{
    /* test a 5x5 symmetric matrix
     * A =
     * | 1  1  .  2  .  |
     * | 2  1  1  .  2  |
     * | .  2  1  1  .  |
     * | 3  .  2  1  1  |
     * | .  3  .  2  1  |
     */

    std::vector<scalar> d{1., 1., 1., 1., 1.};
    std::vector<scalar> u{1., 2., 1., 2., 1., 1.};
    std::vector<scalar> l{2., 2., 3., 2., 3., 2.};
    std::vector<label> p{12, 0, 1, 6,  13, 2,  3,  7, 14,
                         4,  8, 9, 15, 5,  10, 11, 16};

    label total_nnz{17};
    label upper_nnz{6};

    std::vector<scalar> res{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<scalar> exp{1., 1., 2., 2., 1., 1., 2., 2., 1.,
                            1., 3., 2., 1., 1., 3., 2., 1.};

    Foam::non_symmetric_update(total_nnz, upper_nnz, p.data(), 1.0, d.data(),
                               u.data(), l.data(), res.data());

    EXPECT_EQ(res, exp);
}

TEST(HostMatrixConversion, init_local_sparsisty)
{
    /* test a 5x5 symmetric matrix
     * A =
     *     0  1  2  3  4
     * 0 | x  x  .  x  .  |
     * 1 | x  x  x  .  x  |
     * 2 | .  x  x  x  .  |
     * 3 | x  .  x  x  x  |
     * 4 | .  x  .  x  x  |
     */

    const label nrows = 5;
    const label upper_nnz = 6;
    const bool is_symmetric = true;

    std::vector<label> upper{1, 3, 2, 4, 3, 4};
    std::vector<label> lower{0, 0, 1, 1, 2, 3};

    std::vector<label> rows{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<label> cols{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<label> permute{0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0};

    std::vector<label> rows_exp{0, 0, 0, 1, 1, 1, 1, 2, 2,
                                2, 3, 3, 3, 3, 4, 4, 4};
    std::vector<label> cols_exp{0, 1, 3, 0, 1, 2, 4, 1, 2,
                                3, 0, 2, 3, 4, 1, 3, 4};
    std::vector<label> permute_exp{6, 0, 1, 0, 7, 2, 3, 2, 8,
                                   4, 1, 4, 9, 5, 3, 5, 10};
    Foam::init_local_sparsity(nrows, upper_nnz, is_symmetric, upper.data(),
                              lower.data(), rows.data(), cols.data(),
                              permute.data());

    EXPECT_EQ(rows, rows_exp);
    EXPECT_EQ(cols, cols_exp);
    EXPECT_EQ(permute, permute_exp);
}
