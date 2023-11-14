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
    std::vector<label> p{12, 0, 1, 
        6, 13, 2, 3,
        7, 14, 4, 
        8, 9, 15, 5, 
        10, 11, 16};

    label total_nnz{17};
    label upper_nnz{6};

    std::vector<scalar> res{0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<scalar> exp{
        1., 1., 2.,
        2., 1., 1., 2.,
        2., 1., 1.,
        3., 2., 1., 1.,
        3., 2., 1.};

    Foam::non_symmetric_update(total_nnz, upper_nnz, p.data(), 1.0, d.data(),
                           u.data(), l.data(), res.data());

    EXPECT_EQ(res, exp);
}
