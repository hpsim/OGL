#include "gtest/gtest.h"
#include "HostMatrix/HostMatrix.H"

TEST(HelloWord, symmetric_update){
    /* test a 5x5 symmetric matrix
     * A =
     * | 1  1  .  2  .  |
     * | 1  1  1  .  2  |
     * | .  1  1  1  .  |
     * | 2  .  1  1  1  |
     * | .  2  .  1  1  |
     */

    std::vector<scalar> d { 1., 1., 1., 1., 1. }; 
    std::vector<scalar> u { 1., 2., 1., 2., 1. }; 
    std::vector<label>  p { 6, 0, 1, 7, 2, 3, 8, 4, 9};

    label total_nnz {15};
    label upper_nnz {5};

    std::vector<scalar> res {15, 0.0};
    std::vector<scalar> exp {1.,1.,2.,1.,1.,1.,2.,1.,1.,1.,2.,1.,1.,1.,2.,1.,1.};

    Foam::symmetric_update(total_nnz, upper_nnz, p.data(), 1.0, d.data(), u.data(), res.data());

    EXPECT_EQ(res, exp);
}
