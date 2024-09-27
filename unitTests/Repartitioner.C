// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Repartitioner.H"
#include "OGL/common.H"


#include "gtest/gtest.h"

#include "mpi.h"

#include <cstdlib>

//---------------------------------------------
// some_header.h
extern int my_argc;
extern char **my_argv;
// eof
//---------------------------------------------

//---------------------------------------------
// main.cpp
int my_argc;
char **my_argv;

using vec = std::vector<label>;
using vec_vec = std::vector<std::vector<label>>;
using vec_vec_vec = std::vector<vec_vec>;

class RepartitionerEnvironment : public testing::Environment {
public:
    void SetUp()
    {
        args_ = std::make_shared<Foam::argList>(my_argc, my_argv);
        time = std::make_shared<Foam::Time>("controlDict", *args_.get());
        db = std::make_shared<Foam::objectRegistry>(*time.get());
        Foam::dictionary dict;
        dict.add("executor", "reference");
        exec = std::make_shared<ExecutorHandler>(time->thisDb(), dict, "dummy",
                                                 true);

        if (exec->get_communicator()->size() < 2) {
            std::cout << "At least 2 CPU processes should be used!"
                      << std::endl;
            std::abort();
        }
        // delete listener on ranks != 0
        // to clean up output
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        // if (exec->get_communicator()->rank() != 0) {
        //     delete listeners.Release(listeners.default_result_printer());
        // }
    }

    std::shared_ptr<Foam::argList> args_;
    std::shared_ptr<const Foam::Time> time;
    std::shared_ptr<Foam::objectRegistry> db;
    std::shared_ptr<ExecutorHandler> exec;
};

const testing::Environment *global_env =
    AddGlobalTestEnvironment(new RepartitionerEnvironment);

class RepartitionerFixture : public testing::Test {
public:
    ExecutorHandler exec =
        *((RepartitionerEnvironment *)global_env)->exec.get();
    label rank = exec.get_rank();
    const gko::experimental::mpi::communicator comm =
        *(exec.get_communicator().get());
};

/* @brief Test fixture class for 1D mesh
 *
 * The mesh has the following structure
 * ranks:    0     1     2     3
 * cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
 * cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- local row ids
 */
class RepartitionerFixture1D
    : public RepartitionerFixture,
      public testing::WithParamInterface<std::tuple<bool, int>> {
public:
    label local_size = 2;  // matrix rows, same for all ranks

    // local data
    std::vector<label> rows{0, 0, 1, 1};
    std::vector<label> cols{0, 1, 0, 1};
    std::vector<label> mapping{0, 1, 2, 3};
    std::vector<gko::span> spans{gko::span{0, 5}};

    // Setup data
    // non local data
    vec_vec non_local_rows{{1}, {0, 1}, {0, 1}, {0}};
    // non local columns in global idxs
    vec_vec non_local_cols{{2}, {1, 4}, {3, 6}, {5}};
    vec_vec non_local_mapping{{0}, {0, 1}, {0, 1}, {0}};
    // communication partners (ranks)
    vec_vec ids{{1}, {0, 2}, {1, 3}, {2}};

    std::vector<std::vector<gko::span>> non_local_spans{
        {gko::span{0, 1}},                   // rank 0
        {gko::span{0, 1}, gko::span{1, 2}},  // rank 1
        {gko::span{0, 1}, gko::span{1, 2}},  // rank 2
        {gko::span{0, 1}},                   // rank 3
    };

    vec_vec non_local_ranks{
        {1},     // rank 0
        {0, 2},  // rank 1
        {1, 3},  // rank 2
        {2},     // rank 3
    };

    // Expected results
    // number of non-zeros of each sparsity pattern
    std::map<label, std::vector<label>> exp_local_nnz{
        {1, {4, 4, 4, 4}},
        {2, {10, 0, 10, 0}},
        {4, {22, 0, 0, 0}},
    };

    std::map<label, std::vector<label>> exp_non_local_nnz{
        {1, {1, 2, 2, 1}},
        {2, {1, 0, 1, 0}},
        {4, {0, 0, 0, 0}},
    };

    std::vector<std::vector<label>> comm_target_ids{{1}, {0, 2}, {1, 3}, {2}};
    // expected local row indices in local indices
    // first map fused true/false
    // local row indices
    // repartitioned 2 [ 0 1 , 2 3 | 0 1 , 2 3 ] | new  boundary , old boundary
    // repartitioned 4 [ 0 1 , 2 3 , 4 5 , 6 7 ] | new  boundary , old boundary
    // the last two elements are now local interfaces, they are in in order
    // of target_ids
    vec nf_rows_2 = {0, 0, 1, 1, 2, 2, 3, 3, 2, 1};
    vec nf_rows_4 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
                     5, 6, 6, 7, 7, 2, 1, 4, 3, 6, 5};
    // fused
    vec f_rows_2 = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    vec f_rows_4 = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                    4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7};
    std::map<bool, std::map<label, vec_vec>> exp_local_rows{
        {false,
         {{1, {rows, rows, rows, rows}},
          {2, {nf_rows_2, {}, nf_rows_2, {}}},
          {4, {nf_rows_4, {}, {}, {}}}}},
        {true,
         {{1, {rows, rows, rows, rows}},
          {2, {f_rows_2, {}, f_rows_2, {}}},
          {4, {f_rows_4, {}, {}, {}}}}}};

    vec nf_cols_2 = {0, 1, 0, 1, 2, 3, 2, 3, 1, 2};
    vec nf_cols_4 = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4,
                     5, 6, 7, 6, 7, 1, 2, 3, 4, 5, 6};
    // fused
    vec f_cols_2 = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    vec f_cols_4 = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4,
                    3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7};
    // first map fused true/false
    // expected local col indices in local indices
    std::map<bool, std::map<label, vec_vec>> exp_local_cols{
        {false,
         {{1, {cols, cols, cols, cols}},
          {2, {nf_cols_2, {}, nf_cols_2, {}}},
          {4, {nf_cols_4, {}, {}, {}}}}},
        {true,
         {{1, {cols, cols, cols, cols}},
          {2, {f_cols_2, {}, f_cols_2, {}}},
          {4, {f_cols_4, {}, {}, {}}}}}};

    vec nf_map_2{0, 1, 2, 3, 4, 5, 6, 7,  // here the interface values start
                 0, 1};                   // <- they are currently unused
    vec nf_map_4{0, 1, 2,  3,  4,  5,  6,  7,
                 8, 9, 10, 11, 12, 13, 14, 15,  // here the interface start
                 0, 1, 2,  3,  4,  5};          // <- unused values because we
                                                // dont permute atm
    vec f_map_2{0, 1, 2, 3, 8, 9, 4, 5, 6, 7};
    vec f_map_4{0,  1, 2, 3,  16, 17, 4,  5,  6,  7,  18,
                19, 8, 9, 10, 11, 20, 21, 12, 13, 14, 15};
    std::map<bool, std::map<label, vec_vec>> exp_local_mapping{
        {false,
         {{1, {mapping, mapping, mapping, mapping}},
          {2, {nf_map_2, {}, nf_map_2, {}}},
          {4, {nf_map_4, {}, {}, {}}}}},
        {true,
         {{1, {mapping, mapping, mapping, mapping}},
          {2, {f_map_2, {}, f_map_2, {}}},
          {4, {f_map_4, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_begin{
        {false,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0, 8, 9}, {}, {0, 8, 9}, {}}},
          {4, {{0, 16, 17, 18, 19, 20, 21}, {}, {}, {}}}}},
        {true,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{0}, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_end{
        {false,
         {{1, {{5}, {5}, {5}, {5}}},
          {2, {{8, 9, 10}, {}, {8, 9, 10}, {}}},
          {4, {{16, 17, 18, 19, 20, 21, 22}, {}, {}, {}}}}},
        {true,
         {{1, {{5}, {5}, {5}, {5}}},
          {2, {{10}, {}, {10}, {}}},
          {4, {{22}, {}, {}, {}}}}}};

    // number of rows of local matrix
    std::map<label, vec> exp_local_dim{
        {1, {local_size, local_size, local_size, local_size}},
        {2, {2 * local_size, 0, 2 * local_size, 0}},
        {4, {4 * local_size, 0, 0, 0}},
    };

    // non local data
    std::map<bool, std::map<label, vec_vec>> exp_non_local_rows{
        {false,
         {{1, non_local_rows}, {2, {{3}, {}, {0}, {}}}, {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_rows},
          {2, {{3}, {}, {0}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    // non local cols are in global indices
    std::map<bool, std::map<label, vec_vec>> exp_non_local_cols{
        {false,
         {{1, non_local_cols}, {2, {{4}, {}, {3}, {}}}, {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_cols},
          {2, {{4}, {}, {3}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_non_local_mapping{
        {false,
         {{1, {{0}, {0, 1}, {0, 1}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, {{0}, {0, 1}, {0, 1}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{}, {}, {}, {}}}}}};
};

/* @brief Test fixture class for 1D mesh
 *
 * The mesh has the following structure
 *         local ids          |   global ids
 * cells: [ 2 3 | 2 3 ]       |  [ 10 11 | 14 15 ]
 *   2    [ 0 1 | 0 1 ]  3    |  [  8  9 | 12 13 ]
 *        ------+------       |  --------+--------
 *        [ 2 3 | 2 3 ]       |  [  2  3 |  6  7 ]
 *   0    [ 0 1 | 0 1 ]  1    |  [  0  1 |  4  5 ]
 */
class RepartitionerFixture2D
    : public RepartitionerFixture,
      public testing::WithParamInterface<std::tuple<bool, int>> {
public:
    label local_size = 4;

    vec_vec idxs{
        {0, 2, 0, 1},  // rank 0
        {1, 3, 0, 1},  // rank 1
        {2, 3, 0, 2},  // rank 2
        {2, 3, 1, 3}   // rank 3
    };

    std::vector<label> rows{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    std::vector<label> cols{0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
    std::vector<label> mapping{8, 0, 1, 4, 9, 2, 5, 10, 3, 6, 7, 11};
    std::vector<gko::span> spans{gko::span{0, 13}};

    // Setup data
    // non local data
    vec_vec non_local_rows{
        {1, 3, 2, 3}, {0, 2, 2, 3}, {0, 1, 1, 3}, {0, 1, 0, 2}};
    // non local columns in global idxs
    vec_vec non_local_cols{
        {4, 6, 8, 9}, {1, 3, 12, 13}, {2, 3, 12, 14}, {6, 7, 9, 11}};
    vec_vec non_local_mapping{
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}};
    // communication partners (ranks)
    vec_vec comm_target_ids{{1, 2}, {0, 3}, {0, 3}, {1, 2}};

    std::vector<std::vector<gko::span>> non_local_spans{
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 0
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 1
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 2
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 3
    };

    vec_vec non_local_ranks{
        {1},     // rank 0
        {0, 2},  // rank 1
        {1, 3},  // rank 2
        {2},     // rank 3
    };

    // expected values
    std::map<label, vec> exp_local_nnz{
        {1, {12, 12, 12, 12}}, {2, {28, 0, 28, 0}}, {4, {64, 0, 0, 0}}};
    std::map<label, vec> exp_local_dim{
        {1, {4, 4, 4, 4}}, {2, {8, 0, 8, 0}}, {4, {16, 0, 0, 0}}};

    std::map<label, std::vector<label>> exp_non_local_nnz{
        {1, {4, 4, 4, 4}}, {2, {4, 0, 4, 0}}, {4, {0, 0, 0, 0}}};

    // expected local row indices in local indices
    // first map fused true/false
    // local row indices
    /*
     * The mesh has the following structure
     *         local ids          |   global ids
     * cells: [ 2 3 | 6 7 ]       |  [ 10 11 | 14 15 ]
     *   2    [ 0 1 | 4 5 ]  3    |  [  8  9 | 12 13 ]
     *        ------+------       |  --------+--------
     *        [ 2 3 | 6 7 ]       |  [  2  3 |  6  7 ]
     *   0    [ 0 1 | 4 5 ]  1    |  [  0  1 |  4  5 ]
     *   */
    vec nf_rows_2 = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
                     4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 4, 6, 1, 3};
    vec nf_cols_2 = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3, 4, 5,
                     6, 4, 5, 7, 4, 6, 7, 5, 6, 7, 1, 3, 4, 6};
    // [upper [0-3|12-15], lower [4,7|16-19], diag [8, 11|20-23], interfaces
    // [24-27]]
    vec nf_map_2 = {8,  0,  1,  4,  9,  2,
                    5,  10, 3,  6,  7,  11,  // here first sd is done
                    20, 12, 13, 16, 21, 14,
                    17, 22, 15, 18, 19, 23,  // here interfaces start
                    0,  1,  2,  3};
    vec f_rows_2 = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3,
                    4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7};
    vec f_cols_2 = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 1, 2, 3, 6,
                    1, 4, 5, 6, 4, 5, 7, 3, 4, 6, 7, 5, 6, 7};

    // rank 0                          | rank 1
    // [upper, lower, diag, interfaces | upper, lower, diag, interfaces]
    // vec f_map_2 = {8,  0,  1,  4,  9,  2,  24, 5,  10, 3,  6,  7,  11, 24,
    //                26, 20, 12, 13, 16, 21, 14, 26, 17, 22, 15, 18, 19, 23};
    //                the one below is broken
    vec f_map_2 = {8,  0,  1,  4,  9,  2,  24, 5,  10, 3,  6,  7,  11, 24,
                   26, 20, 12, 13, 16, 21, 14, 26, 17, 22, 15, 18, 19, 23};
    /*
     * The mesh has the following structure
     *         local ids          |   global ids
     * cells: [10 11|14 15]       |  [ 10 11 | 14 15 ]
     *   2    [ 8 9 |12 13]  3    |  [  8  9 | 12 13 ]
     *        ------+------       |  --------+--------
     *        [ 2 3 | 6 7 ]       |  [  2  3 |  6  7 ]
     *   0    [ 0 1 | 4 5 ]  1    |  [  0  1 |  4  5 ]
     *   */
    // NOTE the interfaces are in order of the comm_target indices
    // ie [0, 0, 0 (1), (1), 2 (2), 2 (3), 2 (2), 2 (3)]
    vec nf_rows_4 = {0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,
                     4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  8,  8,
                     8,  9,  9,  9,  10, 10, 10, 11, 11, 11, 12, 12, 12,
                     13, 13, 13, 14, 14, 14, 15, 15, 15, 4,  6,  8,  9,
                     1,  3,  12, 13, 2,  3,  12,  14,  6,  7,  9,  11};
    vec nf_cols_4 = {0,  1,  2,  0,  1,  3,  0,  2,  3,  1,  2,  3,  4,
                     5,  6,  4,  5,  7,  4,  6,  7,  5,  6,  7,  8,  9,
                     10, 8,  9,  11, 8,  10, 11, 9,  10, 11, 12, 13, 14,
                     12, 13, 15, 12, 14, 15, 13, 14, 15, 1,  3,  2,  3,
                     4,  6,  6,  7,  8,  9,  9,  11, 12, 13, 12, 14};
    vec nf_map_4{8,  0,  1,  4,  9,  2,  5,  10, 3,  6,  7,  11,  // 1st sd done
                 20, 12, 13, 16, 21, 14, 17, 22, 15, 18, 19, 23,  // 2nd sd done
                 32, 24, 25, 28, 33, 26, 29, 34, 27, 30, 31, 35,  // 3rd sd done
                 44, 36, 37, 40, 45, 38, 41, 46, 39, 42, 43, 47,  // 4th sd done
                 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                 12, 13, 14, 15};
    // fused
    vec f_rows_4 = {0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,
                    3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  6,  6,  6,
                    6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,
                    9,  9,  10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12,
                    12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15};
    vec f_cols_4 = {0,  1,  2,  0,  1,  3,  4,  0,  2,  3,  8,  1,  2,
                    3,  6,  9,  1,  4,  5,  6,  4,  5,  7,  3,  4,  6,
                    7,  12, 5,  6,  7,  13, 2,  8,  9,  10, 3,  8,  9,
                    11, 12, 8,  10, 11, 9,  10, 11, 14, 6,  9,  12, 13,
                    14, 7,  12, 13, 15, 11, 12, 14, 15, 13, 14, 15};
    vec f_map_4{0,  1, 2, 3,  16, 17, 4,  5,  6,  7,  18,
                19, 8, 9, 10, 11, 20, 21, 12, 13, 14, 15};

    std::map<bool, std::map<label, vec_vec>> exp_local_rows{
        {false,
         {{1, {rows, rows, rows, rows}},
          {2, {nf_rows_2, {}, nf_rows_2, {}}},
          {4, {nf_rows_4, {}, {}, {}}}}},
        {true,
         {{1, {rows, rows, rows, rows}},
          {2, {f_rows_2, {}, f_rows_2, {}}},
          {4, {f_rows_4, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_local_cols{
        {false,
         {{1, {cols, cols, cols, cols}},
          {2, {nf_cols_2, {}, nf_cols_2, {}}},
          {4, {nf_cols_4, {}, {}, {}}}}},
        {true,
         {{1, {cols, cols, cols, cols}},
          {2, {f_cols_2, {}, f_cols_2, {}}},
          {4, {f_cols_4, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_local_mapping{
        {false,
         {{1, {mapping, mapping, mapping, mapping}},
          {2, {nf_map_2, {}, nf_map_2, {}}},
          {4, {nf_map_4, {}, {}, {}}}}},
        {true,
         {{1, {mapping, mapping, mapping, mapping}},
          {2, {f_map_2, {}, f_map_2, {}}},
          {4, {f_map_4, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_begin{
        {false,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0, 24, 26}, {}, {0, 24, 26}, {}}},
          {4, {{0, 48, 50, 52, 54, 56, 58, 60, 62}, {}, {}, {}}}}},
        {true,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{0}, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_end{
        {false,
         {{1, {{13}, {13}, {13}, {13}}},
          {2, {{24, 26, 28}, {}, {24, 26, 28}, {}}},
          {4, {{48, 50, 52, 54, 56, 58, 60, 62, 64}, {}, {}, {}}}}},
        {true,
         {{1, {{13}, {13}, {13}, {13}}},
          {2, {{10}, {}, {10}, {}}},
          {4, {{22}, {}, {}, {}}}}}};

    // non local data
    std::map<bool, std::map<label, vec_vec>> exp_non_local_rows{
        {false,
         {{1, non_local_rows},
          {2, {{2, 3, 6, 7}, {}, {0, 1, 4, 5}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_rows},
          {2, {{}, {}, {}, {}}},  // missing
          {4, {{}, {}, {}, {}}}}}};

    // non local cols are in global indices
    std::map<bool, std::map<label, vec_vec>> exp_non_local_cols{
        {false,
         {{1, non_local_cols},
          {2, {{8, 9, 12, 13}, {}, {2, 3, 6, 7}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_cols}, {2, {{}, {}, {}, {}}}, {4, {{}, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_non_local_mapping{
        {false,
         {{1, {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}},
          {2, {{0, 1, 2, 3}, {}, {0, 1, 2, 3}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_cols}, {2, {{}, {}, {}, {}}}, {4, {{}, {}, {}, {}}}}}};
};

INSTANTIATE_TEST_SUITE_P(RepartitionerFixture1DInstantiation,
                         RepartitionerFixture1D,
                         testing::Combine(testing::Values(false),
                                          testing::Values(1, 2, 4)));
INSTANTIATE_TEST_SUITE_P(
    RepartitionerFixture2DInstantiation, RepartitionerFixture2D,
    testing::Combine(testing::Values(false), testing::Values(1, 2, 4))  //,
    // [](const testing::TestParamInfo<MyTestSuite::ParamType> &info) {
    //     // Can use info.param here to generate the test suffix
    //     std::string name = "foo";
    //     return name;
    // }
);

TEST_P(RepartitionerFixture1D, can_create_repartitioner)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    // Act
    auto repartitioner = Repartitioner(10, ranks_per_gpu, 0, exec, fused);
    // Assert
    EXPECT_EQ(repartitioner.get_ranks_per_gpu(), ranks_per_gpu);
}

TEST_P(RepartitionerFixture1D, has_correct_properties_for_n_rank)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    // Act
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);

    // Assert
    EXPECT_EQ(repartitioner.is_owner(exec),
              (rank % ranks_per_gpu == 0) ? true : false);

    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, exec),
        (repartitioner.is_owner(exec) ? ranks_per_gpu * local_size : 0));
}

TEST_P(RepartitionerFixture2D, can_convert_to_global)
{
    // Arrange
    using namespace gko::experimental::distributed;
    auto ref_exec = exec.get_ref_exec();
    auto partition = gko::share(
        build_partition_from_local_size<label, label>(ref_exec, comm, 4));

    vec_vec exp_global_idxs{
        {4, 6, 8, 9},    // rank 0
        {1, 3, 12, 13},  // rank 1
        {2, 3, 12, 14},  // rank 2
        {6, 7, 9, 11}    // rank 3
    };

    // Act
    auto global_idxs = detail::convert_to_global(
        partition, idxs[rank].data(), non_local_spans[rank], comm_target_ids[rank]);

    // Assert
    EXPECT_EQ(global_idxs, exp_global_idxs[rank]);
}


TEST_P(RepartitionerFixture1D, can_exchange_spans_and_ranks_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();

    std::vector<gko::span> spans{{0, 5}, {5, 10}};
    std::vector<label> ranks{rank + 1, rank + 2};

    std::map<label, vec_vec> exp_ranks;
    exp_ranks.emplace(1, vec_vec{{1, 2}, {2, 3}, {3, 4}, {4, 5}});
    exp_ranks.emplace(2, vec_vec{{1, 2, 2, 3}, {}, {3, 4, 4, 5}, {}});
    exp_ranks.emplace(4, vec_vec{{1, 2, 2, 3, 3, 4, 4, 5}, {}, {}, {}});

    std::map<label, vec_vec> exp_spans_begin;
    exp_spans_begin.emplace(1, vec_vec{{0, 5}, {0, 5}, {0, 5}, {0, 5}});
    exp_spans_begin.emplace(2, vec_vec{{0, 5, 10, 15}, {}, {0, 5, 10, 15}, {}});
    exp_spans_begin.emplace(
        4, vec_vec{{0, 5, 10, 15, 20, 25, 30, 35}, {}, {}, {}});
    std::map<label, vec_vec> exp_spans_end;
    exp_spans_end.emplace(1, vec_vec{{5, 10}, {5, 10}, {5, 10}, {5, 10}});
    exp_spans_end.emplace(2, vec_vec{{5, 10, 15, 20}, {}, {5, 10, 15, 20}, {}});
    exp_spans_end.emplace(4,
                          vec_vec{{5, 10, 15, 20, 25, 30, 35, 40}, {}, {}, {}});

    // Act
    auto [new_spans, origins, gathered_ranks] =
        detail::exchange_spans_ranks(exec, ranks_per_gpu, spans, ranks);

    std::vector<label> res_spans_begin{};
    std::vector<label> res_spans_end{};

    for (auto span : new_spans) {
        res_spans_begin.push_back(span.begin);
        res_spans_end.push_back(span.end);
    }

    // Assert
    ASSERT_EQ(res_spans_begin, exp_spans_begin[ranks_per_gpu][rank]);
    ASSERT_EQ(res_spans_end, exp_spans_end[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture1D, can_repartition_sparsity_pattern_1D_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{2, 2}, rows, cols, mapping, spans);

    auto non_local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{2, non_local_ranks[rank].size()},
        non_local_rows[rank], non_local_cols[rank], non_local_mapping[rank],
        non_local_spans[rank]);

    // Act
    auto [repart_local, repart_non_local, tracking] =
        repartitioner.repartition_sparsity(exec, local_sparsity,
                                           non_local_sparsity,
                                           comm_target_ids[rank], fused);
    // Assert
    // local properties
    ASSERT_EQ(repart_local->num_nnz, exp_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->dim[0], exp_local_dim[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->dim[1], exp_local_dim[ranks_per_gpu][rank]);

    auto res_local_rows = convert_to_vector(repart_local->row_idxs);
    auto res_local_cols = convert_to_vector(repart_local->col_idxs);
    auto res_local_mapping = convert_to_vector(repart_local->ldu_mapping);
    ASSERT_EQ(res_local_rows, exp_local_rows[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_cols, exp_local_cols[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_mapping, exp_local_mapping[fused][ranks_per_gpu][rank]);

    std::vector<label> res_local_spans_begin{};
    std::vector<label> res_local_spans_end{};
    for (auto [begin, end] : repart_local->spans) {
        res_local_spans_begin.push_back(begin);
        res_local_spans_end.push_back(end);
    }
    ASSERT_EQ(res_local_spans_begin,
              exp_local_spans_begin[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_spans_end,
              exp_local_spans_end[fused][ranks_per_gpu][rank]);

    // non local properties
    ASSERT_EQ(repart_non_local->num_nnz,
              exp_non_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->dim[0], exp_local_dim[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->dim[1], exp_non_local_nnz[ranks_per_gpu][rank]);

    auto res_non_local_rows = convert_to_vector(repart_non_local->row_idxs);
    auto res_non_local_cols = convert_to_vector(repart_non_local->col_idxs);
    ASSERT_EQ(res_non_local_rows,
              exp_non_local_rows[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_cols,
              exp_non_local_cols[fused][ranks_per_gpu][rank]);
    auto res_non_local_mapping =
        convert_to_vector(repart_non_local->ldu_mapping);
    ASSERT_EQ(res_non_local_mapping,
              exp_non_local_mapping[fused][ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture2D, can_repartition_sparsity_pattern_2D_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{4, 4}, rows, cols, mapping, spans);

    auto non_local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{4, 4}, non_local_rows[rank], non_local_cols[rank],
        non_local_mapping[rank], non_local_spans[rank]);

    // Act
    auto [repart_local, repart_non_local, tracking] =
        repartitioner.repartition_sparsity(exec, local_sparsity,
                                           non_local_sparsity,
                                           comm_target_ids[rank], fused);
    // Assert
    // local properties
    ASSERT_EQ(repart_local->num_nnz, exp_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->dim[0], exp_local_dim[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->dim[1], exp_local_dim[ranks_per_gpu][rank]);

    auto res_local_rows = convert_to_vector(repart_local->row_idxs);
    auto res_local_cols = convert_to_vector(repart_local->col_idxs);
    auto res_local_mapping = convert_to_vector(repart_local->ldu_mapping);
    ASSERT_EQ(res_local_rows.size(),
              exp_local_rows[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_rows[i],
                  exp_local_rows[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i;
    }
    ASSERT_EQ(res_local_cols.size(),
              exp_local_cols[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_cols.size(); i++) {
        ASSERT_EQ(res_local_cols[i],
                  exp_local_cols[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i;
    }
    ASSERT_EQ(res_local_mapping.size(),
              exp_local_cols[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_cols.size(); i++) {
        ASSERT_EQ(res_local_cols[i],
                  exp_local_cols[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i;
    }

    std::vector<label> res_local_spans_begin{};
    std::vector<label> res_local_spans_end{};
    for (auto [begin, end] : repart_local->spans) {
        res_local_spans_begin.push_back(begin);
        res_local_spans_end.push_back(end);
    }
    ASSERT_EQ(res_local_spans_begin,
              exp_local_spans_begin[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_spans_end,
              exp_local_spans_end[fused][ranks_per_gpu][rank]);

    // non local properties
    ASSERT_EQ(repart_non_local->num_nnz,
              exp_non_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->dim[0], exp_local_dim[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->dim[1], exp_non_local_nnz[ranks_per_gpu][rank]);

    auto res_non_local_rows = convert_to_vector(repart_non_local->row_idxs);
    auto res_non_local_cols = convert_to_vector(repart_non_local->col_idxs);
    ASSERT_EQ(res_non_local_rows,
              exp_non_local_rows[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_cols,
              exp_non_local_cols[fused][ranks_per_gpu][rank]);
    auto res_non_local_mapping =
        convert_to_vector(repart_non_local->ldu_mapping);
    ASSERT_EQ(res_non_local_mapping,
              exp_non_local_mapping[fused][ranks_per_gpu][rank]);
}


TEST_P(RepartitionerFixture1D, can_repartition_1D_comm_pattern_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    // expected communication ranks
    std::map<label, vec_vec> exp_res_ids{};

    exp_res_ids[1] = ids;  // in the ranks_per_gpu==1 case nothing changes
    // only communication partners are 0-2 and 2-0
    exp_res_ids.emplace(2, vec_vec{{2}, {}, {0}, {}});
    // no communication if all ranks are repartitioned to single owner
    exp_res_ids.emplace(4, vec_vec{{}, {}, {}, {}});

    // expected communication sizes
    std::map<label, vec_vec> exp_res_sizes;
    exp_res_sizes.emplace(1, vec_vec({{1}, {1, 1}, {1, 1}, {1}}));
    exp_res_sizes.emplace(2, vec_vec({{1}, {}, {1}, {}}));
    exp_res_sizes.emplace(4, vec_vec({{}, {}, {}, {}}));

    // expected rows
    vec_vec_vec rows{{{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}};
    std::map<label, vec_vec> exp_res_rows;
    exp_res_rows.emplace(1, vec_vec{{1}, {0, 1}, {0, 1}, {0}});
    // after repartitioning with 2 ranks_per_gpu [ 0 1 2 3 | 0 1 2 3 ] <-
    // local row ids
    exp_res_rows.emplace(2, vec_vec{{3}, {}, {0}, {}});
    // after repartitioning with 4 ranks_per_gpu [ 0 1 2 3 4 5 6 7 ] <-
    // local row ids
    exp_res_rows.emplace(4, vec_vec{{}, {}, {}, {}});

    // the original comm_pattern
    auto comm_pattern =
        std::make_shared<CommunicationPattern>(exec, ids[rank], rows[rank]);

    // Act
    auto repart_comm_pattern =
        repartitioner.repartition_comm_pattern(exec, comm_pattern);

    // Assert
    auto res_ids = repart_comm_pattern->target_ids;
    EXPECT_EQ(res_ids, exp_res_ids[ranks_per_gpu][rank]);

    auto res_sizes = repart_comm_pattern->target_sizes;
    EXPECT_EQ(res_sizes, exp_res_sizes[ranks_per_gpu][rank]);

    auto total_rank_send_idx = repart_comm_pattern->total_rank_send_idx();
    auto res_rows = total_rank_send_idx;
    EXPECT_EQ(res_rows, exp_res_rows[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture2D, can_repartition_2D_comm_pattern_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    // expected communication ranks
    std::map<label, vec_vec> exp_res_ids{};
    exp_res_ids[1] = comm_target_ids;  // in the ranks_per_gpu==1 case nothing changes
    // only communication partners are 0-2 and 2-0
    exp_res_ids.emplace(2, vec_vec{{2}, {}, {0}, {}});
    // no communication if all ranks are repartitioned to single owner
    exp_res_ids.emplace(4, vec_vec{{}, {}, {}, {}});

    // expected communication sizes
    std::map<label, vec_vec> exp_res_sizes;
    exp_res_sizes.emplace(1, vec_vec({{2, 2}, {2, 2}, {2, 2}, {2, 2}}));
    exp_res_sizes.emplace(2, vec_vec({{4}, {}, {4}, {}}));
    exp_res_sizes.emplace(4, vec_vec({{}, {}, {}, {}}));

    // expected rows
    vec_vec_vec rows{
        {{1, 3}, {2, 3}}, {{0, 2}, {2, 3}}, {{0, 1}, {1, 3}}, {{0, 1}, {0, 2}}};

    std::map<label, vec_vec> exp_res_rows;
    exp_res_rows.emplace(
        1, vec_vec{{1, 3, 2, 3}, {0, 2, 2, 3}, {0, 1, 1, 3}, {0, 1, 0, 2}});
    // after repartitioning with 2 ranks_per_gpu
    // cells: [ 2 3  6 7 ]
    //   2    [ 0 1  4 5 ]  3
    //        -------------
    //        [ 2 3  6 7 ]
    //   0    [ 0 1  4 5 ]  1
    exp_res_rows.emplace(2, vec_vec{{2, 3, 6, 7}, {}, {0, 1, 4, 5}, {}});
    // after repartitioning with 4 ranks_per_gpu [ 0 1 2 3 4 5 6 7 ] <-
    // local row ids
    exp_res_rows.emplace(4, vec_vec{{}, {}, {}, {}});

    std::map<label, vec_vec> exp_gather_idx;
    exp_gather_idx.emplace(
        1, vec_vec{{0, 2, 0, 1}, {1, 3, 0, 1}, {2, 3, 0, 2}, {2, 3, 1, 3}});
    exp_gather_idx.emplace(2, vec_vec{{0, 1, 4, 5}, {}, {2, 3, 6, 7}, {}});
    exp_gather_idx.emplace(4, vec_vec{{}, {}, {}, {}});

    // the original comm_pattern
    auto comm_pattern =
        std::make_shared<CommunicationPattern>(exec, comm_target_ids[rank], rows[rank]);

    // Act
    auto repart_comm_pattern =
        repartitioner.repartition_comm_pattern(exec, comm_pattern);

    // Assert
    auto res_ids = repart_comm_pattern->target_ids;

    EXPECT_EQ(res_ids, exp_res_ids[ranks_per_gpu][rank]);

    auto res_sizes = repart_comm_pattern->target_sizes;
    EXPECT_EQ(res_sizes, exp_res_sizes[ranks_per_gpu][rank]);

    auto total_rank_send_idx = repart_comm_pattern->total_rank_send_idx();
    auto res_rows = total_rank_send_idx;
    EXPECT_EQ(res_rows, exp_res_rows[ranks_per_gpu][rank]);

    auto total_recv_gather_idx =
        repart_comm_pattern->compute_recv_gather_idxs(exec);
    auto res_gather_idx = convert_to_vector(total_recv_gather_idx);
    EXPECT_EQ(res_gather_idx, exp_gather_idx[ranks_per_gpu][rank]);
}


int main(int argc, char *argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    my_argc = argc;
    my_argv = argv;

    result = RUN_ALL_TESTS();

    return result;
}
