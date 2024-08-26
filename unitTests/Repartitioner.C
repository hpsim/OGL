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

        auto comm = exec->get_gko_mpi_host_comm();
        if (comm->size() < 2) {
            std::cout << "At least 2 CPU processes should be used!"
                      << std::endl;
            std::abort();
        }
        // delete listener on ranks != 0
        // to clean up output
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        if (comm->rank() != 0) {
            delete listeners.Release(listeners.default_result_printer());
        }
    }

    std::shared_ptr<Foam::argList> args_;
    std::shared_ptr<const Foam::Time> time;
    std::shared_ptr<Foam::objectRegistry> db;
    std::shared_ptr<ExecutorHandler> exec;
};

const testing::Environment *global_env =
    AddGlobalTestEnvironment(new RepartitionerEnvironment);

class RepartitionerFixture : public testing::TestWithParam<int> {
public:
    label local_size = 10;
};

INSTANTIATE_TEST_SUITE_P(RepartitionerFixtureInstantiation,
                         RepartitionerFixture, testing::Values(1, 2, 4));

TEST_P(RepartitionerFixture, can_create_repartitioner)
{
    // Arrange
    label ranks_per_gpu = GetParam();
    auto exec = ((RepartitionerEnvironment *)global_env)->exec;

    // Act
    auto repartitioner = Repartitioner(10, ranks_per_gpu, 0, *exec.get());

    // Assert
    EXPECT_EQ(repartitioner.get_ranks_per_gpu(), ranks_per_gpu);
<<<<<<< HEAD
}


TEST_P(RepartitionerFixture, has_correct_properties_for_n_rank)
{
    // Arrange
    label ranks_per_gpu = GetParam();
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto rank = exec->get_rank();

    // Act
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);

    // Assert
    EXPECT_EQ(repartitioner.is_owner(*exec),
              (rank % ranks_per_gpu == 0) ? true : false);

    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, *exec),
        (repartitioner.is_owner(*exec) ? ranks_per_gpu * local_size : 0));
}

TEST(Repartitioner, can_convert_to_global)
{
    // Arrange
    auto exec = *((RepartitionerEnvironment *)global_env)->exec;
    auto rank = exec.get_rank();
    auto ref_exec = exec.get_ref_exec();
    auto comm = exec.get_communicator();
    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label,
                                                                        label>(
            ref_exec, *comm.get(), 4));

    //         local ids          |   global ids
    // cells: [ 2 3 | 2 3 ]       |  [ 10 11 | 14 15 ]
    //   2    [ 0 1 | 0 1 ]  3    |  [  8  9 | 12 13 ]
    //        -------------       |  -----------------
    //        [ 2 3 | 2 3 ]       |  [  2  3 |  6  7 ]
    //   0    [ 0 1 | 0 1 ]  1    |  [  0  1 |  4  5 ]
    //

    // vector
    std::vector<std::vector<label>> idxs{
        {0, 2, 0, 1},  // rank 0
        {1, 3, 0, 1},  // rank 1
        {2, 3, 0, 3},  // rank 2
        {2, 3, 1, 3}   // rank 3
    };
    // vector
    std::vector<std::vector<label>> ranks{
        {1, 2},  // rank 0
        {0, 3},  // rank 1
        {0, 3},  // rank 2
        {2, 1}   // rank 3
    };
    std::vector<std::vector<gko::span>> spans{
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 0
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 1
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 2
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 3
    };

    std::vector<std::vector<label>> exp_global_idxs{
        {4, 6, 8, 9},    // rank 0
        {1, 3, 12, 13},  // rank 1
        {2, 3, 12, 14},  // rank 2
        {9, 11, 6, 7}    // rank 3
    };

    // Act
    auto global_idxs = detail::convert_to_global(partition, idxs[rank],
                                                 spans[rank], ranks[rank]);

    // Assert
    EXPECT_EQ(global_idxs, exp_global_idxs[rank]);
}

// TEST_P(RepartitionerFixture, can_repartition_sparsity)
// {
//     using vec_vec = std::vector<std::vector<label>>;
//     // Arrange
//     label ranks_per_gpu = GetParam();
//     auto exec = *((RepartitionerEnvironment *)global_env)->exec.get();
//     auto rank = exec.get_rank();
//     auto comm = exec.get_communicator();
//     auto ref_exec = exec.get_ref_exec();
//     auto partition = gko::share(
//         gko::experimental::distributed::build_partition_from_local_size<label,
//                                                                         label>(
//             ref_exec, *comm.get(), 4));

//     //         local ids          |   global ids
//     // cells: [ 2 3 | 2 3 ]       |  [ 10 11 | 14 15 ]
//     //   2    [ 0 1 | 0 1 ]  3    |  [  8  9 | 12 13 ]
//     //        -------------       |  -----------------
//     //        [ 2 3 | 2 3 ]       |  [  2  3 |  6  7 ]
//     //   0    [ 0 1 | 0 1 ]  1    |  [  0  1 |  4  5 ]

//     std::vector<label> local_rows{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
//     std::vector<label> local_cols{0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
//     std::vector<label> local_mapping{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     std::vector<label> local_ranks{rank};
//     std::vector<gko::span> local_spans{gko::span{0, local_rows.size()}};

//     vec_vec non_local_rows{
//         {1, 3, 2, 3},
//         {0, 2, 2, 3},
//         {0, 1, 1, 3},
//         {0, 1, 0, 2},
//     };
//     vec_vec non_local_cols{
//         {4, 6, 8, 9},    // rank 0
//         {1, 3, 12, 13},  // rank 1
//         {2, 3, 12, 14},  // rank 2
//         {6, 7, 9, 11}    // rank 3
//     };
//     std::vector<label> non_local_mapping{0, 1, 2, 3};
//     vec_vec non_local_ranks{
//         {1, 2},  // rank 0
//         {0, 3},  // rank 1
//         {0, 3},  // rank 2
//         {1, 2}   // rank 3
//     };
//     vec_vec non_local_ranks{
//         {1, 2},  // rank 0
//         {0, 3},  // rank 1
//         {0, 3},  // rank 2
//         {1, 2}   // rank 3
//     };
//     std::vector<gko::span> non_local_spans{
//         {1, 2},  // rank 0
//         {0, 3},  // rank 1
//         {0, 3},  // rank 2
//         {1, 2}   // rank 3
//     };
//     std::vector<gko::span> non_local_spans{
//         {1, 2},  // rank 0
//         {0, 3},  // rank 1
//         {0, 3},  // rank 2
//         {1, 2}   // rank 3
//     };

//     auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, exec);

//     auto is_local = repartitioner.build_non_local_interfaces(
//         exec, partition, local_rows, local_cols, local_mapping, local_ranks,
//         local_spans, non_local_rows, non_local_cols, non_local_mapping,
//         non_local_ranks, non_local_origin, non_local_spans);


// }

TEST_P(RepartitionerFixture, can_exchange_spans_and_ranks_for_n_ranks)
{
    using vec_vec = std::vector<std::vector<label>>;
    // Arrange
    label ranks_per_gpu = GetParam();
    auto exec = *((RepartitionerEnvironment *)global_env)->exec.get();
    auto rank = exec.get_rank();
    auto ref_exec = exec.get_ref_exec();

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
    exp_spans_end.emplace(2, vec_vec{{5, 10, 15, 20}, {}, {5, 10, 15, 15}, {}});
    exp_spans_end.emplace(4,
                          vec_vec{{5, 10, 15, 20, 25, 30, 35, 40}, {}, {}, {}});

    // Act
    auto [new_spans, new_ranks] =
        detail::exchange_span_ranks(exec, ranks_per_gpu, spans, ranks);

    std::vector<label> res_spans_begin{};
    std::vector<label> res_spans_end{};

    for (auto span : new_spans) {
        res_spans_begin.push_back(span.begin);
        res_spans_end.push_back(span.end);
    }

    // Assert
    ASSERT_EQ(new_ranks, exp_ranks[ranks_per_gpu][rank]);
    ASSERT_EQ(res_spans_begin, exp_spans_begin[ranks_per_gpu][rank]);
    ASSERT_EQ(res_spans_end, exp_spans_end[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture, can_repartition_sparsity_pattern_1D_for_n_ranks)
{
    // Arrange
    label ranks_per_gpu = GetParam();
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = exec->get_rank();
    auto ref_exec = exec->get_ref_exec();

    // mesh:
    // rank:     0     1     2     3
    // cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
    // cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- local row ids

    std::vector<label> rows{0, 0, 1, 1};
    std::vector<label> cols{0, 1, 0, 1};
    std::vector<label> mapping{0, 1, 2, 3};
    std::vector<gko::span> spans{gko::span {0, 4}};
    std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{2, 2}, rows, cols, mapping, spans, ranks);

    std::vector<label> non_local_rows{1};
    std::vector<label> non_local_cols{0};
    std::vector<label> non_local_mapping{0};
    std::vector<gko::span> non_local_spans{gko::span {0, 1}};
    std::vector<label> non_local_ranks{rank};
    auto non_local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{2, 1}, non_local_rows, non_local_cols,
        non_local_mapping, non_local_spans, non_local_ranks);

    std::map<label, std::vector<label>> exp_local_nnz;
    exp_local_nnz.emplace(1, std::vector<label>{4, 4, 4, 4});
    exp_local_nnz.emplace(2, std::vector<label>{10, 0, 10, 0});
    exp_local_nnz.emplace(4, std::vector<label>{24, 0, 0, 0});

    std::map<label, std::vector<label>> exp_non_local_nnz;
    exp_non_local_nnz.emplace(1, std::vector<label>{1, 2, 2, 1});
    exp_non_local_nnz.emplace(2, std::vector<label>{1, 0, 1, 0});
    exp_non_local_nnz.emplace(4, std::vector<label>{0, 0, 0, 0});

    // Act
    auto [repart_local, repart_non_local, tracking] =
        repartitioner.repartition_sparsity(*exec, local_sparsity,
                                           non_local_sparsity);

    // Assert
    ASSERT_EQ(repart_local->num_nnz, exp_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->num_nnz, exp_non_local_nnz[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture, can_repartition_1D_comm_pattern_for_n_ranks)
{
    // Arrange
    label ranks_per_gpu = GetParam();
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = exec->get_rank();
    auto ref_exec = exec->get_ref_exec();

    // rank:     0     1     2     3
    // cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
    // cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- local row ids
    std::vector<std::vector<label>> ids{{1}, {0, 2}, {1, 3}, {2}};

    // expected communcation ranks
    using vec_vec = std::vector<std::vector<label>>;
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
    using vec_vec_vec = std::vector<vec_vec>;
    vec_vec_vec rows{{{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}};
    std::map<label, vec_vec> exp_res_rows;
    exp_res_rows.emplace(1, vec_vec{{1}, {0, 1}, {0, 1}, {0}});
    // after repartitioning with 2 ranks_per_gpu [ 0 1 2 3 | 0 1 2 3 ] <- local
    // row ids
    exp_res_rows.emplace(2, vec_vec{{3}, {}, {0}, {}});
    // after repartitioning with 4 ranks_per_gpu [ 0 1 2 3 4 5 6 7 ] <- local
    // row ids
    exp_res_rows.emplace(4, vec_vec{{}, {}, {}, {}});

    // the original comm_pattern
    auto comm_pattern =
        std::make_shared<CommunicationPattern>(*exec, ids[rank], rows[rank]);

    auto comm = exec->get_communicator();
    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label,
                                                                        label>(
            ref_exec, *comm.get(), 2));

    // Act
    auto repart_comm_pattern =
        repartitioner.repartition_comm_pattern(*exec, comm_pattern, partition);

    // Assert
    std::vector<label> res_ids(
        repart_comm_pattern->target_ids.get_const_data(),
        repart_comm_pattern->target_ids.get_const_data() +
            repart_comm_pattern->target_ids.get_size());

    EXPECT_EQ(res_ids, exp_res_ids[ranks_per_gpu][rank]);

    std::vector<label> res_sizes(
        repart_comm_pattern->target_sizes.get_const_data(),
        repart_comm_pattern->target_sizes.get_const_data() +
            repart_comm_pattern->target_sizes.get_size());
    EXPECT_EQ(res_sizes, exp_res_sizes[ranks_per_gpu][rank]);

    auto total_rank_send_idx = repart_comm_pattern->total_rank_send_idx();
    std::vector<label> res_rows(
        total_rank_send_idx.get_const_data(),
        total_rank_send_idx.get_const_data() + total_rank_send_idx.get_size());

    EXPECT_EQ(res_rows, exp_res_rows[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture, can_repartition_2D_comm_pattern_for_n_ranks)
{
    // Arrange
    label ranks_per_gpu = GetParam();
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = exec->get_rank();
    auto ref_exec = exec->get_ref_exec();

    // cells: [ 2 3 | 2 3 ]
    //   2    [ 0 1 | 0 1 ]  3
    //        -------------
    //        [ 2 3 | 2 3 ]
    //   0    [ 0 1 | 0 1 ]  1
    std::vector<std::vector<label>> ids{{1, 2}, {0, 3}, {0, 3}, {1, 2}};

    // expected communcation ranks
    using vec_vec = std::vector<std::vector<label>>;
    std::map<label, vec_vec> exp_res_ids{};

    exp_res_ids[1] = ids;  // in the ranks_per_gpu==1 case nothing changes
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
    using vec_vec_vec = std::vector<vec_vec>;
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
    // after repartitioning with 4 ranks_per_gpu [ 0 1 2 3 4 5 6 7 ] <- local
    // row ids
    exp_res_rows.emplace(4, vec_vec{{}, {}, {}, {}});

    // the original comm_pattern
    auto comm_pattern =
        std::make_shared<CommunicationPattern>(*exec, ids[rank], rows[rank]);

    auto comm = exec->get_communicator();
    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label,
                                                                        label>(
            ref_exec, *comm.get(), 4));

    // Act
    auto repart_comm_pattern =
        repartitioner.repartition_comm_pattern(*exec, comm_pattern, partition);

    // Assert
    std::vector<label> res_ids(
        repart_comm_pattern->target_ids.get_const_data(),
        repart_comm_pattern->target_ids.get_const_data() +
            repart_comm_pattern->target_ids.get_size());

    EXPECT_EQ(res_ids, exp_res_ids[ranks_per_gpu][rank]);

    std::vector<label> res_sizes(
        repart_comm_pattern->target_sizes.get_const_data(),
        repart_comm_pattern->target_sizes.get_const_data() +
            repart_comm_pattern->target_sizes.get_size());
    EXPECT_EQ(res_sizes, exp_res_sizes[ranks_per_gpu][rank]);

    auto total_rank_send_idx = repart_comm_pattern->total_rank_send_idx();
    std::vector<label> res_rows(
        total_rank_send_idx.get_const_data(),
        total_rank_send_idx.get_const_data() + total_rank_send_idx.get_size());

    EXPECT_EQ(res_rows, exp_res_rows[ranks_per_gpu][rank]);
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
