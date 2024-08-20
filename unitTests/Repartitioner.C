// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Repartitioner.H"

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

class RepartitionerFixture : public testing::TestWithParam<int> {};

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
}

TEST_P(RepartitionerFixture, has_correct_properties_for_n_rank)
{
    // Arrange
    label ranks_per_gpu = GetParam();
    label local_size = 10;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto rank = exec->get_rank();

    // Act
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);

    // Assert
    EXPECT_EQ(repartitioner.is_owner(*exec), (rank % ranks_per_gpu == 0) ? true : false);

    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, *exec),
        (repartitioner.is_owner(*exec) ? ranks_per_gpu * local_size : 0));

}


TEST(Repartitioner, can_repartition_comm_pattern_for_1_rank)
{
    // Arrange
    label ranks_per_gpu = 1;
    label local_size = 10;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = exec->get_rank();
    auto ref_exec = exec->get_ref_exec();

    // rank:     0     1     2     3
    // cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
    // cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- global row ids
    std::vector<std::vector<label>> ids{{1}, {0, 2}, {1, 3}, {2}};
    std::vector<std::vector<std::vector<label>>> rows{
        {{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}};

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
    // communication ranks are the same as before
    std::vector<label> res_ids(
        repart_comm_pattern->target_ids.get_const_data(),
        repart_comm_pattern->target_ids.get_const_data() +
            repart_comm_pattern->target_ids.get_size());
    EXPECT_EQ(res_ids, ids[rank]);

    std::vector<label> res_sizes(
        repart_comm_pattern->target_sizes.get_const_data(),
        repart_comm_pattern->target_sizes.get_const_data() +
            repart_comm_pattern->target_sizes.get_size());
    std::vector<std::vector<label>> exp_sizes{{1}, {1, 1}, {1, 1}, {1}};
    EXPECT_EQ(res_sizes, exp_sizes[rank]);
}

TEST(Repartitioner, can_repartition_comm_pattern_for_2_rank)
{
    // Arrange
    label ranks_per_gpu = 2;
    label local_size = 10;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = exec->get_rank();
    auto ref_exec = exec->get_ref_exec();

    // rank:     0     1     2     3
    // cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
    // cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- local row ids
    std::vector<std::vector<label>> ids{{1}, {0, 2}, {1, 3}, {2}};
    std::vector<std::vector<std::vector<label>>> rows{
        {{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}};

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
    // on non owner no communication ranks should be left
    std::vector<std::vector<label>> exp_ids{{2}, {}, {0}, {}};
    EXPECT_EQ(res_ids, exp_ids[rank]);

    std::vector<label> res_sizes(
        repart_comm_pattern->target_sizes.get_const_data(),
        repart_comm_pattern->target_sizes.get_const_data() +
            repart_comm_pattern->target_sizes.get_size());
    std::vector<std::vector<label>> exp_sizes{{1}, {}, {1}, {}};
    // thus communication sizes is also empty
    EXPECT_EQ(res_sizes, exp_sizes[rank]);
}


TEST(Repartitioner, can_repartition_comm_pattern_for_4_rank)
{
    // Arrange
    label ranks_per_gpu = 4;
    label local_size = 10;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = exec->get_rank();
    auto ref_exec = exec->get_ref_exec();

    // rank:     0     1     2     3
    // cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
    // cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- global row ids
    std::vector<std::vector<label>> ids{{1}, {0, 2}, {1, 3}, {2}};
    std::vector<std::vector<std::vector<label>>> rows{
        {{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}};

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
    // no communication ranks should be left
    EXPECT_EQ(res_ids.size(), 0);

    std::vector<label> res_sizes(
        repart_comm_pattern->target_sizes.get_const_data(),
        repart_comm_pattern->target_sizes.get_const_data() +
            repart_comm_pattern->target_sizes.get_size());
    // thus communication sizes is also empty
    EXPECT_EQ(res_sizes.size(), 0);
    EXPECT_EQ(repart_comm_pattern->send_idxs.size(), 0);
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
