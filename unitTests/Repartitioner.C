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
    // cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- global row ids
    std::vector<std::vector<label>> ids{{1}, {0, 2}, {1, 3}, {2}};

    // expected communcation ranks
    using vec_vec =std::vector<std::vector<label>>;
    std::map<label, vec_vec> exp_res_ids {};

    exp_res_ids[1] = ids; // in the ranks_per_gpu==1 case nothing changes
    // only communication partners are 0-2 and 2-0
    exp_res_ids.emplace(2, vec_vec{{2}, {}, {0}, {}});
    // no communication if all ranks are repartitioned to single owner
    exp_res_ids.emplace(4, vec_vec{{}, {}, {}, {}});

    // expected communication sizes
    std::map<label, std::vector<std::vector<label>>> exp_res_sizes;
    exp_res_sizes.emplace(1, vec_vec({{1}, {1, 1}, {1, 1}, {1}}));
    exp_res_sizes.emplace(2, vec_vec({{1}, {}, {1}, {}}));
    exp_res_sizes.emplace(4, vec_vec({{}, {}, {}, {}}));

    // expected rows
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

    EXPECT_EQ(res_ids, exp_res_ids[ranks_per_gpu][rank]);

   std::vector<label> res_sizes(
       repart_comm_pattern->target_sizes.get_const_data(),
       repart_comm_pattern->target_sizes.get_const_data() +
           repart_comm_pattern->target_sizes.get_size());
   EXPECT_EQ(res_sizes, exp_res_sizes[ranks_per_gpu][rank]);
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
