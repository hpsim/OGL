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

TEST(Repartitioner, can_create_repartitioner)
{
    // Arrange
    label ranks_per_gpu = 1;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec;
    auto repartitioner = Repartitioner(10, ranks_per_gpu, 0, *exec.get());

    // Assert
    EXPECT_EQ(repartitioner.get_ranks_per_gpu(), ranks_per_gpu);
}

TEST(Repartitioner, has_correct_properties_for_1_rank)
{
    // Arrange
    label ranks_per_gpu = 1;
    label local_size = 10;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);

    // Assert
    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, *exec),
        local_size);

    EXPECT_EQ(
        repartitioner.is_owner(*exec),
        true);
}

TEST(Repartitioner, has_correct_properties_for_4_rank)
{
    // Arrange
    label ranks_per_gpu = 4;
    label local_size = 10;
    auto exec = ((RepartitionerEnvironment *)global_env)->exec.get();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, *exec);
    auto rank = repartitioner.get_rank(*exec);

    // Assert
    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, *exec),
        (repartitioner.is_owner(*exec) ? ranks_per_gpu * local_size : 0 ));

    EXPECT_EQ(
        repartitioner.is_owner(*exec),
        (rank == 0)? true: false);
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
