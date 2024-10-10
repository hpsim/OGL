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

int main(int argc, char *argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    my_argc = argc;
    my_argv = argv;

    result = RUN_ALL_TESTS();

    return result;
}
