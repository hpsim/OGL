// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include "OGL/MatrixWrapper/HostMatrix.H"
#include "OGL/MatrixWrapper/Distributed.H"
#include "OGL/Repartitioner.H"

#include "gtest/gtest.h"

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

class Environment : public testing::Environment {
public:
    void SetUp()
    {
        dict.add("executor", "reference");

        args_ = std::make_shared<Foam::argList>(my_argc, my_argv);
        if (args_->size() != 1) {
            std::cout << "Wrong number of arguments detected: " << args_->size()
                      << ", make sure to run "
                         "with -parallel"
                      << std::endl;
            std::abort();
        }

        runTime_ = std::make_shared<Foam::Time>("controlDict", *args_.get());

        mesh = std::make_shared<Foam::fvMesh>(
            Foam::IOobject(word{""}, runTime_->timeName(), *runTime_.get(),
                           Foam::IOobject::MUST_READ),
            false);

        exec = std::make_shared<ExecutorHandler>(runTime_->thisDb(), dict,
                                                 "dummy", true);

        auto comm = exec->get_gko_mpi_host_comm();
        if (comm->size() != 4 || Pstream::nProcs() != 4) {
            std::cout << "This unit test expects to be run on 4 ranks"
                      << std::endl;
            std::abort();
        }

        // delete listener on ranks != 0
        // to clean up output
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        if (Foam::Pstream::myProcNo() != 0) {
            delete listeners.Release(listeners.default_result_printer());
        }

        word fieldName{"p"};
        dimensionSet ds{0, 0, 0, 0, 0};
        field = std::make_shared<
            GeometricField<scalar, Foam::fvPatchField, Foam::volMesh>>(
            Foam::IOobject(fieldName, runTime_->timeName(), runTime_->thisDb(),
                           Foam::IOobject::MUST_READ),
            *mesh.get(), ds);

        Foam::fvMatrix<scalar> fvMatrix{*field.get(), ds};

        interfaces = field->boundaryField().scalarInterfaces();

        hostMatrix = std::make_shared<HostMatrixWrapper>(
            *exec.get(), runTime_->thisDb(), mesh->lduAddr(), true,
            fvMatrix.diag().data(), fvMatrix.upper().data(),
            fvMatrix.lower().data(), fvMatrix.boundaryCoeffs(),
            fvMatrix.internalCoeffs(), interfaces, dict, "fieldName", 0);
    }

    Foam::lduInterfaceFieldPtrsList interfaces;
    Foam::PtrList<Foam::lduInterfaceField> newInterfaces;
    std::shared_ptr<Foam::argList> args_;
    std::shared_ptr<Foam::Time> runTime_;
    std::shared_ptr<fvMesh> mesh;
    Foam::dictionary dict;
    std::shared_ptr<GeometricField<scalar, Foam::fvPatchField, Foam::volMesh>>
        field;
    std::shared_ptr<const ExecutorHandler> exec;
    std::shared_ptr<const HostMatrixWrapper> hostMatrix;
};

const testing::Environment *global_env =
    AddGlobalTestEnvironment(new Environment);

class DistributedMatrixFixture :
     public testing::TestWithParam<int> {
public:
    ExecutorHandler exec =
        *((Environment *)global_env)->exec.get();
    label rank = exec.get_rank();
    const gko::experimental::mpi::communicator comm =
        *(exec.get_communicator().get());
};

INSTANTIATE_TEST_SUITE_P(DistributedMatrixFixtureInstantiation,
                         DistributedMatrixFixture, testing::Values(1, 2, 4));

TEST_P(DistributedMatrixFixture, canCreateDistributeMatrix)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    label ranks_per_gpu = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto repartitioner = Repartitioner(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);

    std::map<label, vec> exp_local_size;
    exp_local_size.emplace(1, vec{9, 9, 9, 9});
    exp_local_size.emplace(2, vec{18, 0, 18, 0});
    exp_local_size.emplace(4, vec{36, 0 ,0 ,0});

    std::map<label, vec> exp_non_local_size;
    exp_non_local_size.emplace(1, vec{6, 6, 6, 6});
    exp_non_local_size.emplace(2, vec{6, 0, 6, 0});
    exp_non_local_size.emplace(4, vec{0, 0 ,0 ,0});

    auto distributed = RepartDistMatrix<scalar, label, label>::create(exec, "Coo", repartitioner, hostMatrix);

    ASSERT_EQ(distributed->get_local_matrix()->get_size()[0], exp_local_size[ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_local_matrix()->get_size()[1], exp_local_size[ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_non_local_matrix()->get_size()[0], exp_local_size[ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_non_local_matrix()->get_size()[1], exp_non_local_size[ranks_per_gpu][rank]);

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
