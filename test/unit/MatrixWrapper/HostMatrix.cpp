// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "OGL/MatrixWrapper/HostMatrix.hpp"

#include "HostMatrixData.hpp"

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


class HostMatrixEnvironment : public testing::Environment {
public:
    void SetUp()
    {
        dict.add("executor", "reference");
        args_ = std::make_shared<Foam::argList>(my_argc, my_argv);

        name_ = args_->globalCaseName();

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

        auto comm = exec->get_host_comm();
        if (comm->size() != 4 || Pstream::nProcs() != 4) {
            std::cout << "This unit test expects to be run on 4 ranks"
                      << std::endl;
            std::abort();
        }

        // delete listener on ranks != 0
        // to clean up output
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        // if (Foam::Pstream::myProcNo() != 0) {
        //     delete listeners.Release(listeners.default_result_printer());
        // }

        word fieldName{"p"};
        dimensionSet ds{0, 0, 0, 0, 0};
        field = std::make_shared<
            GeometricField<scalar, Foam::fvPatchField, Foam::volMesh>>(
            Foam::IOobject(fieldName, runTime_->timeName(), runTime_->thisDb(),
                           Foam::IOobject::MUST_READ),
            *mesh.get(), ds);

        fvMatrix = std::make_shared<Foam::fvMatrix<scalar>>(*field.get(), ds);

        interfaces = field->boundaryField().scalarInterfaces();

        hostMatrix = std::make_shared<HostMatrixWrapper>(
            *exec.get(), runTime_->thisDb(), mesh->lduAddr(),
            fvMatrix->symmetric(), fvMatrix->diag().data(),
            fvMatrix->upper().data(), fvMatrix->lower().data(),
            fvMatrix->boundaryCoeffs(), fvMatrix->internalCoeffs(), interfaces,
            dict, "fieldName", 0);

        std::cout << __FILE__ << __LINE__ << " done hostmatrixwrapper \n";
        partition_ = gko::share(
            gko::experimental::distributed::build_partition_from_local_size<
                label, label>(exec->get_ref_exec(),
                              *exec->get_host_comm().get(), exp_size[name_]));
        std::cout << __FILE__ << __LINE__ << " done partition \n";
    }

    std::string name_;
    Foam::lduInterfaceFieldPtrsList interfaces;
    Foam::PtrList<Foam::lduInterfaceField> newInterfaces;
    std::shared_ptr<Foam::argList> args_;
    std::shared_ptr<Foam::Time> runTime_;
    std::shared_ptr<fvMesh> mesh;
    Foam::dictionary dict;
    std::shared_ptr<GeometricField<scalar, Foam::fvPatchField, Foam::volMesh>>
        field;
    std::shared_ptr<const ExecutorHandler> exec;
    std::shared_ptr<Foam::fvMatrix<scalar>> fvMatrix;
    std::shared_ptr<const HostMatrixWrapper> hostMatrix;
    std::shared_ptr<gko::experimental::distributed::Partition<label, label>>
        partition_;
};

const testing::Environment *global_env =
    AddGlobalTestEnvironment(new HostMatrixEnvironment);


TEST(HostMatrixTest, returnsCorrectSize)
{
    auto mesh = ((HostMatrixEnvironment *)global_env)->mesh;
    auto fvMatrix = ((HostMatrixEnvironment *)global_env)->fvMatrix;
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto rank = exec->get_host_rank();
    auto name = ((HostMatrixEnvironment *)global_env)->name_;

    EXPECT_EQ(mesh->C().size(), exp_size[name]);
    EXPECT_EQ(hostMatrix->get_size()[0], exp_size[name]);
    EXPECT_EQ(hostMatrix->get_size()[1], exp_size[name]);
    EXPECT_EQ(hostMatrix->get_local_nrows(), exp_size[name]);
    EXPECT_EQ(hostMatrix->get_num_interfaces(), exp_num_interface[name][rank]);
}

TEST(HostMatrixTest, canCreateCommunicationPattern)
{
    std::shared_ptr<const HostMatrixWrapper> hostMatrix =
        ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto commPattern = hostMatrix->create_communication_pattern();
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto rank = exec->get_host_rank();
    auto name = ((HostMatrixEnvironment *)global_env)->name_;

    EXPECT_EQ(commPattern->send_idxs.size(), exp_send_idx_size[name][rank]);
    EXPECT_EQ(commPattern->target_ids, exp_target_ids[name][rank]);
    EXPECT_EQ(commPattern->target_sizes, exp_target_sizes[name][rank]);
}

TEST(HostMatrixTest, canGenerateLocalSparsityPattern)
{
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto name = ((HostMatrixEnvironment *)global_env)->name_;
    auto partition = ((HostMatrixEnvironment *)global_env)->partition_;

    auto [localSparsity, nonLocalSparsity] =
        hostMatrix->compute_sparsity_patterns(partition);

    EXPECT_EQ(localSparsity->get_rows(), exp_local_rows[name]);
    EXPECT_EQ(localSparsity->get_cols(), exp_local_cols[name]);
    EXPECT_EQ(localSparsity->get_map(), exp_local_map[name]);
}

TEST(HostMatrixTest, canGenerateNonLocalSparsityPattern)
{
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto rank = exec->get_host_rank();
    auto name = ((HostMatrixEnvironment *)global_env)->name_;
    auto partition = ((HostMatrixEnvironment *)global_env)->partition_;

    auto [localSparsity, nonLocalSparsity] =
        hostMatrix->compute_sparsity_patterns(partition);

    // we dont test the cols expected for now,
    // as they are in compressed format
    EXPECT_EQ(nonLocalSparsity->get_rows(), exp_non_local_rows[name][rank]);
    EXPECT_EQ(nonLocalSparsity->get_cols(), exp_non_local_cols[name][rank]);
    EXPECT_EQ(nonLocalSparsity->get_map(), exp_non_local_map[name][rank]);
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
