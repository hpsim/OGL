// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include "OGL/MatrixWrapper/HostMatrix.H"

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


class HostMatrixEnvironment : public testing::Environment {
public:
    void SetUp()
    {
        dict.add("executor", "reference");

        args_ = std::make_shared<Foam::argList>(my_argc, my_argv);
        if (args_->size() == 1) {
            std::cout << "Wrong number of arguments detected, make sure to run "
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
        dimensionSet ds{};
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
    Foam::PtrDynList<Foam::lduInterfaceField> newInterfaces;
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
    AddGlobalTestEnvironment(new HostMatrixEnvironment);


TEST(HostMatrix, returnsCorrectSize)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto mesh = ((HostMatrixEnvironment *)global_env)->mesh;
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;

    // the local size is 9
    EXPECT_EQ(mesh->C().size(), 9);
    // which results in a 9x9 matrix
    EXPECT_EQ(hostMatrix->get_size()[0], 9);
    EXPECT_EQ(hostMatrix->get_size()[1], 9);

    EXPECT_EQ(hostMatrix->get_local_nrows(), 9);
}


TEST(HostMatrix, canCreateCommunicationPattern)
{
    std::shared_ptr<const HostMatrixWrapper> hostMatrix =
        ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto commPattern = hostMatrix->create_communication_pattern();
    auto comm = commPattern.get_comm();

    EXPECT_EQ(commPattern.send_idxs.size(), 2);

    std::vector<std::vector<label>> target_ids_exp{
        {1, 2}, {0, 3}, {0, 3}, {1, 2}};
    std::vector<label> target_ids_res(commPattern.target_ids.get_data(),
                                      commPattern.target_ids.get_data() + 2);
    EXPECT_EQ(target_ids_exp[comm.rank()], target_ids_res);

    std::vector<std::vector<label>> target_sizes_exp{
        {3, 3}, {3, 3}, {3, 3}, {3, 3}};
    std::vector<label> target_size_res(commPattern.target_sizes.get_data(),
                                       commPattern.target_sizes.get_data() + 2);
    EXPECT_EQ(target_sizes_exp[comm.rank()], target_size_res);
}

TEST(HostMatrix, canGenerateLocalSparsityPattern)
{
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;

    auto localSparsity =
        hostMatrix->compute_local_sparsity(exec->get_device_exec());
    std::vector<label> rows_expected({0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3,
                                      3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
                                      5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8});
    std::vector<label> cols_expected({0, 1, 3, 0, 1, 2, 4, 1, 2, 5, 0,
                                      3, 4, 6, 1, 3, 4, 5, 7, 2, 4, 5,
                                      8, 3, 6, 7, 4, 6, 7, 8, 5, 7, 8});

    // we have 9x9 matrix with 33 nnz entries
    EXPECT_EQ(localSparsity->num_nnz, 33);
    EXPECT_EQ(localSparsity->dim[0], 9);
    EXPECT_EQ(localSparsity->dim[1], 9);

    // since we don't have any processor interfaces we only have
    // a single interface span ranging from 0 to 33
    EXPECT_EQ(localSparsity->spans.size(), 1);
    EXPECT_EQ(localSparsity->spans[0].begin, 0);
    EXPECT_EQ(localSparsity->spans[0].end, 33);

    auto res_size{localSparsity->col_idxs.get_size()};
    std::vector<label> rows_res(localSparsity->row_idxs.get_data(),
                                localSparsity->row_idxs.get_data() + res_size);
    std::vector<label> cols_res(localSparsity->col_idxs.get_data(),
                                localSparsity->col_idxs.get_data() + res_size);

    EXPECT_EQ(rows_expected, rows_res);
    EXPECT_EQ(cols_expected, cols_res);
}

TEST(HostMatrix, canGenerateNonLocalSparsityPattern)
{
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto comm = exec->get_gko_mpi_device_comm();

    auto nonLocalSparsity =
        hostMatrix->compute_non_local_sparsity(exec->get_device_exec());

    // corresponds to cell ids
    std::vector<std::vector<label>> rows_expected({{2, 5, 8, 6, 7, 8},
                                                   {0, 3, 6, 6, 7, 8},
                                                   {0, 1, 2, 2, 5, 8},
                                                   {0, 1, 2, 0, 3, 6}});
    // cols expected
    std::vector<std::vector<label>> cols_expected({{0, 3, 6, 0, 1, 2},
                                                   {2, 5, 8, 0, 1, 2},
                                                   {6, 7, 8, 0, 3, 6},
                                                   {6, 7, 8, 2, 5, 8}});

    // we dont test the cols expected for now,
    // as they are in compressed format
    EXPECT_EQ(nonLocalSparsity->num_nnz, 6);
    EXPECT_EQ(nonLocalSparsity->spans.size(), 2);

    EXPECT_EQ(nonLocalSparsity->spans[0].begin, 0);
    EXPECT_EQ(nonLocalSparsity->spans[0].end, 3);
    EXPECT_EQ(nonLocalSparsity->spans[1].begin, 3);
    EXPECT_EQ(nonLocalSparsity->spans[1].end, 6);

    auto res_size{nonLocalSparsity->row_idxs.get_size()};
    std::vector<label> rows_res(
        nonLocalSparsity->row_idxs.get_data(),
        nonLocalSparsity->row_idxs.get_data() + res_size);
    std::vector<label> cols_res(
        nonLocalSparsity->col_idxs.get_data(),
        nonLocalSparsity->col_idxs.get_data() + res_size);
    EXPECT_EQ(rows_expected[comm->rank()], rows_res);
    EXPECT_EQ(cols_expected[comm->rank()], cols_res);
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
