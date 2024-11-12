// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "OGL/MatrixWrapper/HostMatrix.H"


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
    std::shared_ptr<Foam::fvMatrix<scalar>> fvMatrix;
    std::shared_ptr<const HostMatrixWrapper> hostMatrix;
};

const testing::Environment *global_env =
    AddGlobalTestEnvironment(new HostMatrixEnvironment);


TEST(HostMatrixP2D, returnsCorrectSize)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto mesh = ((HostMatrixEnvironment *)global_env)->mesh;
    auto fvMatrix = ((HostMatrixEnvironment *)global_env)->fvMatrix;
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto rank = exec->get_rank();

    // first and last rank have a non interface boundary (upper and lower wall)
    std::vector<label> exp_num_interfaces{3, 3, 3, 3};

    std::vector<std::vector<label>> exp_interface_length{
        {2, 2, 4}, {2, 2, 4}, {4, 2, 2}, {4, 2, 2}};

    // the local size is 4 * 2 = 8
    // [ 4 5 6 7 | 4 5 6 7 ]
    // [ 0 1 2 3 | 0 1 2 3 ]
    // ----------+----------
    // [ 4 5 6 7 | 4 5 6 7 ]
    // [ 0 1 2 3 | 0 1 2 3 ]
    // the local size is 2 * 4 = 8
    EXPECT_EQ(mesh->C().size(), 8);
    // which results in a 8x8 matrix
    EXPECT_EQ(hostMatrix->get_size()[0], 8);
    EXPECT_EQ(hostMatrix->get_size()[1], 8);

    EXPECT_EQ(hostMatrix->get_local_nrows(), 8);

    EXPECT_EQ(hostMatrix->get_num_interfaces(), exp_num_interfaces[rank]);
    EXPECT_EQ(hostMatrix->get_local_matrix_nnz(), 28);
    EXPECT_EQ(hostMatrix->get_interface_length(), exp_interface_length[rank]);
}

TEST(HostMatrixP2D, canCreateCommunicationPattern)
{
    std::shared_ptr<const HostMatrixWrapper> hostMatrix =
        ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto commPattern = hostMatrix->create_communication_pattern();
    auto comm = commPattern->get_comm();
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto rank = exec->get_rank();

    // number of neighbours
    // cyclic bc are empty and converted into
    // processorCyclic and two regular processor bc
    std::vector<label> exp_send_idx_size{2, 2, 2, 2};
    EXPECT_EQ(commPattern->send_idxs.size(), exp_send_idx_size[rank]);

    std::vector<std::vector<label>> target_ids_exp{
        {1, 2}, {0, 3}, {0, 3}, {1, 2}};
    std::vector<label> target_ids_res(
        commPattern->target_ids.data(),
        commPattern->target_ids.data() + exp_send_idx_size[rank]);
    EXPECT_EQ(target_ids_exp[comm.rank()], target_ids_res);

    std::vector<std::vector<label>> target_sizes_exp{
        {4, 4}, {4, 4}, {4, 4}, {4, 4}};
    std::vector<label> target_size_res(
        commPattern->target_sizes.data(),
        commPattern->target_sizes.data() + exp_send_idx_size[rank]);
    EXPECT_EQ(target_sizes_exp[comm.rank()], target_size_res);
}

TEST(HostMatrixP2D, canGenerateLocalSparsityPattern)
{
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;

    auto [localSparsity, nonLocalSparsity] =
        hostMatrix->compute_sparsity_patterns(exec->get_device_exec());
    std::vector<label> rows_expected({0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                      2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                                      5, 6, 6, 6, 6, 7, 7, 7});
    std::vector<label> cols_expected({0, 1, 4, 0, 1, 2, 5, 1, 2, 3,
                                      6, 2, 3, 7, 0, 4, 5, 1, 4, 5,
                                      6, 2, 5, 6, 7, 3, 6, 7});

    std::vector<label> mapping_expected({20, 0,  1,  10, 21, 2,  3,  12, 22, 4,
                                         5,  14, 23, 6,  11, 24, 7,  13, 17, 25,
                                         8,  15, 18, 26, 9,  16, 19, 27});

    // we have 8x8 matrix with 26 nnz entries
    EXPECT_EQ(localSparsity->dim[0], 8);
    EXPECT_EQ(localSparsity->dim[1], 8);
    EXPECT_EQ(localSparsity->num_nnz, 28);

    // since we don't have any processor interfaces we only have
    // a single interface span ranging from 0 to 33
    EXPECT_EQ(localSparsity->spans.size(), 1);
    EXPECT_EQ(localSparsity->spans[0].begin, 0);
    EXPECT_EQ(localSparsity->spans[0].end, localSparsity->num_nnz);

    auto rows_res = convert_to_vector(localSparsity->row_idxs);
    EXPECT_EQ(rows_expected, rows_res);

    auto cols_res = convert_to_vector(localSparsity->col_idxs);
    EXPECT_EQ(cols_expected, cols_res);

    auto mapping_res = convert_to_vector(localSparsity->ldu_mapping);
    EXPECT_EQ(mapping_expected, mapping_res);
}

TEST(HostMatrixP2D, canGenerateNonLocalSparsityPattern)
{
    auto hostMatrix = ((HostMatrixEnvironment *)global_env)->hostMatrix;
    auto exec = ((HostMatrixEnvironment *)global_env)->exec;
    auto comm = exec->get_gko_mpi_device_comm();
    auto rank = exec->get_rank();

    auto [localSparsity, nonLocalSparsity] =
        hostMatrix->compute_sparsity_patterns(exec->get_device_exec());

    // corresponds to cell ids
    std::vector<std::vector<label>> rows_expected{{3, 7, 0, 4, 4, 5, 6, 7},
                                                  {0, 4, 3, 7, 4, 5, 6, 7},
                                                  {0, 1, 2, 3, 3, 7, 0, 4},
                                                  {0, 1, 2, 3, 0, 4, 3, 7}};

    // cols expected
    std::vector<std::vector<label>> cols_expected({{0, 4, 3, 7, 0, 1, 2, 3},
                                                   {3, 7, 0, 4, 0, 1, 2, 3},
                                                   {4, 5, 6, 7, 0, 4, 3, 7},
                                                   {4, 5, 6, 7, 3, 7, 0, 4}});

    std::vector<std::vector<label>> mapping_expected{{0, 1, 2, 3, 4, 5, 6, 7},
                                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                                     {0, 1, 2, 3, 4, 5, 6, 7}};

    // we dont test the cols expected for now,
    // as they are in compressed format
    std::vector<label> exp_send_idx_size{8, 8, 8, 8};
    EXPECT_EQ(nonLocalSparsity->num_nnz, exp_send_idx_size[rank]);
    // number of interfaces
    std::vector<label> exp_spans_size{3, 3, 3, 3};
    EXPECT_EQ(nonLocalSparsity->spans.size(), exp_spans_size[rank]);

    // EXPECT_EQ(nonLocalSparsity->spans[0].begin, 0);
    // EXPECT_EQ(nonLocalSparsity->spans[1].begin, 3);
    // EXPECT_EQ(nonLocalSparsity->spans[0].end, 3);
    // EXPECT_EQ(nonLocalSparsity->spans[1].end, 6);

    auto rows_res = convert_to_vector(nonLocalSparsity->row_idxs);
    EXPECT_EQ(rows_expected[rank], rows_res);
    auto mapping_res = convert_to_vector(nonLocalSparsity->ldu_mapping);
    EXPECT_EQ(mapping_expected[rank], mapping_res);
    auto cols_res = convert_to_vector(nonLocalSparsity->col_idxs);
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
