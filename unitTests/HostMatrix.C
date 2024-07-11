// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include "OGL/MatrixWrapper/HostMatrix.H"

#include "gtest/gtest.h"

//---------------------------------------------
// some_header.h
extern int my_argc;
extern char** my_argv;
// eof
//---------------------------------------------

//---------------------------------------------
// main.cpp
int my_argc;
char** my_argv;


class HostMatrixFixture : public testing::Test {

protected:
    HostMatrixFixture()
        : interfaces(), interfaceBouCoeffs(), interfaceIntCoeffs()
    {}

    void SetUp(const std::string &filename) {
        std::cout << "call setup\n";
        // for OpenFOAMs addressing see
        // https://openfoamwiki.net/index.php/OpenFOAM_guide/Matrices_in_OpenFOAM
        //
        std::vector<scalar> d{1., 2., 3., 4., 5.};
        std::vector<scalar> u{10., 11., 20., 12., 21., 13.};

        label total_nnz{17};
        label upper_nnz{6};

        dict.add("executor", "reference");

        // Foam::labelList rows({0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4});
        // Foam::labelList cols({0, 1, 3, 0, 1, 2, 4, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4});
        // std::make_shared<Foam::lduPrimitiveMesh>(0, rows, cols, 0, true );
        //

        std::cout << __FILE__ << "create argList\n";
        args_ =std::make_shared<Foam::argList> (argc, argv, false, false, false);
        std::cout << __FILE__ << "set parallel option\n";
        args_->setOption("parallel");

        std::cout << __FILE__ << "create runtime\n";
        runTime_ = std::make_shared<Foam::Time>("case", *args_.get());
        std::cout << __FILE__ << "create mesh\n";
        mesh =
            std::make_shared<Foam::fvMesh>
            (
                Foam::IOobject
                (
                    word {""},
                    runTime_->timeName(),
                    *runTime_.get(),
                    Foam::IOobject::MUST_READ
                ),
                false
            );

        exec = std::make_shared<ExecutorHandler>(runTime_->thisDb(), dict, "dummy", true);
        word fieldName {"p"};
        dimensionSet ds {};
        auto field = GeometricField<scalar, Foam::fvPatchField, Foam::volMesh>
        (
                Foam::IOobject
                (
                    fieldName,
                    runTime_->timeName(),
                    runTime_->thisDb(),
                    Foam::IOobject::MUST_READ
                ),
            *mesh.get(),
            ds
        );
        //
        // auto matrix = fvMatrix<scalar>(field, ds);

        hostMatrix = std::make_shared<HostMatrixWrapper>(
                *exec.get(), runTime_->thisDb(), 5, upper_nnz, true,
                d.data(), u.data(), u.data(), mesh->lduAddr(),
                interfaceBouCoeffs, interfaceIntCoeffs, interfaces,
                dict, "fieldName", 0
                );
    }

    int argc {0};
    char** argv;
    std::shared_ptr<Foam::argList> args_;
    std::shared_ptr<Foam::Time> runTime_;
    std::shared_ptr<fvMesh> mesh;
    Foam::dictionary dict;
    const Foam::lduInterfaceFieldPtrsList interfaces;
    const Foam::FieldField<Field, scalar> interfaceBouCoeffs;
    const Foam::FieldField<Field, scalar> interfaceIntCoeffs;
    std::shared_ptr<ExecutorHandler> exec;
    std::shared_ptr<const HostMatrixWrapper> hostMatrix;
};

TEST_F(HostMatrixFixture, returnsCorrectSize)
{
    EXPECT_EQ(hostMatrix->get_size()[0], 5);
    EXPECT_EQ(hostMatrix->get_size()[1], 5);
}

TEST_F(HostMatrixFixture, canCreateCommunicationPattern){
    auto commPattern = hostMatrix->create_communication_pattern();
}

TEST_F(HostMatrixFixture, canGenerateLocalSparsityPattern)
{
    auto localSparsity = hostMatrix->compute_local_sparsity(exec->get_device_exec());
    std::vector<label> rows({0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4});
    std::vector<label> cols({0, 1, 3, 0, 1, 2, 4, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4});

    // we have 5x5 matrix with 17 nnz entries
    EXPECT_EQ(localSparsity->size_, 17);
    EXPECT_EQ(localSparsity->dim[0], 5);
    EXPECT_EQ(localSparsity->dim[1], 5);

    // since we don't have any processor interfaces we only have
    // a single interface span ranging from 0 to 17
    EXPECT_EQ(localSparsity->interface_spans.size(), 1);
    EXPECT_EQ(localSparsity->interface_spans[0].begin, 0);
    EXPECT_EQ(localSparsity->interface_spans[0].end, 17);

    // TODO implement
    for (int i=0;i<rows.size();i++) {
    //     EXPECT_EQ(localSparsity->row_idxs.get_data()[i], rows[i]);
    //     EXPECT_EQ(localSparsity->col_idxs.get_data()[i], cols[i]);
    }
}


int main(int argc, char *argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    my_argc = argc;
    my_argv = argv;

    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
