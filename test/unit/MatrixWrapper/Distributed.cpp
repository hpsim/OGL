// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include "OGL/MatrixWrapper/Distributed.hpp"
#include "OGL/MatrixWrapper/HostMatrix.hpp"
#include "OGL/Repartitioner.hpp"

#include "DistributedData.hpp"

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


template <typename ValueType, typename IndexType>
std::vector<ValueType> convert_to_vector(
    std::pair<IndexType, const ValueType *> in)
{
    auto [size, ptr] = in;
    return std::vector<ValueType>(ptr, ptr + size);
}


class Environment : public testing::Environment {
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
        field = std::make_shared<
            GeometricField<scalar, Foam::fvPatchField, Foam::volMesh>>(
            Foam::IOobject(fieldName, runTime_->timeName(), runTime_->thisDb(),
                           Foam::IOobject::MUST_READ),
            *mesh.get());

        dimensionSet ds{0, 0, 0, 0, 0};
        fvMatrix = std::make_shared<Foam::fvMatrix<scalar>>(*field.get(), ds);

        interfaces = field->boundaryField().scalarInterfaces();

        hostMatrix = std::make_shared<HostMatrixWrapper>(
            *exec.get(), runTime_->thisDb(), mesh->lduAddr(),
            fvMatrix->symmetric(), fvMatrix->diag().data(),
            fvMatrix->upper().data(), fvMatrix->lower().data(),
            fvMatrix->boundaryCoeffs(), fvMatrix->internalCoeffs(), interfaces,
            dict, "fieldName", 0);

        // Set all matrix coefficients to some value
        // in order to make the apply test give actual results
        // this is required since creating the host matrix
        // without actual DSL sets all values to zero
        for (int i = 0; i < 12; i++) {
            fvMatrix->upper().data()[i] = 1.0;
            fvMatrix->lower().data()[i] = 2.0;
        }
        for (int i = 0; i < 9; i++) {
            fvMatrix->diag().data()[i] = 3.0;
        }

        // set the interface value, we use get_interface_data here
        // because that is more comfortable.
        for (auto const &[id, value] : hostMatrix->get_interfaces()) {
            if (id >= 0) {
                continue;
            }
            auto [length, const_ptr] = value;
            scalar *data = const_cast<scalar *>(const_ptr);
            data[0] = -1.0;
            data[1] = -2.0;
            data[2] = -3.0;
        }
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
};

const testing::Environment *global_env =
    AddGlobalTestEnvironment(new Environment);

class DistMatL2D
    : public testing::TestWithParam<std::tuple<int, string, bool>> {
public:
    ExecutorHandler exec = *((Environment *)global_env)->exec.get();
    label rank = exec.get_host_rank();
    const gko::experimental::mpi::communicator comm =
        *(exec.get_host_comm().get());
};


INSTANTIATE_TEST_SUITE_P(
    DistMatL2DInit, DistMatL2D,
    testing::Combine(
        testing::Values(1, 2, 4), testing::Values("Coo"),
        testing::Values(true)),  // for now only support fused matrices
    [](const auto &info) {
        // Can use info.param here to generate the test
        // suffix
        std::vector<std::string> names;
        names.emplace_back("ranks");
        names.emplace_back("format");
        names.emplace_back("fuse");
        std::string name = "ranks_";
        name += std::to_string(std::get<0>(info.param));
        name += "_format_";
        name += std::get<1>(info.param);
        name += "_fused_";
        name += std::to_string(std::get<2>(info.param));
        return name;
    });


TEST_P(DistMatL2D, canCreateDistributedMatrix)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto [ranks_per_gpu, matrix_format, fused] = GetParam();

    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);
    auto name = ((Environment *)global_env)->name_;

    gko::dim<2> global_vec_dim{repartitioner->get_orig_partition()->get_size(),
                               1};
    gko::dim<2> local_vec_dim{repartitioner->get_repart_dim()[0], 1};

    auto distributed = create_distributed(exec, repartitioner, hostMatrix,
                                          matrix_format, fused, 0);

    ASSERT_EQ(distributed->get_local_matrix()->get_size()[0],
              exp_local_size[name][ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_local_matrix()->get_size()[1],
              exp_local_size[name][ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_non_local_matrix()->get_size()[0],
              exp_local_size[name][ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_local_matrix()->get_size()[0],
              exp_local_size[name][ranks_per_gpu][rank]);
}

TEST_P(DistMatL2D, hasCorrectLocalMatrix)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto [ranks_per_gpu, matrix_format, fused] = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);
    auto name = ((Environment *)global_env)->name_;

    gko::dim<2> global_vec_dim{repartitioner->get_orig_partition()->get_size(),
                               1};
    gko::dim<2> local_vec_dim{repartitioner->get_repart_dim()[0], 1};

    auto distributed = create_distributed(exec, repartitioner, hostMatrix,
                                          matrix_format, fused, 0);

    auto local =
        (fused) ? gko::as<gko::matrix::Coo<scalar, label>>(
                      distributed->get_local_matrix())
                : detail::convert_combination_to_coo(
                      exec.get_ref_exec(), distributed->get_local_matrix());

    ASSERT_EQ(distributed->get_local_matrix()->get_size()[1],
              exp_local_size[name][ranks_per_gpu][rank]);

    auto res_local_coeffs = convert_to_vector(get_val(local));
    auto res_local_cols = convert_to_vector(get_col(local));
    auto res_local_rows = convert_to_vector(get_row(local));

    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_rows[i],
                  exp_local_rows[name][fused][ranks_per_gpu][rank][i])
            << " failed at index " << i << " on rank " << rank;
    }

    EXPECT_EQ(res_local_cols.size(),
              exp_local_cols[name][fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_cols[i],
                  exp_local_cols[name][fused][ranks_per_gpu][rank][i])
            << " failed at index " << i << " on rank " << rank;
    }

    EXPECT_EQ(res_local_coeffs.size(),
              exp_local_coeffs[name][fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_coeffs[i],
                  exp_local_coeffs[name][fused][ranks_per_gpu][rank][i])
            << " failed at index " << i << " on rank " << rank;
    }
}


TEST_P(DistMatL2D, hasCorrectNonLocalMatrix)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto [ranks_per_gpu, matrix_format, fused] = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto name = ((Environment *)global_env)->name_;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);

    std::map<label, vec> exp_non_local_size;
    exp_non_local_size.emplace(1, vec{6, 6, 6, 6});
    exp_non_local_size.emplace(2, vec{6, 0, 6, 0});
    exp_non_local_size.emplace(4, vec{0, 0, 0, 0});

    auto distributed = create_distributed(exec, repartitioner, hostMatrix,
                                          matrix_format, fused, 0);

    auto non_local =
        (fused) ? gko::as<gko::matrix::Coo<scalar, label>>(
                      distributed->get_non_local_matrix())
                : detail::convert_combination_to_coo(
                      exec.get_ref_exec(), distributed->get_non_local_matrix());

    auto res_non_local_coeffs = convert_to_vector(get_val(non_local));
    auto res_non_local_cols = convert_to_vector(get_col(non_local));
    auto res_non_local_rows = convert_to_vector(get_row(non_local));

    ASSERT_EQ(distributed->get_non_local_matrix()->get_size()[1],
              exp_non_local_size[ranks_per_gpu][rank]);

    ASSERT_EQ(res_non_local_coeffs,
              exp_non_local_coeffs[name][fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_rows,
              exp_non_local_rows[name][fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_cols,
              exp_non_local_cols[name][fused][ranks_per_gpu][rank]);
}

TEST_P(DistMatL2D, canApplyCorrectly)
{
    auto [ranks_per_gpu, format, fused] = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto name = ((Environment *)global_env)->name_;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);

    auto distributed =
        create_distributed(exec, repartitioner, hostMatrix, format, fused, 0);

    gko::dim<2> global_vec_dim{repartitioner->get_orig_partition()->get_size(),
                               1};
    gko::dim<2> local_vec_dim{repartitioner->get_repart_dim()[0], 1};

    auto b = gko::share(gko::experimental::distributed::Vector<scalar>::create(
        exec.get_ref_exec(), comm, global_vec_dim, local_vec_dim, 1));
    b->fill(1);

    auto x = gko::share(gko::experimental::distributed::Vector<scalar>::create(
        exec.get_ref_exec(), comm, global_vec_dim, local_vec_dim, 1));
    x->fill(0);

    // Act
    distributed->apply(b, x);
    auto res_x = std::vector<scalar>(
        x->get_local_vector()->get_const_values(),
        x->get_local_vector()->get_const_values() + local_vec_dim[0]);

    ASSERT_EQ(res_x, exp_x[name][fused][ranks_per_gpu][rank]);
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
