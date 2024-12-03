// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#include "OGL/MatrixWrapper/Distributed.H"
#include "OGL/MatrixWrapper/HostMatrix.H"
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

using vec_vec_s = std::vector<std::vector<scalar>>;


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
    label rank = exec.get_rank();
    const gko::experimental::mpi::communicator comm =
        *(exec.get_communicator().get());

    std::map<label, vec> exp_local_size{
        {1, {9, 9, 9, 9}}, {2, {18, 0, 18, 0}}, {4, {36, 0, 0, 0}}};

    /*
     * The mesh has the following structure
     *         global ids
     *        [24 25 26|33 34 35]
     *        [21 22 23|30 31 32]
     *   2    [18 19 20|27 28 29]  3
     *        ---------+---------
     *        [ 6  7  8|15 16 17]
     *        [ 3  4  5|12 13 14]
     *   0    [ 0  1  2| 9 10 11]  1
     *   */
    std::vector<scalar> exp_local_coeff_1_nf{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                             2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    std::vector<scalar> exp_local_coeff_1_f{3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2,
                                            3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3,
                                            1, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3};

    std::vector<scalar> exp_local_coeff_2_nf{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 3, 1, 2, 3};

    std::vector<scalar> exp_local_coeff_2_f{
        3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1,           // cell 0-2
        2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 2,  //  cell: 3 -5
        2, 3, 1, 2, 2, 3, 1, 2, 2, 3, 3,           // cell: 6-8
        1, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1,           // cell: 9-11
        2, 2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1,  // cell 12-14
        3, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3            // cell 15-17
    };
    std::vector<scalar> exp_local_coeff_4{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};

    std::vector<scalar> exp_local_coeff_4_f{
        3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2,
        3, 1, 2,  //
        2, 3, 1, 1, 2, 2, 3, 1, 2, 2, 2, 3, 3, 3, 1, 3, 1, 1, 2, 3, 1, 1,
        2, 3, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 3, 2, 3, 1, 1,
        2, 2, 3, 1, 2, 2, 2, 3, 3, 1, 3, 1, 1, 2, 2, 3, 1, 1, 3, 2, 3, 1,
        1, 2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3, 1,
        2, 2, 3, 3, 1, 1, 3, 1, 1, 2, 2, 3, 1, 1, 3, 2, 3, 1, 2, 2, 3, 1,
        1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 3, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3};

    std::map<bool, std::map<label, vec_vec_s>> exp_local_coeffs{
        {true,
         {{1,
           {exp_local_coeff_1_f, exp_local_coeff_1_f, exp_local_coeff_1_f,
            exp_local_coeff_1_f}},
          {2, {exp_local_coeff_2_f, {}, exp_local_coeff_2_f, {}}},
          {4, {exp_local_coeff_4_f, {}, {}, {}}}}},
        {false,
         {{1,
           {exp_local_coeff_1_nf, exp_local_coeff_1_nf, exp_local_coeff_1_nf,
            exp_local_coeff_1_nf}},
          {2, {exp_local_coeff_2_nf, {}, exp_local_coeff_2_nf, {}}},
          {4, {exp_local_coeff_4, {}, {}, {}}}}}};

    std::vector<scalar> non_local_coeffs{1, 2, 3, 1, 2, 3};
    std::map<bool, std::map<label, vec_vec_s>> exp_non_local_coeffs{
        {true,
         {{1,
           {{1, 2, 1, 2, 3, 3},
            {1, 2, 3, 1, 2, 3},
            {1, 2, 3, 1, 2, 3},
            {1, 1, 2, 3, 2, 3}}},
          {2, {non_local_coeffs, {}, non_local_coeffs, {}}},
          {4, {{}, {}, {}, {}}}}},
        {false,
         {{1,
           {non_local_coeffs, non_local_coeffs, non_local_coeffs,
            non_local_coeffs}},
          {2, {{1, 2, 3, 1, 2, 3}, {}, {1, 2, 3, 1, 2, 3}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_non_local_rows{
        {true,
         {{1,
           {{2, 5, 6, 7, 8, 8},
            {0, 3, 6, 6, 7, 8},
            {0, 1, 2, 2, 5, 8},
            {0, 0, 1, 2, 3, 6}}},
          {2, {{6, 7, 8, 15, 16, 17}, {}, {0, 1, 2, 9, 10, 11}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {false,
         {{1,
           {{2, 5, 8, 6, 7, 8},
            {0, 3, 6, 6, 7, 8},
            {0, 1, 2, 2, 5, 8},
            {0, 1, 2, 0, 3, 6}}},
          {2, {{6, 7, 8, 15, 16, 17}, {}, {0, 1, 2, 9, 10, 11}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_non_local_cols{
        {true,
         {{1,
           {{0, 1, 3, 4, 2, 5},
            {0, 1, 2, 3, 4, 5},
            {0, 1, 2, 3, 4, 5},
            {0, 3, 1, 2, 4, 5}}},
          {2, {{0, 1, 2, 3, 4, 5}, {}, {0, 1, 2, 3, 4, 5}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {false,
         {{1,
           {{0, 1, 2, 3, 4, 5},
            {0, 1, 2, 3, 4, 5},
            {0, 1, 2, 3, 4, 5},
            {0, 1, 2, 3, 4, 5}}},
          {2, {{0, 1, 2, 3, 4, 5}, {}, {0, 1, 2, 3, 4, 5}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    /*
     * The mesh has the following structure
     *         local ids
     *        [24 25 26|33 34 35]
     *        [21 22 23|30 31 32]
     *        [18 19 20|27 28 29]
     *        ---------+---------
     *        [ 6  7  8|15 16 17]
     *        [ 3  4  5|12 13 14]
     *   0    [ 0  1  2| 9 10 11]  1
     *   */
    //  upper 12, lower 12, diag 9
    vec local_row_1_nf = {0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7, 1, 2, 3, 4, 4,
                          5, 5, 6, 7, 7, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    vec local_row_1_f = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
                         4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8};
    vec local_row_2_nf = {
        0,  0,  1,  1,  2,  3,  3,  4,  4,  5,  6,  7,  9, 9, 10, 10, 11, 12,
        12, 13, 13, 14, 15, 16, 1,  2,  3,  4,  4,  5,  5, 6, 7,  7,  8,  8,
        10, 11, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 0, 1, 2,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 2, 5, 8,  9,  12, 15};

    vec local_row_2_f = {
        0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,
        4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,
        9,  9,  9,  9,  10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13,
        13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17};
    vec local_row_4 = {
        0,  0,  1,  1,  2,  3,  3,  4,  4,  5,  6,  7,  9,  9,  10, 10, 11, 12,
        12, 13, 13, 14, 15, 16, 18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 24, 25,
        27, 27, 28, 28, 29, 30, 30, 31, 31, 32, 33, 34, 1,  2,  3,  4,  4,  5,
        5,  6,  7,  7,  8,  8,  10, 11, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17,
        19, 20, 21, 22, 22, 23, 23, 24, 25, 25, 26, 26, 28, 29, 30, 31, 31, 32,
        32, 33, 34, 34, 35, 35, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 2,  5,  8,  6,  7,  8,  9,  12, 15, 15, 16, 17,
        18, 19, 20, 20, 23, 26, 27, 28, 29, 27, 30, 33};
    vec local_row_4_f = {
        0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,
        4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,
        8,  8,  8,  9,  9,  9,  9,  10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12,
        12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16,
        16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20,
        20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24,
        24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28,
        28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32,
        32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35};
    std::map<bool, std::map<label, vec_vec>> exp_local_rows{
        {true,
         {{1, {local_row_1_f, local_row_1_f, local_row_1_f, local_row_1_f}},
          {2, {local_row_2_f, {}, local_row_2_f, {}}},
          {4, {local_row_4_f, {}, {}, {}}}}},
        {false,
         {{1, {local_row_1_nf, local_row_1_nf, local_row_1_nf, local_row_1_nf}},
          {2, {local_row_2_nf, {}, local_row_2_nf, {}}},
          {4, {local_row_4, {}, {}, {}}}}}};

    vec local_cols_1_nf = {1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 8, 0, 1, 0, 1, 3,
                           2, 4, 3, 4, 6, 5, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    vec local_cols_1_f = {0, 1, 3, 0, 1, 2, 4, 1, 2, 5, 0, 3, 4, 6, 1, 3, 4,
                          5, 7, 2, 4, 5, 8, 3, 6, 7, 4, 6, 7, 8, 5, 7, 8};
    vec local_cols_2 = {
        1,  3,  2,  4,  5,  4,  6,  5,  7,  8,  7,  8,  10, 12, 11, 13, 14, 13,
        15, 14, 16, 17, 16, 17, 0,  1,  0,  1,  3,  2,  4,  3,  4,  6,  5,  7,
        9,  10, 9,  10, 12, 11, 13, 12, 13, 15, 14, 16, 0,  1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 9,  12, 15, 2,  5,  8};
    vec local_cols_2_f = {
        0, 1,  3,  0,  1,  2,  4,  1,  2,  5,  9,               //
        0, 3,  4,  6,  1,  3,  4,  5,  7,  2,  4,  5,  8,  12,  //
        3, 6,  7,  4,  6,  7,  8,  5,  7,  8,  15,              //
        2, 9,  10, 12, 9,  10, 11, 13, 10, 11, 14,              //
        5, 9,  12, 13, 15, 10, 12, 13, 14, 16, 11, 13, 14, 17,  //
        8, 12, 15, 16, 13, 15, 16, 17, 14, 16, 17};
    vec local_cols_4 = {
        1,  3,  2,  4,  5,  4,  6,  5,  7,  8,  7,  8,  10, 12, 11, 13, 14, 13,
        15, 14, 16, 17, 16, 17, 19, 21, 20, 22, 23, 22, 24, 23, 25, 26, 25, 26,
        28, 30, 29, 31, 32, 31, 33, 32, 34, 35, 34, 35, 0,  1,  0,  1,  3,  2,
        4,  3,  4,  6,  5,  7,  9,  10, 9,  10, 12, 11, 13, 12, 13, 15, 14, 16,
        18, 19, 18, 19, 21, 20, 22, 21, 22, 24, 23, 25, 27, 28, 27, 28, 30, 29,
        31, 30, 31, 33, 32, 34, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 9,  12, 15, 18, 19, 20, 2,  5,  8,  27, 28, 29,
        6,  7,  8,  27, 30, 33, 15, 16, 17, 20, 23, 26};
    vec local_cols_4_f = {
        0,  1,  3,  0,  1,  2,  4,  1,  2,  5,  9,               // 0 - 10
        0,  3,  4,  6,  1,  3,  4,  5,  7,  2,  4,  5,  8,  12,  // 11 - 24
        3,  6,  7,  18, 4,  6,  7,  8,  19, 5,  7,  8,  15, 20,  // 12 - 38
        2,  9,  10, 12, 9,  10, 11, 13, 10, 11, 14,              //
        5,  9,  12, 13, 15, 10, 12, 13, 14, 16, 11, 13, 14, 17,  //
        8,  12, 15, 16, 27, 13, 15, 16, 17, 28, 14, 16, 17, 29,  //
        6,  18, 19, 21, 7,  18, 19, 20, 22, 8,  19, 20, 23, 27,  //
        18, 21, 22, 24, 19, 21, 22, 23, 25, 20, 22, 23, 26, 30,  //
        21, 24, 25, 22, 24, 25, 26, 23, 25, 26, 33,              //
        15, 20, 27, 28, 30, 16, 27, 28, 29, 31, 17, 28, 29, 32,  //
        23, 27, 30, 31, 33, 28, 30, 31, 32, 34, 29, 31, 32, 35,  //
        26, 30, 33, 34, 31, 33, 34, 35, 32, 34, 35};
    std::map<bool, std::map<label, vec_vec>> exp_local_cols{
        {true,
         {{1, {local_cols_1_f, local_cols_1_f, local_cols_1_f, local_cols_1_f}},
          {2, {local_cols_2_f, {}, local_cols_2_f, {}}},
          {4, {local_cols_4_f, {}, {}, {}}}}},
        {false,
         {{1,
           {local_cols_1_nf, local_cols_1_nf, local_cols_1_nf,
            local_cols_1_nf}},
          {2, {local_cols_2, {}, local_cols_2, {}}},
          {4, {local_cols_4, {}, {}, {}}}}}};

    vec_vec_s x_1{{5, 7, 7, 7, 9, 10, 7, 10, 13},
                  {6, 7, 6, 9, 9, 8, 10, 10, 10},
                  {6, 9, 10, 7, 9, 10, 6, 8, 10},
                  {7, 9, 9, 9, 9, 8, 9, 8, 7}};
    std::vector<scalar> exp_x_2_1 = {5, 7, 7, 7, 9, 10, 7,  10, 13,
                                     6, 7, 6, 9, 9, 8,  10, 10, 10};
    std::vector<scalar> exp_x_2_2 = {6, 9, 10, 7, 9, 10, 6, 8, 10,
                                     7, 9, 9,  9, 9, 8,  9, 8, 7};
    std::vector<scalar> exp_x_4 = {5, 7, 7,  7,  9,  10, 7, 10, 13, 6, 7, 6,
                                   9, 9, 8,  10, 10, 10, 6, 9,  10, 7, 9, 10,
                                   6, 8, 10, 7,  9,  9,  9, 9,  8,  9, 8, 7};
    std::map<bool, std::map<label, vec_vec_s>> exp_x{
        {true,
         {{1, x_1},
          {2, {exp_x_2_1, {}, exp_x_2_2, {}}},
          {4, {exp_x_4, {}, {}, {}}}}},
        {false,
         {{1, x_1},
          {2, {exp_x_2_1, {}, exp_x_2_2, {}}},
          {4, {exp_x_4, {}, {}, {}}}}}};
};


INSTANTIATE_TEST_SUITE_P(DistMatL2DInit, DistMatL2D,
                         testing::Combine(testing::Values(1, 2, 4),
                                          testing::Values("Coo"),
                                          testing::Values(false, true)),
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

    gko::dim<2> global_vec_dim{repartitioner->get_orig_partition()->get_size(),
                               1};
    gko::dim<2> local_vec_dim{repartitioner->get_repart_dim()[0], 1};

    auto distributed = create_distributed(exec, repartitioner, hostMatrix,
                                          matrix_format, fused);

    ASSERT_EQ(distributed->get_local_matrix()->get_size()[0],
              exp_local_size[ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_local_matrix()->get_size()[1],
              exp_local_size[ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_non_local_matrix()->get_size()[0],
              exp_local_size[ranks_per_gpu][rank]);
    ASSERT_EQ(distributed->get_local_matrix()->get_size()[0],
              exp_local_size[ranks_per_gpu][rank]);
}

TEST_P(DistMatL2D, hasCorrectLocalMatrix)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto [ranks_per_gpu, matrix_format, fused] = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);

    gko::dim<2> global_vec_dim{repartitioner->get_orig_partition()->get_size(),
                               1};
    gko::dim<2> local_vec_dim{repartitioner->get_repart_dim()[0], 1};

    auto distributed = create_distributed(exec, repartitioner, hostMatrix,
                                          matrix_format, fused);

    auto local =
        (fused) ? gko::as<gko::matrix::Coo<scalar, label>>(
                      distributed->get_local_matrix())
                : detail::convert_combination_to_coo(
                      exec.get_ref_exec(), distributed->get_local_matrix());

    ASSERT_EQ(distributed->get_local_matrix()->get_size()[1],
              exp_local_size[ranks_per_gpu][rank]);

    auto res_local_coeffs = convert_to_vector(get_val(local));
    auto res_local_cols = convert_to_vector(get_col(local));
    auto res_local_rows = convert_to_vector(get_row(local));

    EXPECT_EQ(res_local_rows.size(),
              exp_local_rows[fused][ranks_per_gpu][rank].size());


    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_rows[i],
                  exp_local_rows[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i << " on rank " << rank;
    }

    EXPECT_EQ(res_local_cols.size(),
              exp_local_cols[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_cols[i],
                  exp_local_cols[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i << " on rank " << rank;
    }

    EXPECT_EQ(res_local_coeffs.size(),
              exp_local_coeffs[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_coeffs[i],
                  exp_local_coeffs[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i << " on rank " << rank;
    }
}


TEST_P(DistMatL2D, hasCorrectNonLocalMatrix)
{
    /* The test mesh is 6x6 grid decomposed into 4 3x3 subdomains */
    auto [ranks_per_gpu, matrix_format, fused] = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);

    std::map<label, vec> exp_non_local_size;
    exp_non_local_size.emplace(1, vec{6, 6, 6, 6});
    exp_non_local_size.emplace(2, vec{6, 0, 6, 0});
    exp_non_local_size.emplace(4, vec{0, 0, 0, 0});

    auto distributed = create_distributed(exec, repartitioner, hostMatrix,
                                          matrix_format, fused);

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
              exp_non_local_coeffs[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_rows,
              exp_non_local_rows[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_cols,
              exp_non_local_cols[fused][ranks_per_gpu][rank]);
}

TEST_P(DistMatL2D, canApplyCorrectly)
{
    auto [ranks_per_gpu, format, fused] = GetParam();
    auto mesh = ((Environment *)global_env)->mesh;
    auto hostMatrix = ((Environment *)global_env)->hostMatrix;
    auto repartitioner = std::make_shared<Repartitioner>(
        hostMatrix->get_local_nrows(), ranks_per_gpu, 0, exec);

    auto distributed =
        create_distributed(exec, repartitioner, hostMatrix, format, fused);

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

    std::cout << " res_x " << res_x << "\n";

    ASSERT_EQ(res_x, exp_x[fused][ranks_per_gpu][rank]);
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
