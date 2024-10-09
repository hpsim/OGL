// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Repartitioner.H"

/* @brief Test fixture class for 2D mesh
 *
 * The mesh has the following structure
 *         local ids          |   global ids
 * cells: [ 2 3 | 2 3 ]       |  [ 10 11 | 14 15 ]
 *   2    [ 0 1 | 0 1 ]  3    |  [  8  9 | 12 13 ]
 *        ------+------       |  --------+--------
 *        [ 2 3 | 2 3 ]       |  [  2  3 |  6  7 ]
 *   0    [ 0 1 | 0 1 ]  1    |  [  0  1 |  4  5 ]
 */
class RepartitionerFixture2D
    : public RepartitionerFixture,
      public testing::WithParamInterface<std::tuple<bool, int>> {
public:
    label local_size = 4;

    vec_vec idxs{
        {0, 2, 0, 1},  // rank 0
        {1, 3, 0, 1},  // rank 1
        {2, 3, 0, 2},  // rank 2
        {2, 3, 1, 3}   // rank 3
    };

    std::vector<label> rows{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    std::vector<label> cols{0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
    std::vector<label> mapping{8, 0, 1, 4, 9, 2, 5, 10, 3, 6, 7, 11};
    std::vector<gko::span> spans{gko::span{0, 13}};

    // Setup data
    // non local data
    vec_vec non_local_rows{
        {1, 3, 2, 3}, {0, 2, 2, 3}, {0, 1, 1, 3}, {0, 1, 0, 2}};
    // non local columns in global idxs
    vec_vec non_local_cols{
        {4, 6, 8, 9}, {1, 3, 12, 13}, {2, 3, 12, 14}, {6, 7, 9, 11}};
    vec_vec non_local_mapping{
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}};
    // communication partners (ranks)
    vec_vec comm_target_ids{{1, 2}, {0, 3}, {0, 3}, {1, 2}};

    std::vector<std::vector<gko::span>> non_local_spans{
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 0
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 1
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 2
        {gko::span{0, 2}, gko::span{2, 4}},  // rank 3
    };

    vec_vec non_local_ranks{
        {1},     // rank 0
        {0, 2},  // rank 1
        {1, 3},  // rank 2
        {2},     // rank 3
    };

    // expected values
    std::map<label, vec> exp_local_nnz{
        {1, {12, 12, 12, 12}}, {2, {28, 0, 28, 0}}, {4, {64, 0, 0, 0}}};
    std::map<label, vec> exp_local_dim{
        {1, {4, 4, 4, 4}}, {2, {8, 0, 8, 0}}, {4, {16, 0, 0, 0}}};

    std::map<label, std::vector<label>> exp_non_local_nnz{
        {1, {4, 4, 4, 4}}, {2, {4, 0, 4, 0}}, {4, {0, 0, 0, 0}}};

    // expected local row indices in local indices
    // first map fused true/false
    // local row indices
    /*
     * The mesh has the following structure
     *         local ids          |   global ids
     * cells: [ 2 3 | 6 7 ]       |  [ 10 11 | 14 15 ]
     *   2    [ 0 1 | 4 5 ]  3    |  [  8  9 | 12 13 ]
     *        ------+------       |  --------+--------
     *        [ 2 3 | 6 7 ]       |  [  2  3 |  6  7 ]
     *   0    [ 0 1 | 4 5 ]  1    |  [  0  1 |  4  5 ]
     *   */
    vec nf_rows_2 = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
                     4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 4, 6, 1, 3};
    vec nf_cols_2 = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3, 4, 5,
                     6, 4, 5, 7, 4, 6, 7, 5, 6, 7, 1, 3, 4, 6};
    // [upper [0-3|12-15], lower [4,7|16-19], diag [8, 11|20-23], interfaces
    // [24-27]]
    vec nf_map_2 = {8,  0,  1,  4,  9,  2,
                    5,  10, 3,  6,  7,  11,  // here first sd is done
                    20, 12, 13, 16, 21, 14,
                    17, 22, 15, 18, 19, 23,  // here interfaces start
                    24, 25, 26, 27};
    vec f_rows_2 = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3,
                    4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7};
    vec f_cols_2 = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 1, 2, 3, 6,
                    1, 4, 5, 6, 4, 5, 7, 3, 4, 6, 7, 5, 6, 7};

    // [ rank 0            | rank 1             ] [ rank 0    | rank 1 ]
    // [upper, lower, diag | upper, lower,  diag] [interfaces | interfaces]
    // [0-3,   4-7,   8-11 | 12-15, 16-19, 20-23] [24-25,     | 26-27]
    // interfaces start at 24
    vec f_map_2 = {8,  0,  1,  4,  9,  2,  26, 5,  10, 3,  6,  7,  11, 27,
                   24, 20, 12, 13, 16, 21, 14, 25, 17, 22, 15, 18, 19, 23};
    /*
     * The mesh has the following structure
     *         local ids          |   global ids
     * cells: [10 11|14 15]       |  [ 10 11 | 14 15 ]
     *   2    [ 8 9 |12 13]  3    |  [  8  9 | 12 13 ]
     *        ------+------       |  --------+--------
     *        [ 2 3 | 6 7 ]       |  [  2  3 |  6  7 ]
     *   0    [ 0 1 | 4 5 ]  1    |  [  0  1 |  4  5 ]
     *   */
    // NOTE the interfaces are in order of the comm_target indices
    // ie [0, 0, 0 (1), (1), 2 (2), 2 (3), 2 (2), 2 (3)]
    vec nf_rows_4 = {0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,
                     4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  8,  8,
                     8,  9,  9,  9,  10, 10, 10, 11, 11, 11, 12, 12, 12,
                     13, 13, 13, 14, 14, 14, 15, 15, 15, 4,  6,  8,  9,
                     1,  3,  12, 13, 2,  3,  12, 14, 6,  7,  9,  11};
    vec nf_cols_4 = {0,  1,  2,  0,  1,  3,  0,  2,  3,  1,  2,  3,  4,
                     5,  6,  4,  5,  7,  4,  6,  7,  5,  6,  7,  8,  9,
                     10, 8,  9,  11, 8,  10, 11, 9,  10, 11, 12, 13, 14,
                     12, 13, 15, 12, 14, 15, 13, 14, 15, 1,  3,  2,  3,
                     4,  6,  6,  7,  8,  9,  9,  11, 12, 13, 12, 14};
    vec nf_map_4{8,  0,  1,  4,  9,  2,  5,  10, 3,  6,  7,  11,  // 1st sd done
                 20, 12, 13, 16, 21, 14, 17, 22, 15, 18, 19, 23,  // 2nd sd done
                 32, 24, 25, 28, 33, 26, 29, 34, 27, 30, 31, 35,  // 3rd sd done
                 44, 36, 37, 40, 45, 38, 41, 46, 39, 42, 43, 47,  // 4th sd done
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                 60, 61, 62, 63};
    // fused
    vec f_rows_4 = {
        0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  3,   //
        4,  4,  4,  4,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,   //
        8,  8,  8,  8,  9,  9,  9,  9,  9,  10, 10, 10, 11, 11, 11, 11,  //
        12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15   //
    };
    vec f_cols_4 = {0,  1,  2,  0,  1,  3,  4,  0,  2,  3,  8,  1,  2,
                    3,  6,  9,  1,  4,  5,  6,  4,  5,  7,  3,  4,  6,
                    7,  12, 5,  6,  7,  13, 2,  8,  9,  10, 3,  8,  9,
                    11, 12, 8,  10, 11, 9,  10, 11, 14, 6,  9,  12, 13,
                    14, 7,  12, 13, 15, 11, 12, 14, 15, 13, 14, 15};
    vec f_map_4{
        8,  0,  1,  4,  9,  2,  52, 5,  10, 3,  56, 6,  7,  11, 53, 57,  //
        48, 20, 12, 13, 16, 21, 14, 49, 17, 22, 15, 60, 18, 19, 23, 61,  //
        50, 32, 24, 25, 51, 28, 33, 26, 62, 29, 34, 27, 30, 31, 35, 63,  //
        54, 58, 44, 36, 37, 55, 40, 45, 38, 59, 41, 46, 39, 42, 43, 47};

    std::map<bool, std::map<label, vec_vec>> exp_local_rows{
        {false,
         {{1, {rows, rows, rows, rows}},
          {2, {nf_rows_2, {}, nf_rows_2, {}}},
          {4, {nf_rows_4, {}, {}, {}}}}},
        {true,
         {{1, {rows, rows, rows, rows}},
          {2, {f_rows_2, {}, f_rows_2, {}}},
          {4, {f_rows_4, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_local_cols{
        {false,
         {{1, {cols, cols, cols, cols}},
          {2, {nf_cols_2, {}, nf_cols_2, {}}},
          {4, {nf_cols_4, {}, {}, {}}}}},
        {true,
         {{1, {cols, cols, cols, cols}},
          {2, {f_cols_2, {}, f_cols_2, {}}},
          {4, {f_cols_4, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_local_mapping{
        {false,
         {{1, {mapping, mapping, mapping, mapping}},
          {2, {nf_map_2, {}, nf_map_2, {}}},
          {4, {nf_map_4, {}, {}, {}}}}},
        {true,
         {{1, {mapping, mapping, mapping, mapping}},
          {2, {f_map_2, {}, f_map_2, {}}},
          {4, {f_map_4, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_begin{
        {false,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0, 24, 26}, {}, {0, 24, 26}, {}}},
          {4, {{0, 48, 50, 52, 54, 56, 58, 60, 62}, {}, {}, {}}}}},
        {true,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{0}, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_end{
        {false,
         {{1, {{13}, {13}, {13}, {13}}},
          {2, {{24, 26, 28}, {}, {24, 26, 28}, {}}},
          {4, {{48, 50, 52, 54, 56, 58, 60, 62, 64}, {}, {}, {}}}}},
        {true,
         {{1, {{13}, {13}, {13}, {13}}},
          {2, {{28}, {}, {28}, {}}},
          {4, {{64}, {}, {}, {}}}}}};

    // non local data
    std::map<bool, std::map<label, vec_vec>> exp_non_local_rows{
        {false,
         {{1, non_local_rows},
          {2, {{2, 3, 6, 7}, {}, {0, 1, 4, 5}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, {{1, 2, 3, 3}, {0, 2, 2, 3}, {0, 1, 1, 3}, {0, 0, 1, 2}}},
          {2, {{2, 3, 6, 7}, {}, {0, 1, 4, 5}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    // non local cols are in global indices
    std::map<bool, std::map<label, vec_vec>> exp_non_local_cols{
        {false,
         {{1, {{4, 6, 8, 9}, {1, 3, 12, 13}, {2, 3, 12, 14}, {6, 7, 9, 11}}},
          {2, {{8, 9, 12, 13}, {}, {2, 3, 6, 7}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, {{4, 8, 6, 9}, {1, 3, 12, 13}, {2, 3, 12, 14}, {6, 9, 7, 11}}},
          {2, {{8, 9, 12, 13}, {}, {2, 3, 6, 7}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_non_local_mapping{
        {false,
         {{1, {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}},
          {2, {{0, 1, 2, 3}, {}, {0, 1, 2, 3}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, {{0, 2, 1, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}}},
          {2, {{0, 1, 2, 3}, {}, {0, 1, 2, 3}, {}}},
          {4, {{}, {}, {}, {}}}}}};
};

INSTANTIATE_TEST_SUITE_P(RepartitionerFixture2DInstantiation,
                         RepartitionerFixture2D,
                         testing::Combine(testing::Values(false, true),
                                          testing::Values(1, 2, 4)),
                         [](const auto &info) {
                             // Can use info.param here to generate the test
                             // suffix
                             std::vector<std::string> names;
                             names.emplace_back("fuse");
                             names.emplace_back("ranks");
                             std::string name = "fused_";
                             name += std::to_string(std::get<0>(info.param));
                             name += "_ranks_";
                             name += std::to_string(std::get<1>(info.param));
                             return name;
                         });

TEST_P(RepartitionerFixture2D, can_convert_to_global)
{
    // Arrange
    using namespace gko::experimental::distributed;
    auto ref_exec = exec.get_ref_exec();
    auto partition = gko::share(
        build_partition_from_local_size<label, label>(ref_exec, comm, 4));

    vec_vec exp_global_idxs{
        {4, 6, 8, 9},    // rank 0
        {1, 3, 12, 13},  // rank 1
        {2, 3, 12, 14},  // rank 2
        {6, 7, 9, 11}    // rank 3
    };

    // Act
    auto global_idxs =
        detail::convert_to_global(partition, idxs[rank].data(),
                                  non_local_spans[rank], comm_target_ids[rank]);

    // Assert
    EXPECT_EQ(global_idxs, exp_global_idxs[rank]);
}

TEST_P(RepartitionerFixture2D, can_repartition_2D_comm_pattern_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    // expected communication ranks
    std::map<label, vec_vec> exp_res_ids{};
    exp_res_ids[1] =
        comm_target_ids;  // in the ranks_per_gpu==1 case nothing changes
    // only communication partners are 0-2 and 2-0
    exp_res_ids.emplace(2, vec_vec{{2}, {}, {0}, {}});
    // no communication if all ranks are repartitioned to single owner
    exp_res_ids.emplace(4, vec_vec{{}, {}, {}, {}});

    // expected communication sizes
    std::map<label, vec_vec> exp_res_sizes;
    exp_res_sizes.emplace(1, vec_vec({{2, 2}, {2, 2}, {2, 2}, {2, 2}}));
    exp_res_sizes.emplace(2, vec_vec({{4}, {}, {4}, {}}));
    exp_res_sizes.emplace(4, vec_vec({{}, {}, {}, {}}));

    // expected rows
    vec_vec_vec rows{
        {{1, 3}, {2, 3}}, {{0, 2}, {2, 3}}, {{0, 1}, {1, 3}}, {{0, 1}, {0, 2}}};

    std::map<label, vec_vec> exp_res_rows;
    exp_res_rows.emplace(
        1, vec_vec{{1, 3, 2, 3}, {0, 2, 2, 3}, {0, 1, 1, 3}, {0, 1, 0, 2}});
    // after repartitioning with 2 ranks_per_gpu
    // cells: [ 2 3  6 7 ]
    //   2    [ 0 1  4 5 ]  3
    //        -------------
    //        [ 2 3  6 7 ]
    //   0    [ 0 1  4 5 ]  1
    exp_res_rows.emplace(2, vec_vec{{2, 3, 6, 7}, {}, {0, 1, 4, 5}, {}});
    // after repartitioning with 4 ranks_per_gpu [ 0 1 2 3 4 5 6 7 ] <-
    // local row ids
    exp_res_rows.emplace(4, vec_vec{{}, {}, {}, {}});

    std::map<label, vec_vec> exp_gather_idx;
    exp_gather_idx.emplace(
        1, vec_vec{{0, 2, 0, 1}, {1, 3, 0, 1}, {2, 3, 0, 2}, {2, 3, 1, 3}});
    exp_gather_idx.emplace(2, vec_vec{{0, 1, 4, 5}, {}, {2, 3, 6, 7}, {}});
    exp_gather_idx.emplace(4, vec_vec{{}, {}, {}, {}});

    // the original comm_pattern
    auto comm_pattern = std::make_shared<CommunicationPattern>(
        exec, comm_target_ids[rank], rows[rank]);

    // Act
    auto repart_comm_pattern =
        repartitioner.repartition_comm_pattern(exec, comm_pattern);

    // Assert
    auto res_ids = repart_comm_pattern->target_ids;

    EXPECT_EQ(res_ids, exp_res_ids[ranks_per_gpu][rank]);

    auto res_sizes = repart_comm_pattern->target_sizes;
    EXPECT_EQ(res_sizes, exp_res_sizes[ranks_per_gpu][rank]);

    auto total_rank_send_idx = repart_comm_pattern->total_rank_send_idx();
    auto res_rows = total_rank_send_idx;
    EXPECT_EQ(res_rows, exp_res_rows[ranks_per_gpu][rank]);

    auto total_recv_gather_idx =
        repart_comm_pattern->compute_recv_gather_idxs(exec);
    auto res_gather_idx = convert_to_vector(total_recv_gather_idx);
    EXPECT_EQ(res_gather_idx, exp_gather_idx[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture2D, can_repartition_sparsity_pattern)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{4, 4}, rows, cols, mapping, spans);

    auto non_local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{4, 4}, non_local_rows[rank], non_local_cols[rank],
        non_local_mapping[rank], non_local_spans[rank]);

    // Act
    auto [repart_local, repart_non_local, tracking] =
        repartitioner.repartition_sparsity(exec, local_sparsity,
                                           non_local_sparsity,
                                           comm_target_ids[rank], fused);
    // Assert
    // local properties
    ASSERT_EQ(repart_local->num_nnz, exp_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->dim[0], exp_local_dim[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->dim[1], exp_local_dim[ranks_per_gpu][rank]);

    auto res_local_rows = convert_to_vector(repart_local->row_idxs);
    auto res_local_cols = convert_to_vector(repart_local->col_idxs);
    auto res_local_mapping = convert_to_vector(repart_local->ldu_mapping);
    ASSERT_EQ(res_local_rows.size(),
              exp_local_rows[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_rows.size(); i++) {
        ASSERT_EQ(res_local_rows[i],
                  exp_local_rows[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i;
    }
    ASSERT_EQ(res_local_cols.size(),
              exp_local_cols[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_cols.size(); i++) {
        ASSERT_EQ(res_local_cols[i],
                  exp_local_cols[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i;
    }
    ASSERT_EQ(res_local_mapping.size(),
              exp_local_mapping[fused][ranks_per_gpu][rank].size());
    for (size_t i = 0; i < res_local_mapping.size(); i++) {
        ASSERT_EQ(res_local_mapping[i],
                  exp_local_mapping[fused][ranks_per_gpu][rank][i])
            << " failed at index " << i;
    }

    std::vector<label> res_local_spans_begin{};
    std::vector<label> res_local_spans_end{};
    for (auto [begin, end] : repart_local->spans) {
        res_local_spans_begin.push_back(begin);
        res_local_spans_end.push_back(end);
    }
    ASSERT_EQ(res_local_spans_begin,
              exp_local_spans_begin[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_spans_end,
              exp_local_spans_end[fused][ranks_per_gpu][rank]);

    // non local properties
    ASSERT_EQ(repart_non_local->num_nnz,
              exp_non_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->dim[0], exp_local_dim[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->dim[1], exp_non_local_nnz[ranks_per_gpu][rank]);

    auto res_non_local_rows = convert_to_vector(repart_non_local->row_idxs);
    auto res_non_local_cols = convert_to_vector(repart_non_local->col_idxs);
    ASSERT_EQ(res_non_local_rows,
              exp_non_local_rows[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_non_local_cols,
              exp_non_local_cols[fused][ranks_per_gpu][rank]);
    auto res_non_local_mapping =
        convert_to_vector(repart_non_local->ldu_mapping);
    ASSERT_EQ(res_non_local_mapping,
              exp_non_local_mapping[fused][ranks_per_gpu][rank]);
}
