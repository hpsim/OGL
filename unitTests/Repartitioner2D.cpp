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
class RepartitionerFixture2D : public RepartitionerFixture,
                               public testing::WithParamInterface<int> {
public:
    label local_size = 4;

    vec_vec idxs{
        {0, 2, 0, 1},  // rank 0
        {1, 3, 0, 1},  // rank 1
        {2, 3, 0, 2},  // rank 2
        {2, 3, 1, 3}   // rank 3
    };

    vec_vec rows{{0, 0, 1, 2}, {1, 2, 3, 3}, {0, 1, 2, 3}};
    vec_vec cols{{1, 2, 3, 3}, {0, 0, 1, 2}, {0, 1, 2, 3}};
    vec_vec mapping{{8, 0, 1, 4, 9, 2, 5, 10, 3, 6, 7, 11}};

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
    // [upper [0-3|12-15], lower [4,7|16-19], diag [8, 11|20-23], interfaces
    // [24-27]]
    vec_vec rows_2 = {{0, 0, 1, 2}, {1, 2, 3, 3}, {0, 1, 2, 3}, {4, 4, 5, 6},
                      {5, 6, 7, 7}, {4, 5, 6, 7}, {1, 3},       {4, 6}};
    vec_vec cols_2 = {{1, 2, 3, 3}, {0, 0, 1, 2}, {0, 1, 2, 3}, {5, 6, 7, 7},
                      {4, 4, 5, 6}, {4, 5, 6, 7}, {4, 6},       {1, 3}};

    // [ rank 0            | rank 1             ] [ rank 0    | rank 1 ]
    // [upper, lower, diag | upper, lower,  diag] [interfaces | interfaces]
    // [0-3,   4-7,   8-11 | 12-15, 16-19, 20-23] [24-25,     | 26-27]
    // interfaces start at 24
    vec_vec map_2 = {{}};
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
    // fused
    vec_vec rows_4 = {
        {0, 0, 1, 2},   {1, 2, 3, 3},     {0, 1, 2, 3},     {4, 4, 5, 6},
        {5, 6, 7, 7},   {4, 5, 6, 7},     {8, 8, 9, 10},    {9, 10, 11, 11},
        {8, 9, 10, 11}, {12, 12, 13, 14}, {13, 14, 15, 15}, {12, 13, 14, 15},
        {1, 3},         {2, 3},           {4, 6},           {6, 7},
        {8, 9},         {9, 11},          {12, 13},         {12, 14}};
    vec_vec cols_4 = {
        {1, 2, 3, 3},   {0, 0, 1, 2},     {0, 1, 2, 3},     {5, 6, 7, 7},
        {4, 4, 5, 6},   {4, 5, 6, 7},     {9, 10, 11, 11},  {8, 8, 9, 10},
        {8, 9, 10, 11}, {13, 14, 15, 15}, {12, 12, 13, 14}, {12, 13, 14, 15},
        {4, 6},         {8, 9},           {1, 3},           {12, 13},
        {2, 3},         {12, 14},         {6, 7},           {9, 11}};
    vec_vec map_4{{}};

    std::map<label, vec_vec_vec> exp_local_rows{{1, {rows, rows, rows, rows}},
                                                {2, {rows_2, {}, rows_2, {}}},
                                                {4, {rows_4, {}, {}, {}}}};

    std::map<label, vec_vec_vec> exp_local_cols{{1, {cols, cols, cols, cols}},
                                                {2, {cols_2, {}, cols_2, {}}},
                                                {4, {cols_4, {}, {}, {}}}};

    std::map<label, vec_vec_vec> exp_local_mapping{
        {1, {mapping, mapping, mapping, mapping}},
        {2, {map_2, {}, map_2, {}}},
        {4, {map_4, {}, {}, {}}}};

    // non local data
    std::map<label, vec_vec_vec> exp_non_local_rows{
        {1,
         {{{1, 3}, {2, 3}},
          {{0, 2}, {2, 3}},
          {{0, 1}, {1, 3}},
          {{0, 1}, {0, 2}}}},
        {2, {{{2, 3}, {6, 7}}, {}, {{0, 1}, {4, 5}}, {}}},
        {4, {{}, {}, {}, {}}}};

    // non local cols are in global indices
    std::map<label, vec_vec_vec> exp_non_local_cols{
        {1,
         {{{4, 6}, {8, 9}},
          {{1, 3}, {12, 13}},
          {{2, 3}, {12, 14}},
          {{6, 7}, {9, 11}}}},
        {2, {{{8, 9}, {12, 13}}, {}, {{2, 3}, {6, 7}}, {}}},
        {4, {{}, {}, {}, {}}}};

    std::map<label, vec_vec> exp_non_local_mapping{
        {1, {{0, 2, 1, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}}},
        {2, {{0, 1, 2, 3}, {}, {0, 1, 2, 3}, {}}},
        {4, {{}, {}, {}, {}}}};
};

INSTANTIATE_TEST_SUITE_P(RepartitionerFixture2DInstantiation,
                         RepartitionerFixture2D, testing::Values(1, 2, 4),
                         [](const auto &info) {
                             std::vector<std::string> names;
                             names.emplace_back("ranks");
                             std::string name = "ranks_";
                             name += std::to_string(info.param);
                             return name;
                         });


TEST_P(RepartitionerFixture2D, can_repartition_2D_comm_pattern_for_n_ranks)
{
    // Arrange
    auto ranks_per_gpu = GetParam();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, exec);
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
    auto ranks_per_gpu = GetParam();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, exec);
    auto ref_exec = exec.get_ref_exec();

    std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>();
    // upper
    local_sparsity->insert_interface({0, 0, 1, 2}, {1, 2, 3, 3}, rank, rank);
    // lower
    local_sparsity->insert_interface({1, 2, 3, 3}, {0, 0, 1, 2}, rank, rank);
    // diag
    local_sparsity->insert_interface({0, 1, 2, 3}, {0, 1, 2, 3}, rank, rank);

    auto non_local_sparsity = std::make_shared<SparsityPattern>();
    if (rank == 0) {
        non_local_sparsity->insert_interface({1, 3}, {4, 6}, 0, 1);
        non_local_sparsity->insert_interface({2, 3}, {8, 9}, 0, 2);
    }
    if (rank == 1) {
        non_local_sparsity->insert_interface({0, 2}, {1, 3}, 1, 0);
        non_local_sparsity->insert_interface({2, 3}, {12, 13}, 1, 3);
    }
    if (rank == 2) {
        non_local_sparsity->insert_interface({0, 1}, {2, 3}, 2, 0);
        non_local_sparsity->insert_interface({1, 3}, {12, 14}, 2, 3);
    }
    if (rank == 3) {
        non_local_sparsity->insert_interface({0, 1}, {6, 7}, 3, 1);
        non_local_sparsity->insert_interface({0, 2}, {9, 11}, 2, 2);
    }

    // Act
    auto [repart_local, repart_non_local] = repartitioner.repartition_sparsity(
        exec, local_sparsity, non_local_sparsity);

    // Assert
    // local properties
    ASSERT_EQ(repart_local->get_nnz(), exp_local_nnz[ranks_per_gpu][rank]);

    ASSERT_EQ(repart_local->get_rows(), exp_local_rows[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->get_cols(), exp_local_cols[ranks_per_gpu][rank]);
    //         ASSERT_EQ(repart_local->get_map(),
    //                   exp_local_mapping[fused][ranks_per_gpu][rank]);
    //
    //     // non local properties
    //     ASSERT_EQ(repart_non_local->get_nnz(),
    //               exp_non_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->get_rows(),
              exp_non_local_rows[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->get_cols(),
              exp_non_local_cols[ranks_per_gpu][rank]);
    //     ASSERT_EQ(repart_non_local->get_map(),
    //               exp_non_local_mapping[fused][ranks_per_gpu][rank]);
}
