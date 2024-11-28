// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Repartitioner.H"

/* @brief Test fixture class for 1D mesh
 *
 */
class RepartitionerFixture1D : public RepartitionerFixture,
                               public testing::WithParamInterface<int> {
public:
    label local_size = 2;  // matrix rows, same for all ranks

    // local data
    vec_vec rows{{0}, {1}, {0, 1}};
    vec_vec cols{{1}, {0}, {0, 1}};
    // mapping is [u, l, d]
    vec_vec mapping{{0}, {0}, {0, 1}};
    std::vector<gko::span> spans{gko::span{0, 5}};

    // Setup data
    // non local data
    // non local columns in global idxs
    vec_vec non_local_mapping{{0}, {0, 1}, {0, 1}, {0}};
    // communication partners (ranks)
    vec_vec ids{{1}, {0, 2}, {1, 3}, {2}};

    // Expected results
    // number of non-zeros of each sparsity pattern
    std::map<label, std::vector<label>> exp_local_nnz{
        {1, {4, 4, 4, 4}},
        {2, {10, 0, 10, 0}},
        {4, {22, 0, 0, 0}},
    };

    std::map<label, std::vector<label>> exp_non_local_nnz{
        {1, {1, 2, 2, 1}},
        {2, {1, 0, 1, 0}},
        {4, {0, 0, 0, 0}},
    };

    std::vector<std::vector<label>> comm_target_ids{{1}, {0, 2}, {1, 3}, {2}};
    // expected local row indices in local indices
    // first map fused true/false
    // local row indices
    // repartitioned 2 [ 0 1 , 2 3 | 0 1 , 2 3 ] | new  boundary , old boundary
    // repartitioned 4 [ 0 1 , 2 3 , 4 5 , 6 7 ] | new  boundary , old boundary
    // the last two elements are now local interfaces, they are in in order
    // of target_ids
    vec_vec rows_2 = {{0}, {1}, {0, 1}, {2}, {3}, {2, 3}, {1}, {2}};
    vec_vec rows_4 = {{0}, {1}, {0, 1}, {2}, {3}, {2, 3}, {4}, {5}, {4, 5},
                      {6}, {7}, {6, 7}, {1}, {2}, {3},    {4}, {5}, {6}};
    std::map<label, vec_vec_vec> exp_local_rows{{1, {rows, rows, rows, rows}},
                                                {2, {rows_2, {}, rows_2, {}}},
                                                {4, {rows_4, {}, {}, {}}}};

    vec_vec cols_2 = {{1}, {0}, {0, 1}, {3}, {2}, {2, 3}, {2}, {1}};
    vec_vec cols_4 = {{1}, {0}, {0, 1}, {3}, {2}, {2, 3}, {5}, {4}, {4, 5},
                      {7}, {6}, {6, 7}, {2}, {1}, {4},    {3}, {6}, {5}};
    // first map fused true/false
    // expected local col indices in local indices
    std::map<label, vec_vec_vec> exp_local_cols{{{1, {cols, cols, cols, cols}},
                                                 {2, {cols_2, {}, cols_2, {}}},
                                                 {4, {cols_4, {}, {}, {}}}}};

    // [u l d | u l d ], [ i | i ]
    vec_vec map_2{{0}, {0}, {0, 1}, {0}, {0}, {0, 1}, {0}, {0}};
    vec_vec map_4{{0}, {0}, {0, 1}, {0}, {0}, {0, 1}, {0}, {0}, {0, 1},
                  {0}, {0}, {0, 1}, {0}, {0}, {0},    {0}, {0}, {0}};
    std::map<label, vec_vec_vec> exp_local_mapping{
        {1, {mapping, mapping, mapping, mapping}},
        {2, {map_2, {}, map_2, {}}},
        {4, {map_4, {}, {}, {}}}};

    // non local data
    std::map<label, vec_vec_vec> exp_non_local_rows{
        {1, {{{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}}},
        {2, {{{3}}, {}, {{0}}, {}}},
        {4, {{}, {}, {}, {}}}};

    // non local cols are in global indices
    std::map<label, vec_vec_vec> exp_non_local_cols{
        {1, {{{2}}, {{1}, {4}}, {{3}, {6}}, {{5}}}},
        {2, {{{4}}, {}, {{3}}, {}}},
        {4, {{}, {}, {}, {}}}};

    std::map<label, vec_vec_vec> exp_non_local_map{
        {1, {{{0}}, {{0}, {0}}, {{0}, {0}}, {{0}}}},
        {2, {{{0}}, {}, {{0}}, {}}},
        {4, {{}, {}, {}, {}}}};
};


INSTANTIATE_TEST_SUITE_P(RepartitionerFixture1DInstantiation,
                         RepartitionerFixture1D, testing::Values(1, 2, 4),
                         [](const auto &info) {
                             // Can use info.param here to generate the test
                             // suffix
                             std::vector<std::string> names;
                             names.emplace_back("ranks");
                             std::string name = "ranks_";
                             name += std::to_string(info.param);
                             return name;
                         });

TEST_P(RepartitionerFixture1D, can_create_repartitioner)
{
    // Arrange
    auto ranks_per_gpu = GetParam();
    // Act
    auto repartitioner = Repartitioner(10, ranks_per_gpu, 0, exec);
    // Assert
    EXPECT_EQ(repartitioner.get_ranks_per_gpu(), ranks_per_gpu);
}

TEST_P(RepartitionerFixture1D, has_correct_properties_for_n_rank)
{
    // Arrange
    auto ranks_per_gpu = GetParam();
    // Act
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, exec);

    // Assert
    EXPECT_EQ(repartitioner.is_owner(exec),
              (rank % ranks_per_gpu == 0) ? true : false);

    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, exec),
        (repartitioner.is_owner(exec) ? ranks_per_gpu * local_size : 0));
}


TEST_P(RepartitionerFixture1D, can_repartition_sparsity_pattern)
{
    // Arrange
    auto ranks_per_gpu = GetParam();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, exec);

    // std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>();
    // upper
    local_sparsity->insert_interface({0}, {1}, rank, rank);
    // lower
    local_sparsity->insert_interface({1}, {0}, rank, rank);
    // diag
    local_sparsity->insert_interface({0, 1}, {0, 1}, rank, rank);

    auto non_local_sparsity = std::make_shared<SparsityPattern>();

    if (rank == 0) {
        non_local_sparsity->insert_interface({1}, {2}, 0, 1);
    }
    if (rank == 1) {
        non_local_sparsity->insert_interface({0}, {1}, 1, 0);
        non_local_sparsity->insert_interface({1}, {4}, 1, 2);
    }
    if (rank == 2) {
        non_local_sparsity->insert_interface({0}, {3}, 2, 1);
        non_local_sparsity->insert_interface({1}, {6}, 2, 3);
    }
    if (rank == 3) {
        non_local_sparsity->insert_interface({0}, {5}, 3, 2);
    }

    // Act
    auto [repart_local, repart_non_local] = repartitioner.repartition_sparsity(
        exec, local_sparsity, non_local_sparsity);
    // Assert
    // local properties
    ASSERT_EQ(repart_local->get_nnz(), exp_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->get_rows(), exp_local_rows[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->get_cols(), exp_local_cols[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_local->get_map(), exp_local_mapping[ranks_per_gpu][rank]);

    // non local properties
    ASSERT_EQ(repart_non_local->get_nnz(),
              exp_non_local_nnz[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->get_rows(),
              exp_non_local_rows[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->get_cols(),
              exp_non_local_cols[ranks_per_gpu][rank]);
    ASSERT_EQ(repart_non_local->get_map(),
              exp_non_local_map[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture1D, can_repartition_comm_pattern)
{
    // Arrange
    auto ranks_per_gpu = GetParam();
    auto repartitioner = Repartitioner(local_size, ranks_per_gpu, 0, exec);

    // expected communication ranks
    std::map<label, vec_vec> exp_res_ids{};

    exp_res_ids[1] = ids;  // in the ranks_per_gpu==1 case nothing changes
    // only communication partners are 0-2 and 2-0
    exp_res_ids.emplace(2, vec_vec{{2}, {}, {0}, {}});
    // no communication if all ranks are repartitioned to single owner
    exp_res_ids.emplace(4, vec_vec{{}, {}, {}, {}});

    // expected communication sizes
    std::map<label, vec_vec> exp_res_sizes;
    exp_res_sizes.emplace(1, vec_vec({{1}, {1, 1}, {1, 1}, {1}}));
    exp_res_sizes.emplace(2, vec_vec({{1}, {}, {1}, {}}));
    exp_res_sizes.emplace(4, vec_vec({{}, {}, {}, {}}));

    // expected rows
    vec_vec_vec rows{{{1}}, {{0}, {1}}, {{0}, {1}}, {{0}}};
    std::map<label, vec_vec> exp_res_rows;
    exp_res_rows.emplace(1, vec_vec{{1}, {0, 1}, {0, 1}, {0}});
    // after repartitioning with 2 ranks_per_gpu [ 0 1 2 3 | 0 1 2 3 ] <-
    // local row ids
    exp_res_rows.emplace(2, vec_vec{{3}, {}, {0}, {}});
    // after repartitioning with 4 ranks_per_gpu [ 0 1 2 3 4 5 6 7 ] <-
    // local row ids
    exp_res_rows.emplace(4, vec_vec{{}, {}, {}, {}});

    // the original comm_pattern
    auto comm_pattern =
        std::make_shared<CommunicationPattern>(exec, ids[rank], rows[rank]);

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
}
