// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Repartitioner.H"

/* @brief Test fixture class for 1D mesh
 *
 * The mesh has the following structure
 * ranks:    0     1     2     3
 * cells: [ 0 1 | 2 3 | 4 5 | 6 7 ] <- global row ids
 * cells: [ 0 1 | 0 1 | 0 1 | 0 1 ] <- local row ids
 */
class RepartitionerFixture1D
    : public RepartitionerFixture,
      public testing::WithParamInterface<std::tuple<bool, int>> {
public:
    label local_size = 2;  // matrix rows, same for all ranks

    // local data
    std::vector<label> rows{0, 0, 1, 1};
    std::vector<label> cols{0, 1, 0, 1};
    // mapping is [u, l, d]
    std::vector<label> mapping{2, 0, 1, 3};
    std::vector<gko::span> spans{gko::span{0, 5}};

    // Setup data
    // non local data
    vec_vec non_local_rows{{1}, {0, 1}, {0, 1}, {0}};
    // non local columns in global idxs
    vec_vec non_local_cols{{2}, {1, 4}, {3, 6}, {5}};
    vec_vec non_local_mapping{{0}, {0, 1}, {0, 1}, {0}};
    // communication partners (ranks)
    vec_vec ids{{1}, {0, 2}, {1, 3}, {2}};

    std::vector<std::vector<gko::span>> non_local_spans{
        {gko::span{0, 1}},                   // rank 0
        {gko::span{0, 1}, gko::span{1, 2}},  // rank 1
        {gko::span{0, 1}, gko::span{1, 2}},  // rank 2
        {gko::span{0, 1}},                   // rank 3
    };

    vec_vec non_local_ranks{
        {1},     // rank 0
        {0, 2},  // rank 1
        {1, 3},  // rank 2
        {2},     // rank 3
    };

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
    vec nf_rows_2 = {0, 0, 1, 1, 2, 2, 3, 3, 2, 1};
    vec nf_rows_4 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
                     5, 6, 6, 7, 7, 2, 1, 4, 3, 6, 5};
    // fused
    vec f_rows_2 = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    vec f_rows_4 = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                    4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7};
    std::map<bool, std::map<label, vec_vec>> exp_local_rows{
        {false,
         {{1, {rows, rows, rows, rows}},
          {2, {nf_rows_2, {}, nf_rows_2, {}}},
          {4, {nf_rows_4, {}, {}, {}}}}},
        {true,
         {{1, {rows, rows, rows, rows}},
          {2, {f_rows_2, {}, f_rows_2, {}}},
          {4, {f_rows_4, {}, {}, {}}}}}};

    vec nf_cols_2 = {0, 1, 0, 1, 2, 3, 2, 3, 1, 2};
    vec nf_cols_4 = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4,
                     5, 6, 7, 6, 7, 1, 2, 3, 4, 5, 6};
    // fused
    vec f_cols_2 = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    vec f_cols_4 = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4,
                    3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7};
    // first map fused true/false
    // expected local col indices in local indices
    std::map<bool, std::map<label, vec_vec>> exp_local_cols{
        {false,
         {{1, {cols, cols, cols, cols}},
          {2, {nf_cols_2, {}, nf_cols_2, {}}},
          {4, {nf_cols_4, {}, {}, {}}}}},
        {true,
         {{1, {cols, cols, cols, cols}},
          {2, {f_cols_2, {}, f_cols_2, {}}},
          {4, {f_cols_4, {}, {}, {}}}}}};

    vec nf_map_2{2, 0, 1, 3, 6, 4, 5, 7,  // here the interface values start
                 8, 9};                   // <- they are currently unused
    vec nf_map_4{2,  0,  1,  3,  6,  4,  5,  7,
                 10, 8,  9,  11, 14, 12, 13, 15,  // here the interface start
                 16, 17, 18, 19, 20, 21};         // <- unused values because we
                                                  // dont permute atm
    // [u l d | u l d ], [ i | i ]
    vec f_map_2{2, 0, 1, 3, 9, 8, 6, 4, 5, 7};
    vec f_map_4{2,  0,  1, 3, 17, 16, 6,  4,  5,  7,  19,
                18, 10, 8, 9, 11, 21, 20, 14, 12, 13, 15};
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
          {2, {{0, 8, 9}, {}, {0, 8, 9}, {}}},
          {4, {{0, 16, 17, 18, 19, 20, 21}, {}, {}, {}}}}},
        {true,
         {{1, {{0}, {0}, {0}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{0}, {}, {}, {}}}}}};

    // [start,end) indices of interface/submatrices
    std::map<bool, std::map<label, vec_vec>> exp_local_spans_end{
        {false,
         {{1, {{5}, {5}, {5}, {5}}},
          {2, {{8, 9, 10}, {}, {8, 9, 10}, {}}},
          {4, {{16, 17, 18, 19, 20, 21, 22}, {}, {}, {}}}}},
        {true,
         {{1, {{5}, {5}, {5}, {5}}},
          {2, {{10}, {}, {10}, {}}},
          {4, {{22}, {}, {}, {}}}}}};

    // number of rows of local matrix
    std::map<label, vec> exp_local_dim{
        {1, {local_size, local_size, local_size, local_size}},
        {2, {2 * local_size, 0, 2 * local_size, 0}},
        {4, {4 * local_size, 0, 0, 0}},
    };

    // non local data
    std::map<bool, std::map<label, vec_vec>> exp_non_local_rows{
        {false,
         {{1, non_local_rows}, {2, {{3}, {}, {0}, {}}}, {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_rows},
          {2, {{3}, {}, {0}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    // non local cols are in global indices
    std::map<bool, std::map<label, vec_vec>> exp_non_local_cols{
        {false,
         {{1, non_local_cols}, {2, {{4}, {}, {3}, {}}}, {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, non_local_cols},
          {2, {{4}, {}, {3}, {}}},
          {4, {{}, {}, {}, {}}}}}};

    std::map<bool, std::map<label, vec_vec>> exp_non_local_mapping{
        {false,
         {{1, {{0}, {0, 1}, {0, 1}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{}, {}, {}, {}}}}},
        {true,
         {{1, {{0}, {0, 1}, {0, 1}, {0}}},
          {2, {{0}, {}, {0}, {}}},
          {4, {{}, {}, {}, {}}}}}};
};


INSTANTIATE_TEST_SUITE_P(RepartitionerFixture1DInstantiation,
                         RepartitionerFixture1D,
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

TEST_P(RepartitionerFixture1D, can_create_repartitioner)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    // Act
    auto repartitioner = Repartitioner(10, ranks_per_gpu, 0, exec, fused);
    // Assert
    EXPECT_EQ(repartitioner.get_ranks_per_gpu(), ranks_per_gpu);
}

TEST_P(RepartitionerFixture1D, has_correct_properties_for_n_rank)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    // Act
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);

    // Assert
    EXPECT_EQ(repartitioner.is_owner(exec),
              (rank % ranks_per_gpu == 0) ? true : false);

    EXPECT_EQ(
        repartitioner.compute_repart_size(local_size, ranks_per_gpu, exec),
        (repartitioner.is_owner(exec) ? ranks_per_gpu * local_size : 0));
}

TEST_P(RepartitionerFixture1D, can_exchange_spans_and_ranks_for_n_ranks)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();

    std::vector<gko::span> spans{{0, 5}, {5, 10}};
    std::vector<label> ranks{rank + 1, rank + 2};

    std::map<label, vec_vec> exp_ranks;
    exp_ranks.emplace(1, vec_vec{{1, 2}, {2, 3}, {3, 4}, {4, 5}});
    exp_ranks.emplace(2, vec_vec{{1, 2, 2, 3}, {}, {3, 4, 4, 5}, {}});
    exp_ranks.emplace(4, vec_vec{{1, 2, 2, 3, 3, 4, 4, 5}, {}, {}, {}});

    std::map<label, vec_vec> exp_spans_begin;
    exp_spans_begin.emplace(1, vec_vec{{0, 5}, {0, 5}, {0, 5}, {0, 5}});
    exp_spans_begin.emplace(2, vec_vec{{0, 5, 10, 15}, {}, {0, 5, 10, 15}, {}});
    exp_spans_begin.emplace(
        4, vec_vec{{0, 5, 10, 15, 20, 25, 30, 35}, {}, {}, {}});
    std::map<label, vec_vec> exp_spans_end;
    exp_spans_end.emplace(1, vec_vec{{5, 10}, {5, 10}, {5, 10}, {5, 10}});
    exp_spans_end.emplace(2, vec_vec{{5, 10, 15, 20}, {}, {5, 10, 15, 20}, {}});
    exp_spans_end.emplace(4,
                          vec_vec{{5, 10, 15, 20, 25, 30, 35, 40}, {}, {}, {}});

    // Act
    auto [new_spans, origins, gathered_ranks] =
        detail::exchange_spans_ranks(exec, ranks_per_gpu, spans, ranks);

    std::vector<label> res_spans_begin{};
    std::vector<label> res_spans_end{};

    for (auto span : new_spans) {
        res_spans_begin.push_back(span.begin);
        res_spans_end.push_back(span.end);
    }

    // Assert
    ASSERT_EQ(res_spans_begin, exp_spans_begin[ranks_per_gpu][rank]);
    ASSERT_EQ(res_spans_end, exp_spans_end[ranks_per_gpu][rank]);
}

TEST_P(RepartitionerFixture1D, can_repartition_sparsity_pattern)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

    std::vector<label> ranks{rank};
    auto local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{2, 2}, rows, cols, mapping, spans);

    auto non_local_sparsity = std::make_shared<SparsityPattern>(
        ref_exec, gko::dim<2>{2, non_local_ranks[rank].size()},
        non_local_rows[rank], non_local_cols[rank], non_local_mapping[rank],
        non_local_spans[rank]);

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
    ASSERT_EQ(res_local_rows, exp_local_rows[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_cols, exp_local_cols[fused][ranks_per_gpu][rank]);
    ASSERT_EQ(res_local_mapping, exp_local_mapping[fused][ranks_per_gpu][rank]);

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

TEST_P(RepartitionerFixture1D, can_repartition_comm_pattern)
{
    // Arrange
    auto [fused, ranks_per_gpu] = GetParam();
    auto repartitioner =
        Repartitioner(local_size, ranks_per_gpu, 0, exec, fused);
    auto ref_exec = exec.get_ref_exec();

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
