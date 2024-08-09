// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later



  std::pair<SparsityPatternVector, std::vector<bool>>
 Repartitioner::   build_non_local_interfaces(const ExecutorHandler &exec_handler,
                               SparsityPatternVector &loc,
                               const SparsityPatternVector &non_loc) const
{

        LOG_1(verbose_, "start build non local interfaces")
        std::vector<label> rows, cols, ldu_mapping, ranks, begins, ends;
        std::vector<bool> is_local;
        label merged_ranks_size = non_loc.ranks.size();

        for (label i = 0; i < merged_ranks_size; i++) {
            // these are the begin ends before merging they need to be
            // offseted;
            auto begin = non_loc.begin[i];
            auto end = non_loc.end[i];
            bool local = reparts_to_local(exec_handler, non_loc.ranks[i]);
            is_local.push_back(local);
            if (local) {
                // TODO depending on the interface simple non
                // transforming and interpolating interfaces like now
                // local processor interfaces could be merged
                loc.begin.push_back(loc.rows.size());
                loc.rows.insert(loc.rows.end(), non_loc.rows.begin() + begin,
                                non_loc.rows.begin() + end);
                loc.cols.insert(loc.cols.end(), non_loc.cols.begin() + begin,
                                non_loc.cols.begin() + end);
                loc.mapping.insert(loc.mapping.end(),
                                   non_loc.mapping.data() + begin,
                                   non_loc.mapping.data() + end);
                // TODO store from rank ie from which rank it came
                // this is currently unused
                loc.end.push_back(loc.rows.size());
                loc.ranks.push_back(non_loc.ranks[i]);
            } else {
                begins.push_back(rows.size());
                rows.insert(rows.end(), non_loc.rows.begin() + begin,
                            non_loc.rows.begin() + end);
                cols.insert(cols.end(), non_loc.cols.begin() + begin,
                            non_loc.cols.begin() + end);
                // NOTE we don't do anything with non local mapping
                ldu_mapping.insert(ldu_mapping.end(),
                                   non_loc.cols.begin() + begin,
                                   non_loc.cols.begin() + end);
                ends.push_back(rows.size());
                ranks.push_back(get_owner_rank(non_loc.ranks[i]));
            }
        }
        LOG_1(verbose_, "done build non local interfaces")
        return std::make_pair(
            SparsityPatternVector{rows, cols, ldu_mapping, begins, ends, ranks},
            is_local);


}
