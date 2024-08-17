// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/Repartitioner.H"

std::pair<SparsityPattern, std::vector<bool>>
Repartitioner::build_non_local_interfaces(const ExecutorHandler &exec_handler,
                                          SparsityPattern &loc,
                                          const SparsityPattern &non_loc) const
{
    LOG_1(verbose_, "start build non local interfaces")
    std::vector<label> rows, cols, ldu_mapping, ranks, begins, ends;
    std::vector<bool> is_local;
    label merged_ranks_size = non_loc.rank.size();

    // TODO dont modify SparsityPattern but create vector
    // and use the vector variant
    //   for (label i = 0; i < merged_ranks_size; i++) {
    //       // these are the begin ends before merging they need to be
    //       // offseted;
    //       auto begin = non_loc.spans[i].begin;
    //       auto end = non_loc.spans[i].end;
    //       bool local = reparts_to_local(exec_handler, non_loc.rank[i]);
    //       is_local.push_back(local);
    //       if (local) {
    //           // TODO depending on the interface simple non
    //           // transforming and interpolating interfaces like now
    //           // local processor interfaces could be merged
    //           loc.begin.push_back(loc.row_idxs.size());
    //           loc.row_idxs.insert(loc.row_idxs.end(),
    //           non_loc.row_idxs.get_data() + begin,
    //                           non_loc.row_idxs.get_data() + end);
    //           loc.col_idxs.insert(loc.cols.end(), non_loc.col_idxs.get_data()
    //           + begin,
    //                           non_loc.col_idxs.get_data() + end);
    //           loc.ldu_mapping.insert(loc.ldu_mapping.end(),
    //                              non_loc.ldu_mapping.get_data() + begin,
    //                              non_loc.ldu_mapping.get_data() + end);
    //           // TODO store from rank ie from which rank it came
    //           // this is currently unused
    //           // TODO reimplement end doesnt exist
    //           // loc.end.push_back(loc.row_idxs.size());
    //           // also cannot pushback
    //           // loc.rank.push_back(non_loc.rank[i]);
    //       } else {
    //           begins.push_back(row_idxs.size());
    //           rows.insert(rows.end(), non_loc.row_idxs.begin() + begin,
    //                       non_loc.row_idxs.begin() + end);
    //           cols.insert(cols.end(), non_loc.col_idxs.begin() + begin,
    //                       non_loc.col_idxs.begin() + end);
    //           // NOTE we don't do anything with non local mapping
    //           ldu_mapping.insert(ldu_mapping.end(), non_loc.col_idxs.begin()
    //           + begin,
    //                              non_loc.col_idxs.begin() + end);
    //           ends.push_back(rows.size());
    //           ranks.push_back(get_owner_rank(non_loc.rank[i]));
    //       }
    //   }
    //   LOG_1(verbose_, "done build non local interfaces")
    //   return std::make_pair(
    //       SparsityPattern{rows, cols, ldu_mapping, begins, ends, ranks},
    //       is_local);
}
