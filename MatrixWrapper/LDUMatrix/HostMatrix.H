/*---------------------------------------------------------------------------*\
License
    This file is part of OGL.

    OGL is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OGL is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OGL.  If not, see <http://www.gnu.org/licenses/>.


Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    HostMatrix.H

\*---------------------------------------------------------------------------*/
#ifndef OGL_HostMatrix_INCLUDED_H
#define OGL_HostMatrix_INCLUDED_H
#include <map>
#include <numeric>

#include <ginkgo/ginkgo.hpp>

#include "fvCFD.H"
#include "processorLduInterface.H"

#include <vector>
#include "DevicePersistent/Array/Array.H"
#include "DevicePersistent/DeviceIdGuard/DeviceIdGuard.H"
#include "DevicePersistent/IOGlobalIndex/gkoGlobalIndex.H"
#include "MatrixWrapper/Combination/Combination.H"
#include "MatrixWrapper/CommunicationPattern/CommunicationPattern.H"
#include "MatrixWrapper/SparsityPattern/SparsityPattern.H"
#include "common/common.H"


namespace Foam {

/* @brief based on a comm_pattern consecutive ldu_mappings are computed
 *
 * After merging sparsity patterns during repartitioning the ldu mapping are
 * not consecutive ie they could look like [0 1 2 3 | 0 1 2 3 | 0 1], where
 * | is the former rank boundary based on the comm pattern we compute a new
 * mapping as  [ 0 1 2 3 | 4 5 6 7| 8 9 ] where the following offsets [ 0 |
 * 4 | 8 ] based on the recv_counts are used.
 *
 * */
void make_ldu_mapping_consecutive(
    std::tuple<std::vector<label>, std::vector<label>, std::vector<label>,
               std::vector<label>>
        comm_pattern,
    std::vector<label> &ldu_mapping, label rank, label ranks_per_gpu);

// TODO this should be persistent
class Repartitioner {
private:
    using partition = gko::experimental::distributed::Partition<label, label>;

    const label size_;  //! Size (n matrix rows ) before repartitioning

    const label repart_size_;  //! Size after repartitioning

    const label ranks_per_gpu_;

    const ExecutorHandler &exec_handler_;

    std::shared_ptr<const partition> orig_partition_;


public:
    Repartitioner(label size, label ranks_per_gpu,
                  const ExecutorHandler &exec_handler)
        : size_(size),
          repart_size_(Repartitioner::compute_repart_size(size, ranks_per_gpu,
                                                          exec_handler)),
          ranks_per_gpu_(ranks_per_gpu),
          exec_handler_(exec_handler),
          orig_partition_(gko::share(
              gko::experimental::distributed::build_partition_from_local_size<
                  label, label>(exec_handler_.get_device_exec(),
                                *exec_handler_.get_communicator().get(),
                                size))) {};

    /* returns the owner rank for a given rank */
    label get_owner_rank(label rank) const
    {
        return compute_owner_rank(rank, ranks_per_gpu_);
    };

    /* returns the owner rank for a given rank */
    label get_owner_rank() const { return get_owner_rank(get_rank()); };

    /* returns if current rank is an owner  */
    bool get_owner() const { return get_rank() == get_owner_rank(get_rank()); };

    /* shortcut to current rank */
    label get_rank() const
    {
        return exec_handler_.get_communicator().get()->rank();
    };

    label get_ranks_per_gpu() const { return ranks_per_gpu_; }

    static label compute_repart_size(label local_size, label ranks_per_gpu,
                                     const ExecutorHandler &exec_handler)
    {
        if (ranks_per_gpu == 1) {
            return local_size;
        }

        CommCounts comm_pattern =
            compute_send_recv_counts(exec_handler, ranks_per_gpu, 1);

        std::vector<label> data{local_size};

        auto local_sizes =
            gather_to_owner(exec_handler, comm_pattern, 1, data.data());

        return std::reduce(local_sizes.begin(), local_sizes.end());
    }

    label get_repart_size() const {return repart_size_;}

    std::shared_ptr<const partition> get_orig_partition() const
    {
        return orig_partition_;
    };

    std::pair<std::shared_ptr<SparsityPattern>,
              std::shared_ptr<SparsityPattern>>
    repartition_sparsity(
        std::shared_ptr<SparsityPattern> src_local_pattern,
        std::shared_ptr<SparsityPattern> src_non_local_pattern) const
    {
        // 1. obtain send recv sizes vector
        // here we can reuse code from repartition_comm_pattern
        //
        // 2. initialize send and recv buffer
        // if we keep interfaces separated row and col indices can be
        //
        // 3. ldu mapping needs to be update?
        // 4. dim needs to be updated
        // this is not implemented yet, but we can't fail here for
        // debugging reasons
        auto exec = exec_handler_.get_device_exec();
        auto comm = *exec_handler_.get_communicator().get();
        label rank = get_rank();
        label owner_rank = get_owner_rank();
        bool owner = get_owner();
        label ranks_per_gpu = ranks_per_gpu_;

        auto local_comm_pattern = compute_send_recv_counts(
            exec_handler_, ranks_per_gpu, src_local_pattern->size_);

        label offset = orig_partition_->get_range_bounds()[rank] -
                       orig_partition_->get_range_bounds()[owner_rank];

        auto &exec_handler = exec_handler_;
        auto gather_closure = [exec_handler](auto &comm_pattern, auto &data,
                                             label offset) {
            return gather_to_owner(exec_handler, comm_pattern, data.get_size(),
                                   data.get_data(), offset);
        };

        std::cout << __FILE__ << " rank " << rank << "before gathered_local\n";
        auto merged_local_rows = gather_closure(
            local_comm_pattern, src_local_pattern->row_idxs, offset);

        auto merged_local_cols = gather_closure(
            local_comm_pattern, src_local_pattern->col_idxs, offset);

        auto merged_ldu_mapping = gather_closure(
            local_comm_pattern, src_local_pattern->ldu_mapping, 0);

        std::cout << __FILE__ << " rank " << rank << " gathered_local\n";
        make_ldu_mapping_consecutive(local_comm_pattern, merged_ldu_mapping,
                                     rank, ranks_per_gpu);
        std::cout << __FILE__ << " rank " << rank << " done ldu mapping consecutive\n";

        label rows = (owner) ? merged_local_rows.back() + 1 : 0;
        gko::dim<2> merged_local_dim{rows, rows};

        std::vector<label> merged_begin{0};
        std::vector<label> merged_end{label(merged_local_rows.size())};
        std::vector<label> merged_ranks{rank};

        auto non_local_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu, src_non_local_pattern->size_);

        auto merged_non_local_rows = gather_closure(
            non_local_comm_pattern, src_non_local_pattern->row_idxs, offset);
        std::cout << __FILE__ << " rank " << rank << " gathered_non_local\n";

        // the non local cols are in local idx of other side
        // thus we need the new offset of the other side
        for (label i = 0; i < src_non_local_pattern->rank.size(); i++) {
            auto comm_rank = src_non_local_pattern->rank[i];
            auto [begin, end] = src_non_local_pattern->interface_spans[i];
            label local_offset =
                orig_partition_->get_range_bounds()[comm_rank] -
                orig_partition_->get_range_bounds()[owner_rank];
            auto *data = src_non_local_pattern->col_idxs.get_data() + begin;
            auto size = end - begin;
            std::transform(data, data + size, data,
                           [&](label idx) { return idx + local_offset; });
        }
        std::cout << __FILE__ << " rank " << rank << " done offset non_local\n";

        // we now dealed with the new offset thus we dont need an offset
        // now
        auto merged_non_local_cols = gather_closure(
            non_local_comm_pattern, src_non_local_pattern->col_idxs, 0);

        auto merged_non_local_ldu_mapping = gather_closure(
            non_local_comm_pattern, src_non_local_pattern->row_idxs, 0);

        auto span_comm_pattern = compute_send_recv_counts(
            exec_handler, ranks_per_gpu,
            src_non_local_pattern->interface_spans.size());

        std::cout << __FILE__ << " rank " << rank << " done merged non_local\n";

        std::vector<label> spans_begin;
        std::vector<label> spans_end;

        for (auto elem : src_non_local_pattern->interface_spans) {
            spans_begin.push_back(elem.begin);
            spans_end.push_back(elem.end);
        }

        auto merged_non_local_begin =
            gather_to_owner(exec_handler, span_comm_pattern, spans_begin.size(),
                            spans_begin.data());

        auto merged_non_local_end =
            gather_to_owner(exec_handler, span_comm_pattern, spans_end.size(),
                            spans_end.data());

        auto merged_non_local_ranks = gather_to_owner(
            exec_handler, span_comm_pattern, src_non_local_pattern->rank.size(),
            src_non_local_pattern->rank.data());

        std::cout << __FILE__ << " rank " << rank << " done merged non_local2\n";


        // fix the received merged_non_local_begin and ends
        // we received begins and ends of the interfaces from
        // the neighbour ranks. however the begin and ends are
        // computed base on the non merged interfaces. Now we
        // need to offset these values
        // before merge
        // rank 0 begin [ 0 10 ], end [10 20]
        // rank 1 begin [ 0 10 ], end [10 20]
        // merged begin [ 0 10 0 10 ], end [10 20 10 20]
        // after fix
        // merged begin [ 0 10 20 30 ], end [10 20 30 40]
        for (label i = 1; i < ranks_per_gpu; i++) {
            auto recv_count = std::get<1>(span_comm_pattern)[rank + i];
            auto recv_offset = std::get<3>(span_comm_pattern)[rank + i];
            auto range_offset = merged_non_local_end[recv_offset - 1];
            for (label j = 0; j < recv_count; j++) {
                merged_non_local_begin[recv_offset + j] += range_offset;
                merged_non_local_end[recv_offset + j] += range_offset;
            }
        }


        // TODO remove
        //     if (rank == 0) {
        // std::cout << __FILE__ << " on rank " << rank << "
        // merged_begin"
        //           << merged_begin
        //           << " span_recv_counts " << span_recv_counts
        //           << " span_recv_offsets " << span_recv_offsets
        //           << " merged_non_local_begins " <<
        //           merged_non_local_begin
        //           << " merged_non_local_end " << merged_non_local_end
        //           << " merged_end " << merged_end
        //           << " merged_non_local_rows " <<
        //           merged_non_local_rows
        //           << " merged_ranks " << merged_non_local_ranks <<
        //           "\n";
        // }

        // TODO shorten this
        std::vector<label> gathered_non_local_rows;
        std::vector<label> gathered_non_local_cols;
        std::vector<label> gathered_non_local_mapping;
        std::vector<label> gathered_ranks;
        std::vector<label> gathered_begin;
        std::vector<label> gathered_end;
        label merged_ranks_size = merged_non_local_ranks.size();
        // redistribute interfaces
        for (label i = 0; i < merged_ranks_size; i++) {
            auto merged_target_rank = merged_non_local_ranks[i];
            label to_rank = get_owner_rank(merged_target_rank);
            bool local = rank == to_rank;
            // these are the begin ends before merging they need to be
            // offseted;
            auto begin = merged_non_local_begin[i];
            auto end = merged_non_local_end[i];
            if (local) {
                // TODO depending on the interface simple non
                // transforming and interpolating interfaces like now
                // local processor interfaces could be merged
                label size_before = merged_local_rows.size();
                merged_local_rows.insert(merged_local_rows.end(),
                                         merged_non_local_rows.begin() + begin,
                                         merged_non_local_rows.begin() + end);
                merged_local_cols.insert(merged_local_cols.end(),
                                         merged_non_local_cols.begin() + begin,
                                         merged_non_local_cols.begin() + end);
                merged_ldu_mapping.insert(
                    merged_ldu_mapping.end(),
                    merged_non_local_ldu_mapping.data() + begin,
                    merged_non_local_ldu_mapping.data() + end);
                // TODO store from rank ie from which rank it came
                merged_ranks.push_back(to_rank);

                label size_after = merged_local_rows.size();
                merged_begin.push_back(size_before);
                merged_end.push_back(size_after);
                // TODO remove
                // if (rank == 0) {
                //     std::cout << __FILE__ << " on rank " << rank
                //             << " begin " << begin << " end " << end
                //             << " insert merged_local_rows " <<
                //             merged_local_rows
                //             << "\n";
                //     }
            } else {
                label size_before = gathered_non_local_rows.size();
                gathered_non_local_rows.insert(
                    gathered_non_local_rows.end(),
                    merged_non_local_rows.begin() + begin,
                    merged_non_local_rows.begin() + end);
                // FIXME
                // up until now non_local_cols are stil using
                // local_idx and we need to convert them to
                // consequtive compressed idxs
                gathered_non_local_cols.insert(
                    gathered_non_local_cols.end(),
                    merged_non_local_cols.begin() + begin,
                    merged_non_local_cols.begin() + end);
                // TODO we don't do anything with non local mapping
                gathered_non_local_mapping.insert(
                    gathered_non_local_mapping.end(),
                    merged_non_local_cols.begin() + begin,
                    merged_non_local_cols.begin() + end);
                label size_after = gathered_non_local_rows.size();
                // if (rank == 0) {
                // std::cout << __FILE__ << " on rank " << rank
                //         << " gathered_non_local_rows " <<
                //         gathered_non_local_rows
                //         << " begin " << begin << " end " << end
                //         << " non_local size_before " << size_before
                //         << " non_local size_after " << size_after
                //         << "\n";
                // }
                gathered_begin.push_back(size_before);
                gathered_end.push_back(size_after);
                gathered_ranks.push_back(to_rank);
            }
        }

        // for compressed cols we compute a mapping first
        std::map<label, label> compressed_col_idxs{};
        label unique_idx = 0;
        for (auto idx : gathered_non_local_cols) {
            auto search = compressed_col_idxs.find(idx);
            if (search == compressed_col_idxs.end()) {
                std::cout << " insert " << idx << " " << unique_idx << "\n";
                compressed_col_idxs[idx] = unique_idx;
                unique_idx += 1;
            };
        }

        for (int i = 0; i < gathered_non_local_cols.size(); i++) {
            label uncompressed = gathered_non_local_cols[i];
            label compressed = compressed_col_idxs[uncompressed];
            std::cout << " i " << i << " uncompressed " << uncompressed
                      << " compressed " << compressed << "\n";
            gathered_non_local_cols[i] = compressed;
        }

        // add interfaces
        // to add interfaces we go through the non_local_sparsity
        // pattern and check if interface is still non_local
        //
        // iterate interfaces and send to owner.
        // on owner decide to move to local or non_local
        // for sending to owner we can also use the all_to_all_v
        // approach we than have
        //
        // row [ 4 , 8,  12 | 16 , 17, 18 ] <- local row
        // col [ 1 , 2, 3 | 1, 2, 3 ] <- just the interface ctr
        // from repartition_comm_pattern we could get
        //
        // or we split local / non-local first
        // for this we could store ranges when computing
        // repartition_comm_pattern
        //


        if (owner) {
            auto new_local_spars_pattern = std::make_shared<SparsityPattern>(
                exec, merged_local_rows.size(), merged_local_dim,
                merged_local_rows, merged_local_cols, merged_ldu_mapping,
                merged_begin, merged_end, merged_ranks);

            // if (rank == 0) {
            // std::cout << __FILE__ << " on rank " << rank
            //           << merged_local_rows
            //           << " merged_begin " <<  merged_begin
            //           << " merged_end " <<  merged_end
            //           << "local dim ["
            //           << new_local_spars_pattern->dim[0] << " x "
            //           << new_local_spars_pattern->dim[1] << "] \n";
            //         }

            auto new_non_local_spars_pattern =
                std::make_shared<SparsityPattern>(
                    exec, gathered_non_local_rows.size(),
                    gko::dim<2>{merged_local_dim[0],
                                gathered_non_local_rows.size()},
                    gathered_non_local_rows, gathered_non_local_cols,
                    gathered_non_local_mapping, gathered_begin, gathered_end,
                    gathered_ranks);

            return std::make_pair<std::shared_ptr<SparsityPattern>,
                                  std::shared_ptr<SparsityPattern>>(
                std::move(new_local_spars_pattern),
                std::move(new_non_local_spars_pattern));
        } else {
            auto new_local_spars_pattern =
                std::make_shared<SparsityPattern>(exec, 0);

            auto new_non_local_spars_pattern =
                std::make_shared<SparsityPattern>(exec, 0);

            new_local_spars_pattern->dim = merged_local_dim;
            new_non_local_spars_pattern->dim = merged_local_dim;

            return std::make_pair<std::shared_ptr<SparsityPattern>,
                                  std::shared_ptr<SparsityPattern>>(
                std::move(new_local_spars_pattern),
                std::move(new_non_local_spars_pattern));
        }
    }
};


// Free functions
const lduInterfaceField *interface_getter(
    const lduInterfaceFieldPtrsList &interfaces, const label i);

/** Write contiguous row and col indices from OpenFOAM lower and upper indices
 ** For details on the lower triangular based indexing see
 ** https://openfoamwiki.net/index.php/OpenFOAM_guide/Matrices_in_OpenFOA
 ** Note that the order of the indices are depicted wrong on the wiki
 ** In general the upper triangular matrix is traversed in row major
 ** and the lower triangular matrix in column major order.
 **
 ** @param nrows number of rows
 ** @param upper_nnz number of non zeros in the upper triangular matrix
 ** @param is_symmetric whether matrix is symmetric, in the symmetric case
 ** the lower elements indices in the permute array are computed differently
 ** @param upper pointer to OFs rows array
 ** @param lower pointer to OFs cols array
 ** @param rows pointer to rows array
 ** @param cols pointer to columns array
 ** @param permute pointer to permuter array
 */
void init_local_sparsity(const label nrows, const label upper_nnz,
                         const bool is_symmetric, const label *upper,
                         const label *lower, label *rows, label *cols,
                         label *permute);

/** Perform update of the coefficients array in row major order from a symmetric
 *ldu matrix
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 *upper
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param out pointer to data of output matrix
 **/
void symmetric_update(const label total_nnz, const label upper_nnz,
                      const label *permute, const scalar scale,
                      const scalar *diag, const scalar *upper, scalar *out);


/** Perform update of the coefficients array in row major order from a
 *non-symmetric ldu matrix
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 *upper
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 **        upper_nnz + diag_nnz <= permute for lower entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param lower pointer to lower triangular entries of the original ldu matrix
 ** @param out pointer to data of output matrix
 **/
void non_symmetric_update(const label total_nnz, const label upper_nnz,
                          const label *permute, const scalar scale,
                          const scalar *diag, const scalar *upper,
                          const scalar *lower, scalar *dense);


/** Perform update of the coefficients array in row major order from a
 ** symmetric ldu matrix including local interface data.
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 ** upper
 **
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 **        upper_nnz + diag_nnz <= permute for interface entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param interface pointer to interface entries
 ** @param out pointer to data of output matrix
 **/
void symmetric_update_w_interface(const label total_nnz, const label diag_nnz,
                                  const label upper_nnz, const label *permute,
                                  const scalar scale, const scalar *diag,
                                  const scalar *upper, const scalar *interface,
                                  scalar *dense);


/** Perform update of the coefficients array in row major order from a
 ** non-symmetric ldu matrix including local interface data.
 **
 ** @note this assumes out to be on the host
 ** @note performs contiguous writes into out and random access of diag and
 ** upper
 ** @param total_nnz number of non zeros entries in the matrix
 ** @param upper_nnz number of non zeros entries in the upper triangular matrix
 ** @param permute mapping between the original position in the ldu input data
 **        and the position in the row major ordered output data
 **        ie. permute[i] maps out[i] to original position in diag or upper
 **        with
 **        upper_nnz <= permute < upper_nnz + diag_nnz for diag entries
 **        upper_nnz + diag_nnz <= permute for lower entries
 **        2*upper_nnz + diag_nnz <= permute for interface entries
 ** @param scale scaling factor to scale the matrix entries
 ** @param diag pointer to diagonal entries of the original ldu matrix
 ** @param upper pointer to upper triangular entries of the original ldu matrix
 ** @param lower pointer to lower triangular entries of the original ldu matrix
 ** @param interface pointer to interface entries
 ** @param out pointer to data of output matrix
 **/
void non_symmetric_update_w_interface(
    const label total_nnz, const label diag_nnz, const label upper_nnz,
    const label *permute, const scalar scale, const scalar *diag,
    const scalar *upper, const scalar *lower, const scalar *couple_coeffs,
    scalar *dense);


/* The HostMatrixWrapper class handles the conversion from OpenFOAMs lduMatrix
 * format into Ginkgo array data structures
 *
 * It mainly generates:
 * - sparsity pattern of the local_matrix and
 *   non_local_matrix (with global columns)
 * - CommunicationPattern
 * - implements an interfaces so that
 *   RepartDistMatrix can read_distributed(A_host)
 * */
class HostMatrixWrapper {
private:
    using vec = gko::matrix::Dense<scalar>;
    using idx_array = gko::array<label>;

    const ExecutorHandler exec_;

    const DeviceIdGuardHandler device_id_guard_;

    const label verbose_;

    const word field_name_;

    // Whether the matrix coefficients should be reordered
    // during copy or on device
    const bool reorder_on_copy_;

    const lduAddressing &addr_;

    const scalar *diag_;

    const scalar *upper_;

    const scalar *lower_;

    // multiply the complete system by this factor, ie sAx=sb
    // NOTE this needed to avoid negative diagonal matrix entries, but
    // this could be also achieved by just fliping the sign
    const scalar scaling_;

    // number of local matrix rows
    const label nrows_;

    // number of local upper elements
    // ie coefficients which column_idx < nrows_
    const label upper_nnz_;

    const bool symmetric_;

    // number of local elements on interfaces
    // ie number of interfaces which column_idx is < nrows_
    // and have to be sorted into local matrix
    const label local_interface_nnz_;

    // total number of local upper and lower elements
    // ie 2*upper_nnz_ since the sparsity pattern is symmetric
    const label non_diag_nnz_;

    // nnz of local matrix wo local interfaces
    const label local_matrix_nnz_;

    // nnzs of local matrix including local interfaces
    const label local_matrix_w_interfaces_nnz_;

    // TODO remove
    // mutable SparsityPattern local_sparsity_pattern_;

    // non-local indices
    const label non_local_matrix_nnz_;

    const lduInterfaceFieldPtrsList &interfaces_;

    const FieldField<Field, scalar> &interfaceBouCoeffs_;

    /* Iterates all interfaces and collects the coefficients into a vector
    **
    ** @param local whether local or non local coefficients should be collected
    */
    std::vector<scalar> collect_interface_coeffs(
        const lduInterfaceFieldPtrsList &interfaces_,
        const FieldField<Field, scalar> &interfaceBouCoeffs,
        const bool local) const;


    /* Iterates all interfaces and counts the number of elements
    **
    ** @param interfaces The list of interfaces for the search
    ** @param proc_interfaces Count only elements on (true)
    *processorLduInterfaces or exclude processorLduInterfaces (false)
    */
    label count_interface_nnz(const lduInterfaceFieldPtrsList &interfaces,
                              bool proc_interfaces) const;

    /** Iterates all local interfaces and returns the relative order and
    **corresponding row and column indices
    **
    ** @return vector of tuples containing the interface number, the local row,
    *the local column
    **/
    std::vector<std::tuple<label, label, label>>
    collect_local_interface_indices(
        const lduInterfaceFieldPtrsList &interfaces_) const;


    /** Iterates all interfaces and collect the corresponding cell id (row)
    ** and a unique counter
    **
    ** @return vector of size nnz_non_local, with a running index, row
    ** index, and a unique compressed column index
    ** sections for each interface
    ** ret = [(1,2,1),(2,20, 2), (3, 20, 2) ...]
    **         i0   i1,   i...
    */
    std::vector<std::tuple<label, label, label, label>>
    collect_cells_on_non_local_interface(
        const lduInterfaceFieldPtrsList &interfaces) const;


    // updater


public:
    // segregated matrix wrapper constructor
    HostMatrixWrapper(const ExecutorHandler &exec, const objectRegistry &db,
                      label nrows, label upper_nnz, bool symmetric,
                      const scalar *diag, const scalar *upper,
                      const scalar *lower, const lduAddressing &addr,
                      const FieldField<Field, scalar> &interfaceBouCoeffs,
                      const FieldField<Field, scalar> &interfaceIntCoeffs,
                      const lduInterfaceFieldPtrsList &interfaces,
                      const dictionary &solverControls, const word &fieldName,
                      label verbose);

    // getter

public:
    static std::shared_ptr<HostMatrixWrapper> create(
        const ExecutorHandler &exec, const objectRegistry &db, label nrows,
        label upper_nnz, bool symmetric, const scalar *diag,
        const scalar *upper, const scalar *lower, const lduAddressing &addr,
        const FieldField<Field, scalar> &interfaceBouCoeffs,
        const FieldField<Field, scalar> &interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls, const word &fieldName, label verbose)
    {
        return std::make_shared<HostMatrixWrapper>(
            exec, db, nrows, upper_nnz, symmetric, diag, upper, lower, addr,
            interfaceBouCoeffs, interfaceIntCoeffs, interfaces, solverControls,
            fieldName, verbose);
    };

    void update(const scalar *diag, const scalar *upper, const scalar *lower,
                const FieldField<Field, scalar> &interfaceBouCoeffs,
                const FieldField<Field, scalar> &interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList &interfaces);

    /** Based on OpenFOAMs ldu matrix format this function computes two
     ** consecutive index arrays (local_sparsisty_.row_idxs and col_idxs) in row
     ** major ordering and the permutation index (local_sparsity_.ldu_mapping),
     ** which are required to create to a ginkgo matrix
     **/
    std::shared_ptr<SparsityPattern> compute_local_sparsity(
        std::shared_ptr<const gko::Executor> exec) const;

    /** Copies the LDU matrix coefficients to local_coeffs without changing or
     ** reinstantiating the sparsity pattern.
     ** This uses the local_sparsity_.ldu_mapping to permute the data already
     ** on the host or device to be in row major order.
     **/
    void compute_local_coeffs(std::shared_ptr<const SparsityPattern> sparsity,
                              gko::array<scalar> &target_coeffs) const;

    /** Copies the interface matrix coefficients to non_local_coeffs without
    ** changing or reinstantiating the sparsity pattern.
    **/
    void compute_non_local_coeffs(
        std::shared_ptr<const SparsityPattern> sparsity,
        gko::array<scalar> &target_coeffs) const;

    /** Based on OpenFOAMs interfaces this function computes two
     ** consecutive index arrays (non_local_sparsisty_.row_idxs and col_idxs) in
     ** row *major ordering and the permutation index
     ** (non_local_sparsity_.ldu_mapping), which are required to create a ginkgo
     ** matrix
     **/
    std::shared_ptr<SparsityPattern> compute_non_local_sparsity(
        std::shared_ptr<const gko::Executor> exec) const;

    /** Iterates all interfaces and counts the number of unique neighbour
     ** processors and number of interfaces in total for this processor
     ** and collects all interface cells of this rank.
     **
     ** @param interfaces The list of interfaces for the search
     ** @return the CommunicationPattern
     */
    CommunicationPattern create_communication_pattern(
        const gko::experimental::mpi::communicator &comm) const;


    bool get_verbose() const { return verbose_; }

    label get_local_nrows() const { return nrows_; }

    label get_upper_nnz() const { return upper_nnz_; }

    label get_local_matrix_nnz() const { return local_matrix_nnz_; }

    /** Create a view into host diag ptr  */
    const scalar *get_diag() const { return diag_; }

    /** Create a view into host lower ptr  */
    const scalar *get_lower() const { return lower_; }

    /** Create a view into host lower ptr  */
    const scalar *get_upper() const { return upper_; }

    /** Create a view into host lower ptr  */
    bool symmetric() const { return symmetric_; }

    gko::dim<2> get_size() const { return gko::dim<2>(nrows_, nrows_); }

    const ExecutorHandler &get_exec_handler() const { return exec_; }
};


}  // namespace Foam
#endif
