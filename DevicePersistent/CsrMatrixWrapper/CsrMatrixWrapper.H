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

Class
    Foam::GKOCG

Author: Gregor Olenik <go@hpsim.de>

SourceFiles
    GKOCG.C

\*---------------------------------------------------------------------------*/
#ifndef OGL_IOGKOMatrixHandler_INCLUDED_H
#define OGL_IOGKOMatrixHandler_INCLUDED_H

#include "fvCFD.H"

#include "../DevicePersistentArray/DevicePersistentArray.H"
#include "../DevicePersistentVector/DevicePersistentVector.H"
#include "../ExecutorHandler/ExecutorHandler.H"
#include "../IOGlobalIndex/gkoGlobalIndex.H"

namespace Foam {


template <typename mtx>
struct MatrixInitFunctor {
    using dist_mtx = gko::distributed::Matrix<scalar, label, label>;

    const ExecutorHandler &exec_;

    const PersistentPartition &partition_;

    const PersistentArray<label> &col_idxs_;

    const PersistentArray<label> &row_idxs_;

    const PersistentArray<scalar> &values_;

    const label verbose_;

    MatrixInitFunctor(const ExecutorHandler &exec,
                      const PersistentPartition &partition,
                      const PersistentArray<label> &col_idxs,
                      const PersistentArray<label> &row_idxs,
                      const PersistentArray<scalar> &values,
                      const label verbose)
        : exec_(exec),
          partition_(partition),
          col_idxs_(col_idxs),
          row_idxs_(row_idxs),
          values_(values),
          verbose_(verbose)
    {}

    void update(std::shared_ptr<dist_mtx> &csr_matrix) const
    {
        // FIXME for now we just rebuild the distributed matrix
        // on the Ginkgo side some update functionalities should be implemented

        init()->move_to(csr_matrix.get());
    }

    std::shared_ptr<mtx> init() const
    {
        using coo_mtx = gko::matrix::Coo<scalar>;

        label nCells = partition_.get_local_host_size();
        word msg{"init global csr matrix of size " + std::to_string(nCells)};
        LOG_1(verbose_, msg)

        auto values = values_.get_array();
        auto cols = col_idxs_.get_array();
        auto rows = row_idxs_.get_array();

        // check if sorted
        if (false) {
            bool is_sorted_rows = true;
            bool is_sorted_cols = true;
            auto rows_data = rows->get_const_data();
            auto cols_data = cols->get_const_data();
            for (size_t i = 1; i < cols->get_num_elems(); i++) {
                if (rows_data[i] < rows_data[i - 1]) {
                    is_sorted_rows = false;
                    Info << "rows sorting error element " << i << " row[i] "
                         << rows_data[i] << " row[i-1] " << rows_data[i - 1]
                         << endl;
                }
                // same row but subsequent column is smaller
                if (cols_data[i] < cols_data[i - 1] &&
                    rows_data[i] == rows_data[i - 1]) {
                    is_sorted_cols = false;
                    Info << "cols sorting error element " << i << " row[i] "
                         << rows_data[i] << " row[i-1] " << rows_data[i - 1]
                         << " col[i] " << cols_data[i] << " col[i-1] "
                         << cols_data[i - 1] << endl;
                }
            }

            Info << "is_sorted rows " << is_sorted_rows << endl;
            Info << "is_sorted cols " << is_sorted_cols << endl;

            if (!is_sorted_cols || !is_sorted_rows) {
                for (size_t i = 1; i < cols->get_num_elems(); i++) {
                    Info << i << "sparsity (" << rows_data[i] << ","
                         << cols_data[i] << ")\n";
                }
            }
        }
        auto exec = exec_.get_ref_exec();

        auto start_build= std::chrono::steady_clock::now();
        auto local_mtx = gko::share(
            coo_mtx::create(exec, gko::dim<2>(nCells, nCells), *values.get(),
                            *cols.get(), *rows.get()));
        auto end_build = std::chrono::steady_clock::now();
        auto delta_t_build = std::chrono::duration_cast<std::chrono::microseconds>(
			end_build - start_build).count();                                                      

        gko::matrix_data<scalar, label> A_data;
        auto start_write= std::chrono::steady_clock::now();
        local_mtx->write(A_data);
        auto end_write= std::chrono::steady_clock::now();
        auto delta_t_write = std::chrono::duration_cast<std::chrono::microseconds>(
			end_write - start_write).count();                                                      

        auto num_rows = partition_.get_total_size();
        A_data.size = gko::dim<2>(num_rows, num_rows);

        auto comm = exec_.get_gko_mpi_host_comm();
        auto dist_A = gko::share(dist_mtx::create(*comm.get()));
        auto start_dist= std::chrono::steady_clock::now();
        dist_A->read_distributed(A_data, partition_.get_host_partition().get());
        auto end_dist= std::chrono::steady_clock::now();
        auto delta_t_dist = std::chrono::duration_cast<std::chrono::microseconds>(
			end_dist - start_dist).count();                                                      

        if (partition_.get_ranks_per_gpu() == 1) {
            auto device_mat = gko::share(
                dist_mtx::create(*exec_.get_gko_mpi_device_comm().get()));
            dist_A->move_to(device_mat.get());
            return device_mat;
        }

        // TODO test if this needs to be persistent
        auto start_repart = std::chrono::steady_clock::now();
        auto repartitioner =
            gko::share(gko::distributed::repartitioner<label, label>::create(
                *comm.get(), partition_.get_host_partition(),
                partition_.get_device_partition()));
        auto end_repart = std::chrono::steady_clock::now();
        auto delta_t_repart = std::chrono::duration_cast<std::chrono::microseconds>(
			end_repart - start_repart).count();                                                      

        auto to_mat =
            gko::share(dist_mtx::create(repartitioner->get_to_communicator()));

        auto start_gather = std::chrono::steady_clock::now();
        repartitioner->gather(dist_A.get(), to_mat.get());
        auto end_gather = std::chrono::steady_clock::now();
        auto delta_t_gather= std::chrono::duration_cast<std::chrono::microseconds>(
			end_gather - start_gather).count();                                                      

	std::cout << 
		" delta t build " << delta_t_build <<
		" delta t write " << delta_t_write <<
		" delta t repart " << delta_t_repart << 
		" delta t gather " << delta_t_gather << std::endl;

        auto device_mat = gko::share(
            dist_mtx::create(*exec_.get_gko_mpi_device_comm().get()));

        to_mat->move_to(device_mat.get());

        return device_mat;
    }
};


template <typename mtx>
class MatrixWrapper {
private:
    const label verbose_;

    const bool export_;

    mutable PersistentBase<mtx, MatrixInitFunctor<mtx>> gkomatrix_;

    mutable label prev_solve_iters_ = 0;


public:
    MatrixWrapper(const objectRegistry &db, const PersistentExecutor &exec,
                  const PersistentArray<label> &col_idxs,
                  const PersistentArray<label> &row_idxs,
                  const PersistentArray<scalar> &values,
                  const PersistentPartition &partition,
                  const dictionary &controlDict, const word sys_matrix_name,
                  const label verbose)
        : verbose_(verbose),
          export_(controlDict.lookupOrDefault<Switch>("export", false)),
          gkomatrix_{
              sys_matrix_name + "_matrix", db,
              MatrixInitFunctor<mtx>(exec, partition, col_idxs, row_idxs,
                                     values, verbose_),
              controlDict.lookupOrDefault<Switch>("updateSysMatrix", true),
              verbose_}
    {}

    std::shared_ptr<gko::LinOp> get() const
    {
        return gkomatrix_.get_persistent_object();
    }


    bool get_export() const { return export_; }
};

}  // namespace Foam

#endif
