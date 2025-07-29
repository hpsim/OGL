// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later


#pragma once

#include "OGL/DevicePersistent/Base.hpp"

#include "fvCFD.H"

#include <ginkgo/ginkgo.hpp>

namespace Foam {


struct DeviceIdHandler {
    // ratio of inactive to active ranks on the GPU
    label ranks_per_gpu;

    /*
     * @param gpus_per_rank ratio between active host ranks and active ranks
     * gpus on the node i.e. if gpus_per_rank == 1 all host ranks are active on
     * the gpu, which might lead to oversubscription gpus_per_rank == 2 only
     * every second rank will be active ...
     */
    DeviceIdHandler(label ranks_per_gpu_in) : ranks_per_gpu(ranks_per_gpu_in)
    {
        bool par_run = Pstream::parRun();
        if (!par_run) {
            FatalErrorInFunction << "Only parallel runs are supported for OGL"
                                 << exit(FatalError);
        }

        if (Pstream::nProcs(0) % ranks_per_gpu != 0) {
            FatalErrorInFunction
                << " Total number of ranks = " << Pstream::nProcs(0)
                << " is not a multiple of "
                << " ranksPerGPU " << ranks_per_gpu << exit(FatalError);
        }
    }

    /* @brief compute the local device id
     *
     * @param num_devices_per_node number of devices per node
     * @returns
     */
    label compute_device_id(label num_devices_per_node) const
    {
        // if zero devices present device id is always zero
        if (num_devices_per_node == 0) {
            return 0;
        }
        label global_rank = Pstream::myProcNo();
        // clang-format off
        /* example: machine with 4 cpu cores per node 2 accelerators per node, on node 2
        * global ranks [0, 1, 2, 3 | 4, 5, 6, 7]
        * global device id w/o repart  [0, 0, 1, 1 | 2, 2, 3, 3]
        * global device id w repart  [0, x, 1, x | 2, x, 3, x] x-inactive
        * local device_id w/o repart  [0, 1, 0, 1 | 0, 1, 0, 1]
        * local device_id w repart  [0, x, 1, x | 0, x, 1, x] x-inactive
        */
        // clang-format on

        // global_id rpg = 1: [0, 1, 2, 3 | 4, 5, 6, 7]
        // global_id rpg = 2: [0, 0, 1, 1 | 2, 2, 3, 3]
        label device_global_id = global_rank / ranks_per_gpu;

        // compute local round robin id
        // mod of global_id / num_devices_per_node
        return device_global_id % num_devices_per_node;
    }

    /* @brief check if rank is an owning rank
     */
    bool is_owner() const
    {
        label rank = Pstream::myProcNo();
        label owner_rank = rank - (rank % ranks_per_gpu);
        bool is_owner = owner_rank == rank;
        return is_owner;
    }

    /* @brief compute the group id for the split communicator
     * the group id is either 0 for active and 1 for inactive
     */
    label compute_group() const { return is_owner() ? 0 : 1; }
};

struct ExecutorInitFunctor {
    const DeviceIdHandler device_id_handler_;

    const word executor_name_;

    const word field_name_;

    const label verbose_;

    ExecutorInitFunctor(const word executor_name, const word field_name,
                        const label verbose,
                        const DeviceIdHandler device_id_handler)
        : device_id_handler_(device_id_handler),
          executor_name_(executor_name),
          field_name_(field_name),
          verbose_(verbose)
    {}

    void update(std::shared_ptr<gko::Executor>) const {}

    const std::string not_compiled_tag = "not compiled";
    const gko::version_info version = gko::version_info::get();


    std::shared_ptr<gko::Executor> init() const
    {
        auto host_exec = gko::share(gko::ReferenceExecutor::create());

        auto msg = [](auto exec, auto id) {
            std::string s;
            label global_rank = Pstream::myProcNo();
            label global_ranks = Pstream::nProcs(0);
            label device_ranks = 0;
            label local_rank = 0;
#ifdef WITH_ESI_VERSION
            // auto node_comm = Pstream::commInterHost();
            auto node_comm = Pstream::commIntraHost();
            device_ranks = Pstream::nProcs(node_comm);
            local_rank = Pstream::myProcNo(node_comm);
#endif

            // Pstream::barrier(0);
            // sleep(0.03 * global_rank);
            s += std::string("Create ") + std::string(exec) +
                 std::string(" executor, on node: ") + Foam::hostName() +
                 std::string(" device: ") + std::to_string(id) +
                 std::string(" local rank [") + std::to_string(local_rank) +
                 std::string("/") + std::to_string(device_ranks - 1) +
                 std::string("] global rank [") + std::to_string(global_rank) +
                 std::string("/") + std::to_string(global_ranks - 1) +
                 std::string("]");
            return s;
        };

        if (executor_name_ == "cuda") {
            if (version.cuda_version.tag == not_compiled_tag) {
                FatalErrorInFunction
                    << "CUDA Backend was not compiled. Recompile OGL/Ginkgo "
                       "with CUDA backend enabled."
                    << abort(FatalError);
            }
            label id = device_id_handler_.compute_device_id(
                gko::CudaExecutor::get_num_devices());
            auto out_msg = msg(executor_name_, id);
            if (!device_id_handler_.is_owner()) {
                return host_exec;
            }
            LOG_0(verbose_, out_msg)
            auto ret = gko::share(gko::CudaExecutor::create(id, host_exec));
            return ret;
        }
        if (executor_name_ == "sycl" || executor_name_ == "dpcpp") {
            if (version.dpcpp_version.tag == not_compiled_tag) {
                FatalErrorInFunction
                    << "SYCL Backend was not compiled. Recompile OGL/Ginkgo "
                       "with SYCL backend enabled."
                    << abort(FatalError);
            }
            label id = device_id_handler_.compute_device_id(
                gko::DpcppExecutor::get_num_devices("gpu"));
            LOG_0(verbose_, msg(executor_name_, id))
            return gko::share(gko::DpcppExecutor::create(id, host_exec));
        }
        if (executor_name_ == "hip") {
            if (version.hip_version.tag == not_compiled_tag) {
                FatalErrorInFunction
                    << "HIP Backend was not compiled. Recompile OGL/Ginkgo "
                       "with HIP backend enabled."
                    << abort(FatalError);
            }
            label id = device_id_handler_.compute_device_id(
                gko::HipExecutor::get_num_devices());
            auto out_msg = msg(executor_name_, id);
            if (!device_id_handler_.is_owner()) {
                return host_exec;
            }
            LOG_0(verbose_, out_msg)
            auto ret = gko::share(gko::HipExecutor::create(id, host_exec));
            return ret;
        }
        if (executor_name_ == "omp") {
            if (version.omp_version.tag == not_compiled_tag) {
                FatalErrorInFunction
                    << "OMP Backend was not compiled. Recompile OGL/Ginkgo "
                       "with OMP backend enabled."
                    << abort(FatalError);
            }
            return gko::share(gko::OmpExecutor::create());
        }
        if (executor_name_ == "reference") {
            return host_exec;
        }

        FatalErrorInFunction
            << "OGL does not support the executor: " << executor_name_
            << "\nValid choices are: cuda, hip, sycl, omp, or reference"
            << abort(FatalError);
        return {};
    }
};

class ExecutorHandler
    : public PersistentBase<gko::Executor, ExecutorInitFunctor> {
private:
    const bool gko_force_host_buffer_;

    const bool non_orig_device_comm_;

    // whether to split mpi communicators
    const bool split_comm_;

    // original communicator including all ranks
    mutable std::shared_ptr<gko::experimental::mpi::communicator> host_comm_;

    const bool host_rank_;

    DeviceIdHandler device_id_handler_;

    mutable bool device_comm_init_;

    // device communicator including ranks that are associated to a gpu
    mutable std::shared_ptr<gko::experimental::mpi::communicator> device_comm_;

    // communicator including ranks that are associated to a gpu and cpus that
    // repart to gpu
    mutable std::shared_ptr<gko::experimental::mpi::communicator> repart_comm_;

    const word device_executor_name_;

public:
    ExecutorHandler(const objectRegistry &db, const dictionary &solverControls,
                    const word field_name, DeviceIdHandler device_id_handler)
        : PersistentBase<gko::Executor, ExecutorInitFunctor>(
              solverControls.lookupOrDefault("executor", word("reference")) +
                  +"_" + field_name,
              db,
              ExecutorInitFunctor(
                  solverControls.lookupOrDefault("executor", word("reference")),
                  field_name,
                  solverControls.lookupOrDefault("verbose", label(0)),
                  device_id_handler),
              true, 0),
          gko_force_host_buffer_(
              solverControls.lookupOrDefault("forceHostBuffer", false)),
          non_orig_device_comm_(
              solverControls.lookupOrDefault("MPIxRankOffload", false)),
          split_comm_(solverControls.lookupOrDefault("splitMPIComm", true)),
          host_comm_(std::make_shared<gko::experimental::mpi::communicator>(
              MPI_COMM_WORLD, gko_force_host_buffer_)),
          host_rank_(host_comm_->rank()),
          device_id_handler_(device_id_handler),
          device_comm_init_(false),
          device_comm_({}),
          device_executor_name_(
              solverControls.lookupOrDefault("executor", word("reference")))
    {}

    void init_device_comm() const
    {
        if (split_comm_) {
            // gko comm
            label group = device_id_handler_.compute_group();
            MPI_Comm gko_comm;
            label host_rank = 0;
            MPI_Comm_split(MPI_COMM_WORLD, group, host_rank, &gko_comm);
            device_comm_ =
                std::make_shared<gko::experimental::mpi::communicator>(
                    gko_comm, gko_force_host_buffer_);

            // repart comm
            MPI_Comm repart_comm;
            label device_id = device_id_handler_.compute_device_id(4);
            MPI_Comm_split(MPI_COMM_WORLD, device_id, host_rank, &repart_comm);
            repart_comm_ =
                std::make_shared<gko::experimental::mpi::communicator>(
                    repart_comm, gko_force_host_buffer_);

        } else {
            device_comm_ = host_comm_;
        }
        device_comm_init_ = true;
    }

    bool get_gko_force_host_buffer() const
    {
        return this->gko_force_host_buffer_;
    }

    /* whether the mpi allows to send data directly to a remote rank
     * via pair-wise communication
     * */
    bool get_non_orig_device_comm() const { return non_orig_device_comm_; }

    const std::shared_ptr<gko::Executor> get_device_exec() const
    {
        return this->get_persistent_object();
    }

    const std::shared_ptr<gko::Executor> get_ref_exec() const
    {
        return get_device_exec()->get_master();
    }

    word get_exec_name() const { return device_executor_name_; }

    std::shared_ptr<const gko::experimental::mpi::communicator>
    get_device_comm() const
    {
        if (!device_comm_init_) {
            FatalErrorInFunction << "The device_comm is uninitialised. Call "
                                    "init_device_comm() first"
                                 << exit(FatalError);
            OGL_ASSERT_EQ(device_comm_init_, true);
        }
        return this->device_comm_;
    }

    std::shared_ptr<const gko::experimental::mpi::communicator> get_host_comm()
        const
    {
        return this->host_comm_;
    }

    bool get_split_comm() const { return split_comm_; };

    label get_host_rank() const { return get_host_comm()->rank(); };

    label get_device_rank() const { return get_device_comm()->rank(); };
};

using PersistentExecutor = ExecutorHandler;

}  // namespace Foam
