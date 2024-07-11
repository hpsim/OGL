// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "OGL/CommunicationPattern.H"

#include "gtest/gtest.h"

#include "mpi.h"

class CommunicationPatternFixture : public testing::Test {
protected:
    CommunicationPatternFixture()
        : time(Foam::Time::New()), db(Foam::objectRegistry(*time))
    {
        dict.add("executor", "reference");
        exec = std::make_shared<ExecutorHandler>(db, dict, "dummy", true);
    }

    const Foam::Time *time;
    Foam::objectRegistry db;
    Foam::dictionary dict;
    std::shared_ptr<ExecutorHandler> exec;
};


TEST_F(CommunicationPatternFixture, compute_owner_rank_single_owner)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();

    // Act
    auto owner_rank = compute_owner_rank(comm->rank(), comm->size());

    // Assert
    EXPECT_EQ(owner_rank, 0);
}

TEST_F(CommunicationPatternFixture, compute_owner_rank_two_owners)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto comm_rank = comm->rank();
    auto comm_size = comm->size();

    // Act
    auto owner_rank = compute_owner_rank(comm_rank, comm_size/2);

    // Assert
    if (comm_rank < comm_size/2)
        EXPECT_EQ(owner_rank, 0);
    else
        EXPECT_EQ(owner_rank, comm_size/2);
}

TEST_F(CommunicationPatternFixture, compute_owner_rank_all_owners)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto comm_rank = comm->rank();

    // Act
    auto owner_rank = compute_owner_rank(comm_rank, 1);

    // Assert
    EXPECT_EQ(owner_rank, comm_rank);
}

TEST_F(CommunicationPatternFixture, compute_gather_to_owner_counts_all_owner)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto num_elements = 10;

    // expected results
    // if gathering to just one owner, all 10 elements are send to itself
    std::vector<int> send_counts(comm->size(), 0);
    send_counts[comm->rank()] = num_elements;
    std::vector<std::vector<int>> send_results(comm->size(), send_counts);
    
    // all offsets should be zero
    // last entry in offsets is total number of send elements
    std::vector<int> send_offsets(comm->size() + 1, 0);
    send_offsets.back() = num_elements;
    std::vector<std::vector<int>> offset_results(comm->size(), send_offsets);

    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), 1, label(num_elements));

    // Assert
    // test send counts and revc counts
    EXPECT_EQ(comm_counts.send_counts, send_results[comm->rank()]);
    EXPECT_EQ(comm_counts.recv_counts, send_results[comm->rank()]);

    // test send offsets and revc offsets
    EXPECT_EQ(comm_counts.send_offsets, offset_results[comm->rank()]);
    EXPECT_EQ(comm_counts.recv_offsets, offset_results[comm->rank()]);
}

// Given 4 processes in the communicator
TEST_F(CommunicationPatternFixture, compute_gather_to_owner_counts_single_owner)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();

    // if gathering to just one owner all 10 elements are send to rank 0
    std::vector<std::vector<int>> send_results(comm->size(),
                                          std::vector<int>{10, 0, 0, 0});
    // no rank should recv anything except rank 0
    std::vector<std::vector<int>> recv_results(comm->size(),
                                          std::vector<int>{0, 0, 0, 0});
    recv_results[0] = std::vector<int>{10, 10, 10, 10};

    std::vector<std::vector<int>> send_offsets_results(comm->size() + 1,
                                          std::vector<int>{0, 0, 0, 0, 10});

    std::vector<std::vector<int>> recv_offsets_results(comm->size() + 1,
                                          std::vector<int>{0, 0, 0, 0, 0});
    recv_offsets_results[0] = std::vector<int>{0, 10, 20, 30, 40};

    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), comm->size(), label(10));

    // Assert
    // test if the total number of processes is 4, which is hardcoded here
    EXPECT_EQ(comm->size(), 4);

    // test send counts and revc counts
    EXPECT_EQ(comm_counts.send_counts, send_results[comm->rank()]);
    EXPECT_EQ(comm_counts.recv_counts, recv_results[comm->rank()]);

    // test send offsets and revc offsets
    EXPECT_EQ(comm_counts.send_offsets, send_offsets_results[comm->rank()]);
    EXPECT_EQ(comm_counts.recv_offsets, recv_offsets_results[comm->rank()]);
}

int main(int argc, char *argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
