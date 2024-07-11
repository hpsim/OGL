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
    std::vector<int> recv_counts(send_counts);
    
    // all offsets should be zero
    // last entry in offsets is total number of send elements
    std::vector<int> send_offsets(comm->size() + 1, 0);
    send_offsets.back() = num_elements;
    std::vector<int> recv_offsets(send_offsets);

    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), 1, label(num_elements));

    // Assert
    // test send counts and revc counts
    EXPECT_EQ(comm_counts.send_counts, send_counts);
    EXPECT_EQ(comm_counts.recv_counts, recv_counts);

    // test send counts and revc offsets
    EXPECT_EQ(comm_counts.send_offsets, send_offsets);
    EXPECT_EQ(comm_counts.recv_offsets, recv_offsets);
}

// Given 4 processes in the communicator
TEST_F(CommunicationPatternFixture, compute_gather_to_owner_counts_single_owner)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto num_elements = 10;
    auto comm_size = comm->size();

    // if gathering to just one owner, all 10 elements are send to rank 0
    std::vector<int> send_counts{num_elements, 0, 0, 0};
 
    // no rank should recv anything except rank 0
    std::vector<int> recv_counts(comm_size) ;
    if (comm->rank() == 0)
        recv_counts = std::vector<int>(comm_size, num_elements);
  
    std::vector<int> send_offsets(comm_size+1);
    send_offsets[comm_size] = num_elements;

    std::vector<int> recv_offsets(comm_size+1);
    if (comm->rank() == 0)
    recv_offsets = std::vector<int>{0, num_elements, num_elements*2, num_elements*3, num_elements*4};

    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), comm_size, label(num_elements));

    // Assert
    // test if the total number of processes is 4, which is hardcoded here
    EXPECT_EQ(comm_size, 4);

    // test send counts and revc counts
    EXPECT_EQ(comm_counts.send_counts, send_counts);
    EXPECT_EQ(comm_counts.recv_counts, recv_counts);

    // test send counts and revc offsets
    EXPECT_EQ(comm_counts.send_offsets, send_offsets);
    EXPECT_EQ(comm_counts.recv_offsets, recv_offsets);
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
