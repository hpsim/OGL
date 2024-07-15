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

TEST_F(CommunicationPatternFixture, compute_scatter_from_owner_counts_single_owner)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto num_elements = 10;
    auto comm_size = comm->size();
    auto comm_rank = comm->rank();

    // Scatter different number of elements from owner to each non-owner process
    std::vector<int> send_counts(comm_size);
    
    if (comm_rank == 0)
        for (int i = 0; i < comm_size; i++)
            send_counts[i] = num_elements*i;
      
    std::vector<int> recv_counts(comm_size);

    if (comm_rank != 0)
        recv_counts[0] = num_elements*comm_rank;

    std::vector<int> send_offsets(comm_size);
    if (comm_rank == 0)
        for (int i = 0; i < comm_size-1; i++)
            send_offsets[i+1] = send_offsets[i] + num_elements*i;
    
    std::vector<int> recv_offsets(comm_size);

    // Act
    auto comm_counts =
        compute_scatter_from_owner_counts(*exec.get(), comm_size, label(num_elements * comm_rank)); 

    // Assert
    // test send counts and revc counts
    EXPECT_EQ(comm_counts.send_counts, send_counts);
    EXPECT_EQ(comm_counts.recv_counts, recv_counts);
    EXPECT_EQ(comm_counts.send_offsets, send_offsets);
    EXPECT_EQ(comm_counts.recv_offsets, recv_offsets);
}

TEST_F(CommunicationPatternFixture, compute_gather_to_owner_counts_single_owner)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto num_elements = 10;
    auto comm_size = comm->size();
    auto comm_rank = comm->rank();

    // Gather different number of elements to the only owner
    std::vector<int> send_counts(comm_size);
    send_counts[0] = num_elements * comm_rank;
 
    // no rank should receive anything except the owner prcocess (rank 0)
    std::vector<int> recv_counts(comm_size) ;
    if (comm_rank == 0)
        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = num_elements*i;
        }
  
    std::vector<int> send_offsets(comm_size+1);
    send_offsets.back() = num_elements * comm_rank;

    std::vector<int> recv_offsets(comm_size+1);
    if (comm_rank == 0)
    {
        for (int i = 0; i < comm_size; i++)
        {
            recv_offsets[i+1] = recv_offsets[i] + num_elements*i;
        }
    }
    
    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), comm_size, label(num_elements * comm_rank));

    // Assert
    // test send counts and revc counts
    EXPECT_EQ(comm_counts.send_counts, send_counts);
    EXPECT_EQ(comm_counts.recv_counts, recv_counts);

    // test send counts and revc offsets
    EXPECT_EQ(comm_counts.send_offsets, send_offsets);
    EXPECT_EQ(comm_counts.recv_offsets, recv_offsets);
}

TEST_F(CommunicationPatternFixture, compute_gather_to_owner_counts_two_owners)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto num_elements = 10;
    auto comm_rank = comm->rank();
    auto comm_size = comm->size();

    // Expected results for send counts and recv counts
    std::vector<int> send_counts(comm_size, 0);
    if (comm_rank < comm_size/2)
        send_counts[0] = num_elements;
    else
        send_counts[comm_size/2] = num_elements;
    
    std::vector<int> recv_counts(comm_size, 0);
    if (comm_rank == 0)
    {
        for (int i = 0; i < comm_size/2; i++)
            recv_counts[i] = num_elements;
    }
    else if (comm_rank == comm_size/2)
    {
        for (int i = comm_size/2; i < comm_size; i++)
            recv_counts[i] = num_elements;
    }        
    
    // Expected results for send offsets and recv offsets
    std::vector<int> send_offsets(comm_size + 1, 0);
    send_offsets.back() = num_elements;

    std::vector<int> recv_offsets(comm_size + 1, 0);
    if (comm_rank == 0)
    {
        for (int i = 0; i < comm_size/2; i++)
            recv_offsets[i] = num_elements * i;
        
        recv_offsets.back() = num_elements * comm_size/2;
    }
    else if (comm_rank == comm_size/2)
    {
        for (int i = comm_size/2, j = 0; i < comm_size; i++, j++)
            recv_offsets[i] = num_elements * j;
        
        recv_offsets.back() = num_elements * comm_size/2;
    }

    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), comm_size/2, label(num_elements));

    // Assert
    EXPECT_EQ(comm_counts.send_counts, send_counts);
    EXPECT_EQ(comm_counts.recv_counts, recv_counts);
    EXPECT_EQ(comm_counts.send_offsets, send_offsets);
    EXPECT_EQ(comm_counts.recv_offsets, recv_offsets);
}

TEST_F(CommunicationPatternFixture, compute_gather_to_owner_counts_all_owners)
{
    // Arrange
    auto comm = exec->get_gko_mpi_host_comm();
    auto num_elements = 10;
    auto comm_rank = comm->rank();

    // expected results
    std::vector<int> send_counts(comm->size(), 0);
    send_counts[comm->rank()] = num_elements * comm_rank;
    std::vector<int> recv_counts(send_counts);
    
    // all offsets should be zero
    // last entry in offsets is total number of send elements
    std::vector<int> send_offsets(comm->size() + 1, 0);
    send_offsets.back() = num_elements * comm_rank;
    std::vector<int> recv_offsets(send_offsets);

    // Act
    auto comm_counts =
        compute_gather_to_owner_counts(*exec.get(), 1, label(num_elements * comm_rank));

    // Assert
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
