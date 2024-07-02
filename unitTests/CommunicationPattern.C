// SPDX-FileCopyrightText: 2024 OGL authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/CommunicationPattern/CommunicationPattern.H"
#include "gtest/gtest.h"
#include "mpi.h"

class CommunicationPatternFixture : public testing::Test {
 protected:
  CommunicationPatternFixture():
    time(Foam::Time::New()),
    db(Foam::objectRegistry(*time))
    {
    dict.add("executor", "reference");
  }

  const Foam::Time* time;
  Foam::objectRegistry db;
  Foam::dictionary dict;
};


TEST_F(CommunicationPatternFixture, can_get_comm_size)
{
    auto exec = ExecutorHandler(db, dict, "dummy", true);
    auto comm = exec.get_gko_mpi_device_comm();
    EXPECT_EQ(comm->size(), 4);
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
