#include "parmgmc/mpi_helper.hh"

#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  parmgmc::mpi_helper helper(&argc, &argv);

  return RUN_ALL_TESTS();
}
