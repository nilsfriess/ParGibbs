#include "pargibbs/mpi_helper.hh"

#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  pargibbs::mpi_helper helper(&argc, &argv);

  return RUN_ALL_TESTS();
}
