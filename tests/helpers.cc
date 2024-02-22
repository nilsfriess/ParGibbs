#include "parmgmc/common/helpers.hh"
#include "catch2/matchers/catch_matchers.hpp"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/types.hh"
#include "parmgmc/linear_operator.hh"
#include "petscvec.h"
#include "test_helpers.hh"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <petscdm.h>
#include <petscsystypes.h>
#include <random>

using namespace Catch::Matchers;
namespace pm = parmgmc;

TEST_CASE("fill_vec_rand fills vector with N(0,1) samples", "[.][long]") {
  Vec vec;
  const int size = 5;
  VecCreateSeq(MPI_COMM_WORLD, size, &vec);

  Vec mean;
  VecDuplicate(vec, &mean);
  VecZeroEntries(mean);

  Mat cov;
  MatCreateSeqDense(MPI_COMM_WORLD, size, size, nullptr, &cov);

  std::mt19937 engine;

  const int n_samples = 1000000;

  {
    for (int n = 0; n < n_samples; ++n) {
      // Function that takes size as parameter
      pm::fill_vec_rand(vec, size, engine);

      VecAXPY(mean, 1. / n_samples, vec);
    }

    double norm;
    VecMean(mean, &norm);

    REQUIRE_THAT(norm, WithinAbs(0, 1e-3));
  }

  {
    for (int n = 0; n < n_samples; ++n) {
      // Function that doesn't take size as parameter
      pm::fill_vec_rand(vec, engine);

      VecAXPY(mean, 1. / n_samples, vec);
    }

    double norm;
    VecMean(mean, &norm);

    REQUIRE_THAT(norm, WithinAbs(0, 1e-3));
  }
}

TEST_CASE("make_topmidbot_partition makes correct top and bot partition", "[topmid]") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = pm::LinearOperator(mat);

  pm::BotMidTopPartition partition;
  pm::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4) {
    SKIP("This test case expects exactly 4 MPI ranks");
  }

  // First check if partition.top and partition.bot are ordered correctly: We expect them to be
  // sorted such that their indices are ascending, and if the index is same, then their neighbor
  // indices are in ascending order
  for (int i = 0; i < (int)partition.top.size() - 1; ++i) {
    REQUIRE(partition.top[i].index <= partition.top[i + 1].index);
    if (partition.top[i].index == partition.top[i + 1].index) {
      REQUIRE(partition.top[i].neighbor.index < partition.top[i + 1].neighbor.index);
    }
  }
  // Do the same thing for the partition.bot nodes
  for (int i = 0; i < (int)partition.bot.size() - 1; ++i) {
    REQUIRE(partition.bot[i].index <= partition.bot[i + 1].index);
    if (partition.bot[i].index == partition.bot[i + 1].index) {
      REQUIRE(partition.bot[i].neighbor.index < partition.bot[i + 1].neighbor.index);
    }
  }

  std::vector<pm::BoundaryNode> exp_top;
  std::vector<pm::BoundaryNode> exp_bot;

  switch (rank) {
  case 0: {
    exp_bot.emplace_back(2, 9, 1);
    exp_bot.emplace_back(5, 11, 1);
    exp_bot.emplace_back(6, 15, 2);
    exp_bot.emplace_back(7, 16, 2);
    exp_bot.emplace_back(8, 13, 1);
    exp_bot.emplace_back(8, 17, 2);

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  case 1: {
    exp_bot.emplace_back(5, 22, 3);

    exp_top.emplace_back(0, 2, 0);
    exp_top.emplace_back(2, 5, 0);

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  case 2: {
    exp_bot.emplace_back(5, 23, 3);

    exp_top.emplace_back(0, 6, 0);
    exp_top.emplace_back(1, 7, 0);

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  case 3: {
    exp_top.emplace_back(0, 13, 1);
    exp_top.emplace_back(0, 17, 2);
    exp_top.emplace_back(1, 14, 1);
    exp_top.emplace_back(2, 20, 2);

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  default:
    assert(false && "Unreachable");
  }
}

TEST_CASE("make_topmidbot_partition makes correct interior partition", "[topmid]") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = pm::LinearOperator(mat);

  pm::BotMidTopPartition partition;
  pm::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4) {
    SKIP("This test case expects exactly 4 MPI ranks");
  }

  std::vector<PetscInt> interior(partition.interior1.size() + partition.interior2.size());
  std::copy(partition.interior1.begin(), partition.interior1.end(), interior.begin());
  std::copy(partition.interior2.begin(),
            partition.interior2.end(),
            interior.begin() + partition.interior1.size());
  std::sort(interior.begin(), interior.end());

  switch (rank) {
  case 0: {
    std::vector<PetscInt> exp{0, 1, 3, 4};
    REQUIRE(exp == interior);
  } break;

  case 1: {
    std::vector<PetscInt> exp{1, 3};
    REQUIRE(exp == interior);
  } break;

  case 2: {
    std::vector<PetscInt> exp{3, 4};
    REQUIRE(exp == interior);
  } break;

  case 3: {
    std::vector<PetscInt> exp{3};
    REQUIRE(exp == interior);
  } break;

  default:
    assert(false && "Unreachable");
  }
}

TEST_CASE("make_topmidbot_partition partitions cost properly", "[topmid]") {
  auto dm = create_test_dm(65);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = pm::LinearOperator(mat);

  pm::BotMidTopPartition partition;
  pm::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const PetscInt *ii, *jj;
  PetscBool done;
  MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &ii, &jj, &done);
  assert(done);

  // Compute top cost
  PetscInt top_cost = 0;
  for (auto i : partition.top)
    top_cost += ii[i.index + 1] - ii[i.index];

  // Compute bot cost
  PetscInt bot_cost = 0;
  for (auto i : partition.bot)
    bot_cost += ii[i.index + 1] - ii[i.index];

  // Compute interior cost
  PetscInt interior1_cost = 0;
  for (auto i : partition.interior1)
    interior1_cost += ii[i + 1] - ii[i];

  PetscInt interior2_cost = 0;
  for (auto i : partition.interior2)
    interior2_cost += ii[i + 1] - ii[i];

  // We expect |Int1| + |Top| = |Int2| + |Bot| (where |.| is the cost).
  // Since it might not be possible to partition the domain so that this holds
  // exactly, we allow a 1% error.
  REQUIRE_THAT(interior1_cost + top_cost,
               Catch::Matchers::WithinRel(bot_cost + interior2_cost, 0.01));
}

TEST_CASE("make_topmidbot_partition creates correct high_to_low/low_to_high "
          "scatters",
          "[mpi]") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = pm::LinearOperator(mat);

  pm::BotMidTopPartition partition;
  pm::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4) {
    SKIP("This test case expects exactly 4 MPI ranks");
  }

  PetscInt nroots;
  PetscInt nleaves;
  const PetscInt *ilocal;
  const PetscSFNode *iremote;

  PetscSFGetGraph(partition.high_to_low, &nroots, &nleaves, &ilocal, &iremote);
  CHECK(ilocal == nullptr); // ilocal == nullptr means values are scattered into
                            // contiguous memory

  // Check if iremote has correct values
  std::vector<PetscSFNode> exp_iremote;

  switch (rank) {
  case 0:
    exp_iremote = {{1, 0}, {1, 2}, {2, 0}, {2, 1}, {1, 4}, {2, 2}};
    break;
  case 1:
    exp_iremote = {{3, 1}, {3, 0}};
    break;
  case 2:
    exp_iremote = {{3, 2}, {3, 0}};
    break;
  case 3:
  // No higher processors
  default:
    break;
  }

  REQUIRE(exp_iremote.size() == static_cast<std::size_t>(nleaves));
  for (std::size_t i = 0; i < exp_iremote.size(); ++i) {
    CHECK(exp_iremote[i].index == iremote[i].index);
    CHECK(exp_iremote[i].rank == iremote[i].rank);
  }

  // Do the same thing for low_to_high
  PetscSFGetGraph(partition.low_to_high, &nroots, &nleaves, &ilocal, &iremote);
  CHECK(ilocal == nullptr);

  switch (rank) {
  case 0:
    exp_iremote = {};
    break;
  case 1:
    exp_iremote = {{0, 2}, {0, 5}, {0, 8}};
    break;
  case 2:
    exp_iremote = {{0, 6}, {0, 7}, {0, 8}};
    break;
  case 3:
    exp_iremote = {{1, 4}, {2, 2}, {1, 5}, {2, 5}};
    break;
  default:
    break;
  }

  REQUIRE(exp_iremote.size() == static_cast<std::size_t>(nleaves));
  for (std::size_t i = 0; i < exp_iremote.size(); ++i) {
    CHECK(exp_iremote[i].index == iremote[i].index);
    CHECK(exp_iremote[i].rank == iremote[i].rank);
  }
}