#include "parmgmc/common/helpers.hh"
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
      parmgmc::fill_vec_rand(vec, size, engine);

      VecAXPY(mean, 1. / n_samples, vec);
    }

    double norm;
    VecMean(mean, &norm);

    REQUIRE_THAT(norm, WithinAbs(0, 1e-3));
  }

  {
    for (int n = 0; n < n_samples; ++n) {
      // Function that doesn't take size as parameter
      parmgmc::fill_vec_rand(vec, engine);

      VecAXPY(mean, 1. / n_samples, vec);
    }

    double norm;
    VecMean(mean, &norm);

    REQUIRE_THAT(norm, WithinAbs(0, 1e-3));
  }
}

TEST_CASE("make_topmidbot_partition makes correct top and bot partition") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = parmgmc::LinearOperator(mat);

  parmgmc::BotMidTopPartition partition;
  parmgmc::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4) {
    SKIP("This test case expects exactly 4 MPI ranks");
  }

  switch (rank) {
  case 0: {
    std::vector<PetscInt> exp_top;
    std::vector<PetscInt> exp_bot{2, 5, 6, 7, 8};

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  case 1: {
    std::vector<PetscInt> exp_top{0, 2};
    std::vector<PetscInt> exp_bot{5};

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  case 2: {
    std::vector<PetscInt> exp_top{0, 1};
    std::vector<PetscInt> exp_bot{5};

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  case 3: {
    std::vector<PetscInt> exp_top{0, 1, 2};
    std::vector<PetscInt> exp_bot;

    REQUIRE(exp_top == partition.top);
    REQUIRE(exp_bot == partition.bot);
  } break;

  default:
    assert(false && "Unreachable");
  }
}

TEST_CASE("make_topmidbot_partition makes correct interior partition") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = parmgmc::LinearOperator(mat);

  parmgmc::BotMidTopPartition partition;
  parmgmc::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4) {
    SKIP("This test case expects exactly 4 MPI ranks");
  }

  std::vector<PetscInt> interior(partition.interior1.size() +
                                 partition.interior2.size());
  std::copy(
      partition.interior1.begin(), partition.interior1.end(), interior.begin());
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

TEST_CASE("make_topmidbot_partition partitions cost properly") {
  auto dm = create_test_dm(65);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  MatSetUp(mat);
  auto op = parmgmc::LinearOperator(mat);

  parmgmc::BotMidTopPartition partition;
  parmgmc::make_botmidtop_partition(mat, partition);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const PetscInt *ii, *jj;
  PetscBool done;
  MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &ii, &jj, &done);
  assert(done);

  const auto cost = [&](const std::vector<PetscInt> &indices) {
    PetscInt c = 0;
    for (auto i : indices)
      c += ii[i + 1] - ii[i];
    return c;
  };

  // We expect |Int1| + |Top| = |Int2| + |Bot| (where |.| is the cost).
  // Since it might not be possible to partition the domain so that this holds
  // exactly, we allow a 1% error.
  REQUIRE_THAT(cost(partition.interior1) + cost(partition.top),
               Catch::Matchers::WithinRel(
                   cost(partition.interior2) + cost(partition.bot), 0.01));
}
