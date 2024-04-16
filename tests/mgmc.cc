#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"

#include "test_helpers.hh"

#include <petscvec.h>

#include <pcg_random.hpp>

#include <random>

namespace pm = parmgmc;
using namespace Catch::Matchers;

// TODO: This test is currently not run, since it takes ages and still sometimes
// fails (i.e., we would need even more samples)
TEST_CASE("MGMC sampler converges to target mean", "[.][mg]") {
  const std::size_t nLevels = 3;

  auto dm = create_test_dm(3);
  pm::DMHierarchy dmHierarchy{dm, nLevels};

  auto mat = create_test_mat(dmHierarchy.getFine());
  auto op = std::make_shared<pm::LinearOperator>(mat);

  Vec sample, rhs, mean, expMean;

  MatCreateVecs(mat, &sample, nullptr);
  VecDuplicate(sample, &rhs);
  VecDuplicate(sample, &mean);
  VecDuplicate(sample, &expMean);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pcg32 engine;
  {
    int seed;
    if (rank == 0) {
      seed = std::random_device{}();
    }

    // Send seed to all other processes
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    engine.seed(seed);
    engine.set_stream(rank);
  }

  pm::MultigridSampler sampler(op, dmHierarchy, engine);

  pm::fillVecRand(expMean, engine);
  MatMult(mat, expMean, rhs);

  const std::size_t nSamples = 500'000;

  for (std::size_t n = 0; n < nSamples; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / nSamples, sample);
  }

  PetscReal norm;
  VecNorm(mean, NORM_2, &norm);

  PetscReal normExpected;
  VecNorm(expMean, NORM_2, &normExpected);

  REQUIRE_THAT(norm, WithinRel(normExpected, 0.001));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&expMean);
}
