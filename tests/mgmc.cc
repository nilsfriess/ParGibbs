#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"

#include "test_helpers.hh"

#include <petscvec.h>

#include <pcg_random.hpp>

#include <memory>
#include <random>

namespace pm = parmgmc;
using namespace Catch::Matchers;

TEST_CASE("MGMC sampler converges to target mean", "[.][mpi]") {
  constexpr std::size_t n_levels = 3;

  auto dm = create_test_dm(3);
  auto dm_hierarchy = std::make_shared<pm::DMHierarchy>(dm, n_levels);

  auto [mat, dirichletRows] = create_test_mat(dm_hierarchy->get_fine());
  auto op = std::make_shared<pm::LinearOperator>(mat);

  Vec sample, rhs, mean, exp_mean;

  MatCreateVecs(mat, &sample, nullptr);
  VecDuplicate(sample, &rhs);
  VecDuplicate(sample, &mean);
  VecDuplicate(sample, &exp_mean);

  MatZeroRowsColumns(
      mat, dirichletRows.size(), dirichletRows.data(), 1., sample, rhs);

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

  pm::MultigridSampler sampler(op, dm_hierarchy, &engine);

  pm::fill_vec_rand(exp_mean, engine);
  MatMult(mat, exp_mean, rhs);

  constexpr std::size_t n_samples = 500'000;

  for (std::size_t n = 0; n < n_samples; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / n_samples, sample);
  }

  PetscReal norm;
  VecNorm(mean, NORM_2, &norm);

  PetscReal norm_expected;
  VecNorm(exp_mean, NORM_2, &norm_expected);

  REQUIRE_THAT(norm, WithinRel(norm_expected, 0.001));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&exp_mean);
}