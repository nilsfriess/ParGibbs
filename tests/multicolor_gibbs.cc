#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "parmgmc/common/helpers.hh"
#include "parmgmc/linear_operator.hh"

#include "test_helpers.hh"

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <pcg_random.hpp>

#include <memory>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmtypes.h>
#include <petscksp.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <random>

using namespace Catch::Matchers;
namespace pm = parmgmc;

TEST_CASE("Symmetric sweep is the same as forward+backward sweep", "[.][seq][mpi]") {
  std::mt19937_64 engine(Catch::getSeed());

  auto mat = create_test_mat(9);
  pm::LinearOperator op{mat};
  op.colorMatrix();

  pm::MulticolorGibbsSampler sampler(op, engine);

  Vec sample;
  MatCreateVecs(mat, &sample, nullptr);

  Vec rhs;
  VecDuplicate(sample, &rhs);
  VecZeroEntries(rhs);

  // Perform symmetric sweep
  VecZeroEntries(sample);
  sampler.setSweepType(parmgmc::GibbsSweepType::Symmetric);
  sampler.sample(sample, rhs);

  Vec res1;
  VecDuplicate(sample, &res1);
  VecCopy(sample, res1);

  // Perform forward+backward sweep
  engine.seed(Catch::getSeed()); // Make sure to use the same seed

  VecZeroEntries(sample);
  sampler.setSweepType(parmgmc::GibbsSweepType::Forward);
  sampler.sample(sample, rhs);
  sampler.setSweepType(parmgmc::GibbsSweepType::Backward);
  sampler.sample(sample, rhs);

  Vec res2;
  VecDuplicate(sample, &res2);
  VecCopy(sample, res2);

  // Compare the resulting samples
  PetscInt size;
  VecGetLocalSize(res1, &size);

  const PetscScalar *res1data, *res2data;
  VecGetArrayRead(res1, &res1data);
  VecGetArrayRead(res2, &res2data);

  for (PetscInt i = 0; i < size; ++i)
    REQUIRE_THAT(res1data[i], WithinAbs(res2data[i], 1e-8));

  VecRestoreArrayRead(res2, &res1data);
  VecRestoreArrayRead(res1, &res2data);

  VecDestroy(&res1);
  VecDestroy(&res2);
  VecDestroy(&rhs);
  VecDestroy(&sample);
}

TEST_CASE("Gibbs sampler converges to target mean", "[.][seq][mpi][mg]") {
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

  auto dm = create_test_dm(5);
  auto [mat, dirichletRows] = create_test_mat(dm);

  pm::LinearOperator op{mat};

  Vec sample, rhs, expMean, mean;
  DMCreateGlobalVector(dm, &sample);
  VecDuplicate(sample, &rhs);
  VecDuplicate(sample, &mean);
  VecDuplicate(mean, &expMean);

  pm::fillVecRand(expMean, engine);
  MatZeroRowsColumns(mat, dirichletRows.size(), dirichletRows.data(), 1., sample, expMean);
  MatMult(mat, expMean, rhs);

  op.colorMatrix(dm);

  pm::MulticolorGibbsSampler sampler(op, engine, 1., pm::GibbsSweepType::Forward);

  constexpr std::size_t N_BURNIN = 1000;
  constexpr std::size_t N_SAMPLES = 1'000'000;

  sampler.sample(sample, rhs, N_BURNIN);

  for (std::size_t n = 0; n < N_SAMPLES; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / N_SAMPLES, sample);
  }

  PetscReal normExpected;
  VecNorm(expMean, NORM_INFINITY, &normExpected);

  PetscReal normComputed;
  VecNorm(mean, NORM_INFINITY, &normComputed);

  REQUIRE_THAT(normComputed, WithinRel(normExpected, 0.01));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&expMean);
}
