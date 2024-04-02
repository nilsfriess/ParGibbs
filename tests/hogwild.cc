#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "test_helpers.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/hogwild.hh"

#include <memory>
#include <random>

#include <mpi.h>

#include <pcg_random.hpp>

namespace pm = parmgmc;

TEST_CASE("Hogwild sampler converges to target mean", "[.][seq][mpi][hg]") {
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

  auto op = std::make_shared<pm::LinearOperator>(mat);

  Vec sample, rhs, mean;
  DMCreateGlobalVector(dm, &sample);
  VecDuplicate(sample, &rhs);
  VecDuplicate(sample, &mean);

  pm::fillVecRand(rhs, engine);

  MatZeroRowsColumns(mat, dirichletRows.size(), dirichletRows.data(), 1., sample, rhs);

  op->colorMatrix(dm);

  pm::HogwildGibbsSampler sampler(op, &engine);

  constexpr std::size_t N_BURNIN = 1000;
  constexpr std::size_t N_SAMPLES = 100'000;

  sampler.sample(sample, rhs, N_BURNIN);

  for (std::size_t n = 0; n < N_SAMPLES; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / N_SAMPLES, sample);
  }

  // PetscViewer viewer;
  // PetscViewerVTKOpen(MPI_COMM_WORLD, "vec.vts", FILE_MODE_WRITE, &viewer);
  // PetscObjectSetName((PetscObject)sample, "sample");
  // VecView(sample, viewer);
  // PetscViewerDestroy(&viewer);

  // Compute expected mean = A^{-1} * rhs
  Vec expMean;
  VecDuplicate(mean, &expMean);

  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetFromOptions(ksp);
  KSPSetOperators(ksp, mat, mat);
  KSPSolve(ksp, rhs, expMean);
  KSPDestroy(&ksp);

  PetscReal normExpected;
  VecNorm(expMean, NORM_INFINITY, &normExpected);

  PetscReal normComputed;
  VecNorm(mean, NORM_INFINITY, &normComputed);

  // VecView(sample, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(exp_mean, PETSC_VIEWER_STDOUT_WORLD);

  REQUIRE_THAT(normComputed, Catch::Matchers::WithinRel(normExpected, 0.01));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&expMean);
}
