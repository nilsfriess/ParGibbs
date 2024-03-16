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

TEST_CASE("Symmetric sweep is the same as forward+backward sweep",
          "[.][seq][mpi]") {
  std::mt19937_64 engine(Catch::getSeed());

  auto mat = create_test_mat(9);
  auto op = std::make_shared<pm::LinearOperator>(mat);
  op->color_matrix();

  pm::MulticolorGibbsSampler sampler(op, &engine);

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

TEST_CASE("Gibbs sampler converges to target mean", "[.][seq][mpi]") {
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

  pm::fill_vec_rand(rhs, engine);

  MatZeroRowsColumns(
      mat, dirichletRows.size(), dirichletRows.data(), 1., sample, rhs);

  op->color_matrix(dm);

  pm::MulticolorGibbsSampler sampler(
      op, &engine, 1., pm::GibbsSweepType::Forward);
  sampler.setFixedRhs(rhs);

  constexpr std::size_t n_burnin = 10'000;
  constexpr std::size_t n_samples = 1'000'000;

  sampler.sample(sample, rhs, n_burnin);

  for (std::size_t n = 0; n < n_samples; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / n_samples, sample);
  }

  // PetscViewer viewer;
  // PetscViewerVTKOpen(MPI_COMM_WORLD, "vec.vts", FILE_MODE_WRITE, &viewer);
  // PetscObjectSetName((PetscObject)sample, "sample");
  // VecView(sample, viewer);
  // PetscViewerDestroy(&viewer);

  // Compute expected mean = A^{-1} * rhs
  Vec exp_mean;
  VecDuplicate(mean, &exp_mean);

  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetFromOptions(ksp);
  KSPSetOperators(ksp, mat, mat);
  KSPSolve(ksp, rhs, exp_mean);
  KSPDestroy(&ksp);

  PetscReal norm_expected;
  VecNorm(exp_mean, NORM_INFINITY, &norm_expected);

  PetscReal norm_computed;
  VecNorm(mean, NORM_INFINITY, &norm_computed);

  // VecView(sample, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(exp_mean, PETSC_VIEWER_STDOUT_WORLD);

  REQUIRE_THAT(norm_computed, WithinRel(norm_expected, 0.01));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&exp_mean);
}