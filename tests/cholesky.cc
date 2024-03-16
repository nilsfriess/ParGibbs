#include <petscconf.h>
#if PETSC_HAVE_MKL_CPARDISO

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/cholesky.hh"

#include "test_helpers.hh"

#include <memory>
#include <random>

#include <petscvec.h>
#include <petscviewer.h>

namespace pm = parmgmc;
using namespace Catch::Matchers;

TEST_CASE("Cholesky sampler can be constructed", "[.][mpi]") {
  auto mat = create_test_mat(33);
  auto op = std::make_shared<pm::LinearOperator>(mat, true);

  std::mt19937 engine;

  pm::CholeskySampler sampler(op, &engine);

  Vec sample, rhs;

  MatCreateVecs(mat, &sample, nullptr);
  MatCreateVecs(mat, &rhs, nullptr);

  sampler.sample(sample, rhs);

  VecDestroy(&sample);
  VecDestroy(&rhs);
}

// TODO: This only works in parallel but why? Shouldn't the MKL solver also work
// with a single MPI rank?
TEST_CASE("Cholesky sampler computes samples with correct mean", "[.][mpi][chols]") {
  auto mat = create_test_mat(33);
  auto op = std::make_shared<pm::LinearOperator>(mat, true);

  std::mt19937 engine;

  pm::CholeskySampler sampler(op, &engine);

  Vec sample, exp_mean, rhs, mean;

  MatCreateVecs(mat, &exp_mean, nullptr);
  MatCreateVecs(mat, &sample, nullptr);
  MatCreateVecs(mat, &rhs, nullptr);
  MatCreateVecs(mat, &mean, nullptr);

  // pm::fill_vec_rand(rhs, engine);

  VecSet(exp_mean, 1.);

  MatMult(mat, exp_mean, rhs);

  constexpr std::size_t n_samples = 100000;

  for (std::size_t n = 0; n < n_samples; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / n_samples, sample);
  }

  PetscReal norm;
  VecNorm(mean, NORM_2, &norm);

  // Compute expected mean = A^{-1} * rhs
  // Vec exp_mean;
  // VecDuplicate(mean, &exp_mean);
  // KSP ksp;
  // KSPCreate(PETSC_COMM_WORLD, &ksp);
  // KSPSetOperators(ksp, mat, mat);
  // KSPSolve(ksp, rhs, exp_mean);
  // KSPDestroy(&ksp);


  VecView(exp_mean, PETSC_VIEWER_STDOUT_WORLD);
  VecView(mean, PETSC_VIEWER_STDOUT_WORLD);

  PetscReal norm_expected;
  VecNorm(exp_mean, NORM_2, &norm_expected);

  REQUIRE_THAT(norm, WithinAbs(norm_expected, 1e-3));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&exp_mean);
}

#endif