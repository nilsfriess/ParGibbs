#include <petscconf.h>
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

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

TEST_CASE("Cholesky sampler computes samples with correct mean",
          "[.][seq][mpi]") {
  auto mat = create_test_mat(5);
  auto op = std::make_shared<pm::LinearOperator>(mat, true);

  std::mt19937 engine;

  pm::CholeskySampler sampler(op, &engine);

  Vec sample, expMean, rhs, mean;

  MatCreateVecs(mat, &expMean, nullptr);
  MatCreateVecs(mat, &sample, nullptr);
  MatCreateVecs(mat, &rhs, nullptr);
  MatCreateVecs(mat, &mean, nullptr);

  // pm::fill_vec_rand(rhs, engine);

  VecSet(expMean, 1.);

  MatMult(mat, expMean, rhs);

  const std::size_t nSamples = 300'000;

  for (std::size_t n = 0; n < nSamples; ++n) {
    sampler.sample(sample, rhs);

    VecAXPY(mean, 1. / nSamples, sample);
  }

  PetscReal norm;
  VecNorm(mean, NORM_2, &norm);

  PetscReal normExpected;
  VecNorm(expMean, NORM_2, &normExpected);

  REQUIRE_THAT(norm, WithinAbs(normExpected, 1e-3));

  // Cleanup
  VecDestroy(&mean);
  VecDestroy(&sample);
  VecDestroy(&rhs);
  VecDestroy(&expMean);
}

#endif
