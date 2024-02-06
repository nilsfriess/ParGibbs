#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "parmgmc/linear_operator.hh"

#include "test_helpers.hh"

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>
#include <petscvec.h>
#include <random>

using namespace Catch::Matchers;

TEST_CASE("Symmetric sweep is the same as forward+backward sweep") {
  std::mt19937_64 engine(Catch::getSeed());

  auto mat = create_test_mat(81);
  auto op = std::make_shared<parmgmc::LinearOperator>(mat);

  parmgmc::MulticolorGibbsSampler sampler(op, &engine);

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

  for (PetscInt i=0; i<size; ++i)
    REQUIRE_THAT(res1data[i], WithinAbs(res2data[i], 1e-8));

  VecRestoreArrayRead(res2, &res1data);
  VecRestoreArrayRead(res1, &res2data);

  VecDestroy(&res1);
  VecDestroy(&res2);
  VecDestroy(&rhs);
  VecDestroy(&sample);
}
