#include <petscconf.h>
#if (PETSC_HAVE_MKL_CPARDISO == 1)

#include "catch2/catch_test_macros.hpp"

#include "parmgmc/linear_operator.hh"

#include "parmgmc/samplers/cholesky.hh"
#include "test_helpers.hh"

#include <memory>
#include <petscviewer.h>
#include <random>

namespace pm = parmgmc;

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
#endif