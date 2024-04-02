#include "parmgmc/dm_hierarchy.hh"
#include "test_helpers.hh"

#include <algorithm>
#include <cmath>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>

#include <petscvec.h>
#include <random>

int getDMVertices(DM dm) {
  const auto numVertices = [](PetscInt dim, PetscInt m, PetscInt n, PetscInt p) {
    PetscInt res = m;
    if (dim > 1)
      res *= n;
    if (dim > 2)
      res *= p;
    return res;
  };

  PetscInt dim, m, n, p;

  DMDAGetInfo(dm, &dim, &m, &n, &p, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr);
  return numVertices(dim, m, n, p);
}

TEST_CASE("DMHierarchy creates correct number of levels", "[.][seq][mpi]") {
  auto coarseDm = create_test_dm(5);

  const auto nLevels = 5;

  parmgmc::DMHierarchy dh(coarseDm, nLevels);

  REQUIRE(dh.numLevels() == nLevels);
}

TEST_CASE("DMHierarchy returns correct coarse space", "[.][seq][mpi]") {
  auto coarseDm = create_test_dm(5);

  const auto nLevels = 5;
  parmgmc::DMHierarchy dh(coarseDm, nLevels);

  REQUIRE(dh.getCoarse() == coarseDm);
}

TEST_CASE("DMHierarchy fine grid has correct number of vertices", "[.][seq]") {
  auto coarseDm = create_test_dm(5);

  std::mt19937 engine(Catch::getSeed());
  std::uniform_int_distribution dist(2, 8);

  const auto nLevels = dist(engine);
  parmgmc::DMHierarchy dh(coarseDm, nLevels);

  auto nCoarse = getDMVertices(dh.getCoarse());
  auto nFine = getDMVertices(dh.getFine());

  /* The coarsest grid has 2^n + 1 vertices per dimension, for some positive
   * integer n. Due to the uniform refinement, we expect the next level to have
   * (2^n + 1) + 2^n = 2^(n+1) + 1 vertices. The next then has 2^(n+2) + 1, etc.
   * Thus, the ratio between the fine and coarse level is (2^(n+l-1) + 1) / (2^n
   * + 1). We subtract 1 from both the numerator and the denominator and are
   * then left with 2^(l-1). Taking the log with base two then should give l-1.
   */

  REQUIRE(std::log2((std::sqrt(nFine) - 1) / (std::sqrt(nCoarse) - 1)) == nLevels - 1);
}

TEST_CASE("DMHierarchy.get_interpolation returns correct interpolation operator", "[.][seq][mpi]") {
  auto coarseDm = create_test_dm(5);

  const auto nLevels = 5;
  parmgmc::DMHierarchy dh(coarseDm, nLevels);

  Vec vc, vf;

  DMCreateGlobalVector(dh.getDm(1), &vc);
  DMCreateGlobalVector(dh.getDm(2), &vf);

  // We interpolate a constant vector which should result in a constant vector
  PetscScalar val = 1;

  VecSet(vc, val);
  MatInterpolate(dh.getInterpolation(1), vc, vf);

  PetscInt size;
  VecGetLocalSize(vf, &size);

  const PetscScalar *data;
  VecGetArrayRead(vf, &data);

  REQUIRE(std::all_of(data, data + size, [val](auto v) { return v == val; }));

  VecRestoreArrayRead(vf, &data);

  VecDestroy(&vc);
  VecDestroy(&vf);
}
