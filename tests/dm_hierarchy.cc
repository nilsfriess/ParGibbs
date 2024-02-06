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

int DMGetVertices(DM dm) {
  const auto num_vertices =
      [](PetscInt dim, PetscInt M, PetscInt N, PetscInt P) {
        PetscInt res = M;
        if (dim > 1)
          res *= N;
        if (dim > 2)
          res *= P;
        return res;
      };

  PetscInt dim, M, N, P;

  DMDAGetInfo(dm,
              &dim,
              &M,
              &N,
              &P,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr);
  return num_vertices(dim, M, N, P);
}

TEST_CASE("DMHierarchy creates correct number of levels") {
  auto coarse_dm = create_test_dm(5);

  const auto n_levels = 5;

  parmgmc::DMHierarchy dh(coarse_dm, n_levels);

  REQUIRE(dh.num_levels() == n_levels);
}

TEST_CASE("DMHierarchy returns correct coarse space") {
  auto coarse_dm = create_test_dm(5);

  const auto n_levels = 5;
  parmgmc::DMHierarchy dh(coarse_dm, n_levels);

  REQUIRE(dh.get_coarse() == coarse_dm);
}

TEST_CASE("DMHierarchy fine grid has correct number of vertices") {
  auto coarse_dm = create_test_dm(5);

  std::mt19937 engine(Catch::getSeed());
  std::uniform_int_distribution dist(2, 8);

  const auto n_levels = dist(engine);
  parmgmc::DMHierarchy dh(coarse_dm, n_levels);

  auto n_coarse = DMGetVertices(dh.get_coarse());
  auto n_fine = DMGetVertices(dh.get_fine());

  /* The coarsest grid has 2^n + 1 vertices per dimension, for some positive
   * integer n. Due to the uniform refinement, we expect the next level to have
   * (2^n + 1) + 2^n = 2^(n+1) + 1 vertices. The next then has 2^(n+2) + 1, etc.
   * Thus, the ratio between the fine and coarse level is (2^(n+l-1) + 1) / (2^n
   * + 1). We subtract 1 from both the numerator and the denominator and are
   * then left with 2^(l-1). Taking the log with base two then should give l-1.
   */

  REQUIRE(std::log2((std::sqrt(n_fine) - 1) / (std::sqrt(n_coarse) - 1)) ==
          n_levels - 1);
}

TEST_CASE(
    "DMHierarchy.get_interpolation returns correct interpolation operator") {
  auto coarse_dm = create_test_dm(5);

  const auto n_levels = 5;
  parmgmc::DMHierarchy dh(coarse_dm, n_levels);

  Vec vc, vf;

  DMCreateGlobalVector(dh.get_dm(1), &vc);
  DMCreateGlobalVector(dh.get_dm(2), &vf);

  // We interpolate a constant vector which should result in a constant vector
  PetscScalar val = 1;

  VecSet(vc, val);
  MatInterpolate(dh.get_interpolation(1), vc, vf);

  PetscInt size;
  VecGetLocalSize(vf, &size);

  const PetscScalar *data;
  VecGetArrayRead(vf, &data);

  REQUIRE(std::all_of(data, data + size, [val](auto v) { return v == val; }));

  VecRestoreArrayRead(vf, &data);

  VecDestroy(&vc);
  VecDestroy(&vf);
}
