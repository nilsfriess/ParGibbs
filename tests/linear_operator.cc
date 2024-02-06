#include "parmgmc/linear_operator.hh"

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <petscmat.h>
#include <petscviewer.h>

#include "petscdm.h"
#include "petscis.h"
#include "petscsystypes.h"
#include "test_helpers.hh"

bool verify_coloring(Mat mat, ISColoring coloring) {
  bool success = true;

  PetscInt n_colors;
  IS *is_colors;

  ISColoringGetIS(coloring, PETSC_USE_POINTER, &n_colors, &is_colors);

  const PetscInt *ia, *ja;
  PetscBool done;

  MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &ia, &ja, &done);
  assert(done);

  for (PetscInt n = 0; n < n_colors; ++n) {
    PetscInt issize;
    const PetscInt *indices;
    ISGetLocalSize(is_colors[n], &issize);
    ISGetIndices(is_colors[n], &indices);

    /* We loop over each vertex with the current color and verify that none of
     * the entries in the corresponding row of the matrix appear in the current
     * list of vertices (expect for one, namely the diagonal entry). */
    for (PetscInt k = 0; k < issize; ++k) {
      int occurences = 0;

      auto nz = ia[indices[k] + 1] - ia[indices[k]];
      for (PetscInt i = 0; i < nz; ++i) {
        if (ja[ia[indices[k]] + i] == indices[k])
          occurences++;
      }
      if (occurences != 1) {
        success = false;
        break;
      }
    }

    ISRestoreIndices(is_colors[n], &indices);

    if (!success)
      break;
  }

  MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &ia, &ja, &done);
  ISColoringRestoreIS(coloring, PETSC_USE_POINTER, &is_colors);
  return success;
}

TEST_CASE("LinearOperator can be constructed in sequential", "[seq]") {
  auto mat = create_test_mat(25);
  parmgmc::LinearOperator op(mat);
}

TEST_CASE("LinearOperator can be constructed in parallel", "[.][mpi]") {
  auto mat = create_test_mat(25);
  parmgmc::LinearOperator op(mat);
}

TEST_CASE("LinearOperator works with matrix that has no off-processor entries",
          "[.][mpi]") {
  auto mat = create_test_mat(25, true);
  parmgmc::LinearOperator op(mat);
}

TEST_CASE("LinearOperator.color_matrix() returns proper coloring", "[col]") {
  auto mat = create_test_mat(25);
  parmgmc::LinearOperator op(mat);

  op.color_matrix();

  auto coloring = op.get_coloring();

  REQUIRE(verify_coloring(mat, coloring));
}

TEST_CASE(
    "LinearOperator.color_matrix() returns red/black coloring when given DM",
    "[col]") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);
  parmgmc::LinearOperator op(mat);

  op.color_matrix(dm);

  auto coloring = op.get_coloring();

  REQUIRE(verify_coloring(mat, coloring));

  PetscInt ncolors;
  ISColoringGetColors(coloring, nullptr, &ncolors, nullptr);

  // Red-black coloring should return exactly two colors
  REQUIRE(ncolors == 2);
}
