#include "parmgmc/linear_operator.hh"

#include <catch2/catch_test_macros.hpp>

#include <iostream>

#include <petscmat.h>
#include <petscviewer.h>
#include <petscsystypes.h>

#include "test_helpers.hh"

TEST_CASE("LinearOperator can be constructed", "[.][seq][mpi]") {
  auto mat = create_test_mat(25);
  parmgmc::LinearOperator op(mat);
}

// TEST_CASE("LinearOperator works with matrix that has no off-processor entries",
//           "[.][mpi]") {
//   auto mat = create_test_mat(25, true);
//   parmgmc::LinearOperator op(mat);
// }

TEST_CASE(
    "LinearOperator.has_coloring() returns false if operator has no coloring",
    "[.][seq][mpi]") {
  auto mat = create_test_mat(25);
  parmgmc::LinearOperator op(mat);

  REQUIRE(op.has_coloring() == false);
}

TEST_CASE("LinearOperator.has_coloring() returns true if operator has coloring",
          "[.][seq][mpi]") {
  auto mat = create_test_mat(25);
  parmgmc::LinearOperator op(mat);

  op.color_matrix();

  REQUIRE(op.has_coloring() == true);
}
