#include "catch2/catch_test_macros.hpp"

#include "test_helpers.hh"

#include <cassert>

#include <parmgmc/common/coloring.hh>

#include <petscmat.h>
#include <petscoptions.h>
#include <petscsf.h>
#include <petscsftypes.h>

namespace pm = parmgmc;

TEST_CASE("Coloring constructor creates correct coloring", "[.][mpi]") {
  auto mat = create_test_mat(5);

  pm::Coloring coloring(mat, MATCOLORINGGREEDY);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PetscInt rowstart;
  MatGetOwnershipRange(mat, &rowstart, nullptr);

  coloring.for_each_color([&](auto /*i*/, const auto &color_idxs) {
    // Check that each row does not contain indices of the same color
    for (auto color_idx : color_idxs) {
      PetscInt ncols;
      const PetscInt *cols;
      MatGetRow(mat, rowstart + color_idx, &ncols, &cols, nullptr);

      for (PetscInt i = 0; i < ncols; ++i) {
        auto col = cols[i];
        for (auto idx : color_idxs) {
          if (col == color_idx + rowstart)
            continue;

          REQUIRE(col != idx + rowstart);
        }
      }

      MatRestoreRow(mat, color_idx, &ncols, &cols, nullptr);
    }
  });

  MatDestroy(&mat);
}

TEST_CASE("Coloring constructor creates no coloring in sequential run",
          "[.][seq]") {
  auto mat = create_test_mat(5);

  pm::Coloring coloring(mat);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PetscInt rowstart, rowend;
  MatGetOwnershipRange(mat, &rowstart, &rowend);

  std::vector<PetscInt> expected(rowend - rowstart);
  std::iota(expected.begin(), expected.end(), 0);

  coloring.for_each_color([&](auto /*i*/, const auto &color_idxs) {
    REQUIRE(expected == color_idxs);
  });

  MatDestroy(&mat);
}

TEST_CASE(
    "Coloring constructor creates correct red/black coloring when given DM",
    "[.][seq]") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);

  pm::Coloring coloring(mat, dm);

  coloring.for_each_color([&](auto i, const auto &color_idxs) {
    if (i == 0) { // red
      std::vector<PetscInt> expected;
      for (int i = 0; i < 25; i += 2)
        expected.push_back(i);

      REQUIRE(color_idxs == expected);
    } else if (i == 1) { // black
      std::vector<PetscInt> expected;
      for (int i = 1; i < 25; i += 2)
        expected.push_back(i);

      REQUIRE(color_idxs == expected);
    } else {
      REQUIRE(false); // There should be exactly two colors
    }
  });

  MatDestroy(&mat);
  DMDestroy(&dm);
}

TEST_CASE("Coloring constructor creates correct scatters for r/b coloring",
          "[.][mpi]") {
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4)
    SKIP("exactly four MPI ranks expected");

  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);

  pm::Coloring coloring(mat, dm);

  { // red indices
    PetscInt nroots, nleaves;
    const PetscInt *ilocal;
    const PetscSFNode *iremote;
    PetscSFGetGraph(
        coloring.get_scatter(0), &nroots, &nleaves, &ilocal, &iremote);

    REQUIRE(ilocal == nullptr); // ilocal == nullptr means values are scattered
                                // into contiguous memory

    std::vector<PetscSFNode> expected_iremote;
    PetscInt expected_nleaves = -1;

    switch (rank) {
    case 0:
      expected_nleaves = 4;
      expected_iremote = {{1, 0}, {2, 0}, {1, 4}, {2, 2}};
      break;

    case 1:
      expected_nleaves = 2;
      expected_iremote = {{0, 5}, {3, 1}};
      break;
    case 2:
      expected_nleaves = 2;
      expected_iremote = {{0, 7}, {3, 2}};
      break;

    case 3:
      expected_nleaves = 2;
      expected_iremote = {{1, 4}, {2, 2}};
      break;
    }

    REQUIRE(expected_nleaves == nleaves);
    REQUIRE(expected_iremote ==
            std::vector<PetscSFNode>(iremote, iremote + nleaves));
  }

  MatDestroy(&mat);
  DMDestroy(&dm);
}