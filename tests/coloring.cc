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

  coloring.forEachColor([&](auto /*i*/, const auto &colorIdxs) {
    // Check that each row does not contain indices of the same color
    for (auto colorIdx : colorIdxs) {
      PetscInt ncols;
      const PetscInt *cols;
      MatGetRow(mat, rowstart + colorIdx, &ncols, &cols, nullptr);

      for (PetscInt i = 0; i < ncols; ++i) {
        auto col = cols[i];
        for (auto idx : colorIdxs) {
          if (col == colorIdx + rowstart)
            continue;

          REQUIRE(col != idx + rowstart);
        }
      }

      MatRestoreRow(mat, colorIdx, &ncols, &cols, nullptr);
    }
  });

  MatDestroy(&mat);
}

TEST_CASE("Coloring constructor creates no coloring in sequential run", "[.][seq]") {
  auto mat = create_test_mat(5);

  pm::Coloring coloring(mat);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PetscInt rowstart, rowend;
  MatGetOwnershipRange(mat, &rowstart, &rowend);

  std::vector<PetscInt> expected(rowend - rowstart);
  std::iota(expected.begin(), expected.end(), 0);

  coloring.forEachColor(
      [&](auto /*i*/, const auto &colorIdxs) { REQUIRE(expected == colorIdxs); });

  MatDestroy(&mat);
}

TEST_CASE("Coloring constructor creates correct red/black coloring when given DM", "[.][seq]") {
  auto dm = create_test_dm(5);
  Mat mat;
  DMCreateMatrix(dm, &mat);

  pm::Coloring coloring(mat, dm);

  coloring.forEachColor([&](auto i, const auto &colorIdxs) {
    if (i == 0) { // red
      std::vector<PetscInt> expected;
      for (int i = 0; i < 25; i += 2)
        expected.push_back(i);

      REQUIRE(colorIdxs == expected);
    } else if (i == 1) { // black
      std::vector<PetscInt> expected;
      for (int i = 1; i < 25; i += 2)
        expected.push_back(i);

      REQUIRE(colorIdxs == expected);
    } else {
      REQUIRE(false); // There should be exactly two colors
    }
  });

  MatDestroy(&mat);
  DMDestroy(&dm);
}

TEST_CASE("Coloring constructor creates correct scatters for r/b coloring", "[.][mpi]") {
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
    PetscSFGetGraph(coloring.getScatter(0), &nroots, &nleaves, &ilocal, &iremote);

    REQUIRE(ilocal == nullptr); // ilocal == nullptr means values are scattered
                                // into contiguous memory

    std::vector<PetscSFNode> expectedIremote;
    PetscInt expectedNleaves = -1;

    switch (rank) {
    case 0:
      expectedNleaves = 4;
      expectedIremote = {{1, 0}, {2, 0}, {1, 4}, {2, 2}};
      break;

    case 1:
      expectedNleaves = 2;
      expectedIremote = {{0, 5}, {3, 1}};
      break;
    case 2:
      expectedNleaves = 2;
      expectedIremote = {{0, 7}, {3, 2}};
      break;

    case 3:
      expectedNleaves = 2;
      expectedIremote = {{1, 4}, {2, 2}};
      break;
    }

    REQUIRE(expectedNleaves == nleaves);
    REQUIRE(expectedIremote == std::vector<PetscSFNode>(iremote, iremote + nleaves));
  }

  MatDestroy(&mat);
  DMDestroy(&dm);
}
