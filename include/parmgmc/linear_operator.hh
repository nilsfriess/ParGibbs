#pragma once

#include <cstring>

#include <petsc.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscmat.h>

#include "parmgmc/common/helpers.hh"

namespace parmgmc {
class LinearOperator {
public:
  LinearOperator(Mat mat, bool transfer_ownership = true)
      : mat{mat}, should_delete{transfer_ownership} {
    PetscFunctionBeginUser;

    MatType type;
    PetscCallVoid(MatGetType(mat, &type));

    if (std::strcmp(type, MATMPIAIJ) == 0) {
      PetscCallVoid(ISColoring_for_Mat(mat, &coloring));
      PetscCallVoid(VecScatter_for_Mat(mat, &scatter));
    }

    PetscFunctionReturnVoid();
  }

  Mat get_mat() const { return mat; }
  ISColoring get_coloring() const { return coloring; }
  VecScatter get_scatter() const { return scatter; }

  bool has_coloring() const { return coloring != nullptr; }

  ~LinearOperator() {
    PetscFunctionBeginUser;

    PetscCallVoid(ISColoringDestroy(&coloring));
    PetscCallVoid(VecScatterDestroy(&scatter));

    if (should_delete)
      PetscCallVoid(MatDestroy(&mat));

    PetscFunctionReturnVoid();
  }

private:
  Mat mat = nullptr;

  // Only set in parallel execution (i.e. when type of mat == MATMPIAIJ)
  ISColoring coloring = nullptr;
  VecScatter scatter = nullptr;

  bool should_delete;
};
} // namespace parmgmc
