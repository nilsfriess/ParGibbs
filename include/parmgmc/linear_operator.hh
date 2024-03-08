#pragma once

#include <memory>
#include <petsc.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscmat.h>

#include "parmgmc/common/coloring.hh"

namespace parmgmc {
class LinearOperator {
public:
  explicit LinearOperator(Mat mat, bool transfer_ownership = true)
      : mat{mat}, should_delete{transfer_ownership} {}

  void color_matrix(MatColoringType coloring_type = MATCOLORINGGREEDY) {
    coloring = std::make_unique<Coloring>(mat, coloring_type);
  }

  void color_matrix(DM dm) { coloring = std::make_unique<Coloring>(mat, dm); }

  Mat get_mat() const { return mat; }
  Coloring *get_coloring() const { return coloring.get(); }

  bool has_coloring() const { return coloring != nullptr; }

  ~LinearOperator() {
    PetscFunctionBeginUser;

    if (should_delete)
      PetscCallVoid(MatDestroy(&mat));

    PetscFunctionReturnVoid();
  }

private:
  Mat mat = nullptr;

  std::unique_ptr<Coloring> coloring;

  bool should_delete;
};
} // namespace parmgmc
