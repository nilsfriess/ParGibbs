#pragma once

#include <memory>
#include <petsc.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscmat.h>

#include "parmgmc/common/coloring.hh"

namespace parmgmc {
enum class PetscMatType { MPIAIJ, SEQAIJ };

class LinearOperator {
public:
  explicit LinearOperator(Mat mat, bool transfer_ownership = true)
      : mat{mat}, should_delete{transfer_ownership} {
    PetscFunctionBeginUser;
    MatType type;
    PetscCallVoid(MatGetType(mat, &type));

    if (std::strcmp(type, MATMPIAIJ) == 0)
      mattype = PetscMatType::MPIAIJ;
    else if (std::strcmp(type, MATSEQAIJ) == 0)
      mattype = PetscMatType::SEQAIJ;
    else
      PetscCheckAbort(false,
                      MPI_COMM_WORLD,
                      PETSC_ERR_SUP,
                      "Only MATMPIAIJ and MATSEQAIJ types are supported");

    PetscFunctionReturnVoid();
  }

  void color_matrix(MatColoringType coloring_type = MATCOLORINGGREEDY) {
    coloring = std::make_unique<Coloring>(mat, coloring_type);
  }

  void color_matrix(DM dm) { coloring = std::make_unique<Coloring>(mat, dm); }

  Mat get_mat() const { return mat; }
  Coloring *get_coloring() const { return coloring.get(); }

  bool has_coloring() const { return coloring != nullptr; }

  PetscMatType get_mat_type() const { return mattype; }

  ~LinearOperator() {
    PetscFunctionBeginUser;

    if (should_delete)
      PetscCallVoid(MatDestroy(&mat));

    PetscFunctionReturnVoid();
  }

private:
  Mat mat = nullptr;
  PetscMatType mattype;

  std::unique_ptr<Coloring> coloring;

  bool should_delete;
};
} // namespace parmgmc
