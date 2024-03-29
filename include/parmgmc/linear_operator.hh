#pragma once

#include <memory>
#include <petsc.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscmat.h>

#include "parmgmc/common/coloring.hh"

namespace parmgmc {
enum class PetscMatType { MPIAij, SEQAij };

class LinearOperator : public std::enable_shared_from_this<LinearOperator> {
public:
  explicit LinearOperator(Mat mat, bool transferOwnership = true)
      : mat{mat}, shouldDelete{transferOwnership} {
    PetscFunctionBeginUser;
    MatType type;
    PetscCallVoid(MatGetType(mat, &type));

    if (std::strcmp(type, MATMPIAIJ) == 0)
      mattype = PetscMatType::MPIAij;
    else if (std::strcmp(type, MATSEQAIJ) == 0)
      mattype = PetscMatType::SEQAij;
    else
      PetscCheckAbort(false, MPI_COMM_WORLD, PETSC_ERR_SUP,
                      "Only MATMPIAIJ and MATSEQAIJ types are supported");

    PetscFunctionReturnVoid();
  }

  void colorMatrix() {
    MatColoringType coloringType = MATCOLORINGGREEDY;
    coloring = std::make_unique<Coloring>(mat, coloringType);
  }

  void colorMatrix(DM dm) { coloring = std::make_unique<Coloring>(mat, dm); }

  [[nodiscard]] Mat getMat() const { return mat; }
  [[nodiscard]] Coloring *getColoring() const { return coloring.get(); }

  [[nodiscard]] bool hasColoring() const { return coloring != nullptr; }

  [[nodiscard]] PetscMatType getMatType() const { return mattype; }

  ~LinearOperator() {
    PetscFunctionBeginUser;

    if (shouldDelete)
      PetscCallVoid(MatDestroy(&mat));

    PetscFunctionReturnVoid();
  }

  LinearOperator(LinearOperator &&) = default;

private:
  Mat mat = nullptr;
  PetscMatType mattype;

  std::unique_ptr<Coloring> coloring;

  bool shouldDelete;
};
} // namespace parmgmc
