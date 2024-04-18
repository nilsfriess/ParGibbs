#pragma once

#include <petsc.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscmat.h>

#include "parmgmc/common/coloring.hh"

namespace parmgmc {
enum class PetscMatType { MPIAij, SEQAij };

class LinearOperator {
public:
  LinearOperator() = default;

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
    coloring = Coloring{mat};
  }

  void colorMatrix(DM dm) { coloring = Coloring{mat, dm}; }

  [[nodiscard]] Mat getMat() const { return mat; }
  [[nodiscard]] const Coloring &getColoring() const { return coloring; }

  [[nodiscard]] bool hasColoring() const { return coloring.isValid(); }

  [[nodiscard]] PetscMatType getMatType() const { return mattype; }

  ~LinearOperator() {
    PetscFunctionBeginUser;

    if (shouldDelete)
      PetscCallVoid(MatDestroy(&mat));

    PetscFunctionReturnVoid();
  }

  LinearOperator(const LinearOperator &) = delete;
  LinearOperator &operator=(const LinearOperator &) = delete;

  LinearOperator(LinearOperator &&other) noexcept
      : mat{other.mat}, mattype{other.mattype}, coloring{std::move(other.coloring)},
        shouldDelete{other.shouldDelete} {
    other.mat = nullptr;
  }

  LinearOperator &operator=(LinearOperator &&other) noexcept {
    mat = other.mat;
    mattype = other.mattype;
    coloring = std::move(other.coloring);
    shouldDelete = other.shouldDelete;

    other.mat = nullptr;

    return *this;
  }

private:
  Mat mat = nullptr;
  PetscMatType mattype;

  Coloring coloring;

  bool shouldDelete;
};
} // namespace parmgmc
