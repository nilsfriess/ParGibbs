#pragma once

#include <memory>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include "parmgmc/common/types.hh"

namespace parmgmc {
/* Represents a low rank matrix of the form A S A^T.
   The template parameter can either be Vec, if the matrix S is a diagonal
   matrix or Mat if it is a full matrix. */
template <typename MiddleMat = Mat> class LowrankUpdate {
  static_assert(std::is_same_v<MiddleMat, Mat> ||
                std::is_same_v<MiddleMat, Vec>);

public:
  LowrankUpdate(Mat A, MiddleMat S, MiddleMat S_chol)
      : A{A}, S{S}, S_chol{S_chol} {
    PetscFunctionBeginUser;

    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCallVoid(VecDuplicate(S, &z));
    } else {
      PetscCallVoid(MatCreateVecs(S, &z, NULL));
    }

    PetscFunctionReturnVoid();
  }

  ~LowrankUpdate() { VecDestroy(&z); }

  PetscErrorCode apply(Vec xin, Vec xout) const {
    PetscFunctionBeginUser;

    PetscCall(MatMult(A, xin, z));

    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCall(VecPointwiseMult(z, z, S));
    } else {
      // PetscCall(MatMult(S, xout, z));
      static_assert(!std::is_same_v<MiddleMat, Mat>, "Not implemented yet");
    }

    PetscCall(MatMultTranspose(A, z, xout));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode apply_cholesky_L(Vec xin, Vec xout) const {
    PetscFunctionBeginUser;

    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCall(VecPointwiseMult(z, xin, S_chol));
    } else {
      // PetscCall(MatMult(S_chol, xout, z));
      // PetscCall(VecCopy(z, xout));
      static_assert(!std::is_same_v<MiddleMat, Mat>, "Not implemented yet");
    }

    PetscCall(MatMultTranspose(A, z, xout));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode create_compatible_vecs(Vec *big, Vec *small) const {
    PetscFunctionBeginUser;

    if (big != NULL) {
      PetscCall(MatCreateVecs(A, big, NULL));
    }

    if (small != NULL) {
      if constexpr (std::is_same_v<MiddleMat, Vec>) {
        PetscCall(VecDuplicate(S, small));
      } else {
        PetscCall(MatCreateVecs(S, small, NULL));
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  Mat A;
  MiddleMat S;
  MiddleMat S_chol;

  Vec z; // work vector
};

struct GridOperator {
  /* Constructs a GridOperator instance for a 2d structured grid of size
   * global_x*global_y and a matrix representing an operator defined on that
   * grid. The parameter mat_assembler must be a function with signature `void
   * mat_assembler(Mat &, DM dm)` that assembles the matrix. Note that
   * the nonzero pattern is alredy set before this function is called, so only
   * the values have to be set (e.g., using PETSc's MatSetValuesStencil).
   */
  template <class MatAssembler>
  GridOperator(PetscInt global_x, PetscInt global_y, Coordinate lower_left,
               Coordinate upper_right, MatAssembler &&mat_assembler)
      : global_x{global_x}, global_y{global_y},
        meshwidth_x{(upper_right.x - lower_left.x) / (global_x - 1)},
        meshwidth_y{(upper_right.y - lower_left.y) / (global_y - 1)} {
    auto call = [&](auto err) { PetscCallAbort(MPI_COMM_WORLD, err); };

    const PetscInt dof_per_node = 1;
    const PetscInt stencil_width = 1;

    PetscFunctionBeginUser;
    call(DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE,
                      DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,
                      global_x,
                      global_y,
                      PETSC_DECIDE,
                      PETSC_DECIDE,
                      dof_per_node,
                      stencil_width,
                      NULL,
                      NULL,
                      &dm));
    call(DMSetUp(dm));
    call(DMDASetUniformCoordinates(
        dm, lower_left.x, upper_right.x, lower_left.y, upper_right.y, 0, 0));

    // Allocate memory for matrix and initialise non-zero pattern
    call(DMCreateMatrix(dm, &mat));

    // Call provided assembly functor to fill matrix
    call(mat_assembler(mat, dm));

    PetscFunctionReturnVoid();
  }

  ~GridOperator() {
    MatDestroy(&mat);
    DMDestroy(&dm);
  }

  void set_lowrank_factor(Mat A, Vec S, Vec S_chol) {
    this->lowrank_update = std::make_unique<LowrankUpdate<Vec>>(A, S, S_chol);
  }

  PetscErrorCode apply(Vec xin, Vec xout) const {
    PetscFunctionBeginUser;

    PetscCall(MatMult(mat, xin, xout));

    if (lowrank_update) {
      Vec tmp;
      PetscCall(VecDuplicate(xout, &tmp));
      PetscCall(lowrank_update->apply(xin, tmp));
      PetscCall(VecAXPY(xout, 1., tmp));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscInt global_x;
  PetscInt global_y;

  PetscReal meshwidth_x;
  PetscReal meshwidth_y;

  DM dm;
  Mat mat;

  std::unique_ptr<LowrankUpdate<Vec>> lowrank_update;
};
} // namespace parmgmc
