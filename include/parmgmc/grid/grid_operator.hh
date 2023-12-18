#pragma once

#include <iostream>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>

#include "parmgmc/common/types.hh"

namespace parmgmc {
struct GridOperator {
  /* Constructs a GridOperator instance for a 2d structured grid of size
   * global_x*global_y and a matrix representing an operator defined on that
   * grid. The parameter mat_assembler must be a function with signature `void
   * mat_assembler(Mat &, const Grid &)` that assembles the matrix. Note that
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

  PetscInt global_x;
  PetscInt global_y;

  PetscReal meshwidth_x;
  PetscReal meshwidth_y;

  DM dm;
  Mat mat;
};
} // namespace parmgmc
