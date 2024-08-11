/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/problems.h"

#include <petscdmda.h>
#include <petscerror.h>

PetscErrorCode MatAssembleShiftedLaplaceFD(DM dm, PetscReal kappa, Mat mat)
{
  MatStencil row, cols[5];
  PetscReal  hinv2, vals[5];
  PetscInt   mx, my, xm, ym, xs, ys;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(dm, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(dm, &xs, &ys, 0, &xm, &ym, 0));

  hinv2 = 1. / ((mx - 1) * (mx - 1));
  for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
      row.j = j;
      row.i = i;
      if (i == 0 || j == 0 || i == mx - 1 || j == my - 1) {
        vals[0] = 1;
        PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &row, vals, INSERT_VALUES));
      } else {
        cols[0].j = j - 1;
        cols[0].i = i;
        vals[0]   = -hinv2;

        cols[1].j = j;
        cols[1].i = i - 1;
        vals[1]   = -hinv2;

        cols[2].j = j;
        cols[2].i = i;
        vals[2]   = 4 * hinv2 + kappa * kappa;

        cols[3].j = j + 1;
        cols[3].i = i;
        vals[3]   = -hinv2;

        cols[4].j = j;
        cols[4].i = i + 1;
        vals[4]   = -hinv2;

        PetscCall(MatSetValuesStencil(mat, 1, &row, 5, cols, vals, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(mat, MAT_SPD, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
