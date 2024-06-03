/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/problems.h"

#include <petscdmda.h>

PetscErrorCode MatAssembleShiftedLaplaceFD(DM dm, PetscReal kappainv, Mat mat)
{
  PetscInt      k;
  MatStencil    row, cols[5];
  PetscReal     hinv2, vals[5];
  DMDALocalInfo info;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(dm, &info));
  hinv2 = 1. / ((info.mx - 1) * (info.mx - 1));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      row.j = j;
      row.i = i;

      k = 0;

      if (j != 0) {
        cols[k].j = j - 1;
        cols[k].i = i;
        vals[k]   = -hinv2;
        ++k;
      }

      if (i != 0) {
        cols[k].j = j;
        cols[k].i = i - 1;
        vals[k]   = -hinv2;
        ++k;
      }

      cols[k].j = j;
      cols[k].i = i;
      vals[k]   = 4 * hinv2 + 1. / (kappainv * kappainv);
      ++k;

      if (j != info.my - 1) {
        cols[k].j = j + 1;
        cols[k].i = i;
        vals[k]   = -hinv2;
        ++k;
      }

      if (i != info.mx - 1) {
        cols[k].j = j;
        cols[k].i = i + 1;
        vals[k]   = -hinv2;
        ++k;
      }

      PetscCall(MatSetValuesStencil(mat, 1, &row, k, cols, vals, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
