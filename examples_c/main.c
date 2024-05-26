#include "parmgmc/parmgmc.h"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscpc.h>
#include <petscsys.h>

#include <petscviewer.h>
#include <stdio.h>

static PetscErrorCode MatAssembleShiftedLaplaceFD(DM dm, PetscReal kappainv, Mat mat)
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
        vals[k]   = hinv2;
        ++k;
      }

      if (i != info.mx - 1) {
        cols[k].j = j;
        cols[k].i = i + 1;
        vals[k]   = hinv2;
        ++k;
      }

      PetscCall(MatSetValuesStencil(mat, 1, &row, k, cols, vals, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  DM da;
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  Mat mat;
  PetscCall(DMCreateMatrix(da, &mat));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 1, mat));

  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, mat, mat));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  Vec x, b;
  PetscCall(MatCreateVecs(mat, &x, &b));
  PetscCall(VecSet(b, 1));
  PetscCall(KSPSolve(ksp, b, x));

  // Clean up
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&mat));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
}
