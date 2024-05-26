#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/random/ziggurat.h"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscpc.h>
#include <petscsys.h>

#include <petscviewer.h>
#include <cstdio>

static PetscErrorCode MatAssembleLaplaceFD(DM dm, Mat mat)
{
  PetscInt      k;
  MatStencil    row, cols[5];
  PetscReal     vals[5];
  DMDALocalInfo info;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(dm, &info));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      row.j = j;
      row.i = i;

      k = 0;

      if (j != 0) {
        cols[k].j = j - 1;
        cols[k].i = i;
        vals[k]   = -1;
        ++k;
      }

      if (i != 0) {
        cols[k].j = j;
        cols[k].i = i - 1;
        vals[k]   = -1;
        ++k;
      }

      cols[k].j = j;
      cols[k].i = i;
      vals[k]   = 4;
      ++k;

      if (j != info.my - 1) {
        cols[k].j = j + 1;
        cols[k].i = i;
        vals[k]   = -1;
        ++k;
      }

      if (i != info.mx - 1) {
        cols[k].j = j;
        cols[k].i = i + 1;
        vals[k]   = -1;
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
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(PCRegister("hogwild", PCCreate_Hogwild));
  PetscCall(PCRegister("gibbs", PCCreate_Gibbs));
  PetscCall(PetscRandomRegister("ziggurat", PetscRandomCreate_Ziggurat));

  DM da;
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  Mat mat;
  PetscCall(DMCreateMatrix(da, &mat));
  PetscCall(MatAssembleLaplaceFD(da, mat));

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
