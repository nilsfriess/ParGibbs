#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/random/ziggurat.h"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscksp.h>
#include <petscpc.h>
#include <petscsys.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  printf("Start\n");

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PCRegister("hogwild", PCCreate_Hogwild));
  PetscCall(PetscRandomRegister("ziggurat", PetscRandomCreate_Ziggurat));

  DM da;
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  Mat mat;
  PetscCall(DMCreateMatrix(da, &mat));
  PetscCall(MatShift(mat, 1));

  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, mat, mat));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  Vec x, b;
  PetscCall(MatCreateVecs(mat, &x, &b));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(PetscFinalize());
}
