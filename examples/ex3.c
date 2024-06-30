/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Tests the multicolour SOR method.
 */

/**************************** Test specification ****************************/
// Omega = 1
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -ksp_error_if_not_converged

// Omega = 1.2
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -mc_sor_omega 1.2 -ksp_type richardson -ksp_error_if_not_converged
/****************************************************************************/

#include "parmgmc/mc_sor.h"
#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>

#include <petsc.h>
#include <petscdm.h>
#include <petscmath.h>
#include <petscpc.h>
#include <petscsystypes.h>
#include <petscksp.h>
#include <petscvec.h>

typedef struct {
  MCSOR mc;
} *AppCtx;

PetscErrorCode apply(PC pc, Vec x, Vec y)
{
  AppCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MCSORApply(ctx->mc, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM     dm;
  Mat    A;
  Vec    b, x;
  KSP    ksp;
  PC     pc;
  AppCtx appctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 3, 3, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDASetUniformCoordinates(dm, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(dm, 0, A));

  PetscCall(PetscNew(&appctx));
  PetscCall(MCSORCreate(A, 1., PETSC_TRUE, &appctx->mc));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, appctx));
  PetscCall(PCShellSetApply(pc, apply));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecSetRandom(b, NULL));

  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(MCSORDestroy(&appctx->mc));
  PetscCall(PetscFree(appctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
}
