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
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -ksp_error_if_not_converged -dm_refine 2 -ksp_monitor

// Omega = 1.2
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -mc_sor_omega 1.2 -ksp_type richardson -ksp_error_if_not_converged -dm_refine 2 -ksp_monitor

// Standalone SOR with low-rank update
// RUN: %cc %s -o %t %flags -g && %mpirun -np %NP %t -ksp_type richardson -dm_refine 3 -with_lr -ksp_error_if_not_converged -ksp_monitor

// FGMRES + SOR with low-rank update
// RUN: %cc %s -o %t %flags -g && %mpirun -np %NP %t -ksp_type fgmres -dm_refine 4 -with_lr -ksp_converged_reason -ksp_monitor

// GAMG+Gibbs with low rank update
// RUN1: %cc %s -o %t %flags -g && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type gamg  -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_pc_type gibbs -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_levels_ksp_max_it 2 -gamgmc_mg_coarse_ksp_max_it 4 %opts
/****************************************************************************/

#include <parmgmc/mc_sor.h>
#include <parmgmc/ms.h>
#include <parmgmc/obs.h>
#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>

#include <petsc.h>
#include <petsc/private/kspimpl.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscdt.h>
#include <petscfe.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscksp.h>
#include <petscvec.h>
#include <time.h>

typedef struct {
  MCSOR mc;
} *AppCtx;

/* typedef struct _SampleCtx { */
/*   Vec         mean, y, mean_exact, tmp, samples[10]; */
/*   PetscScalar mean_exact_norm; */
/*   void       *dctx; */
/* } *SampleCtx; */

/* static PetscErrorCode SampleCallback(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx) */
/* { */
/*   SampleCtx  *sctx = ctx; */
/*   Vec         mean = (*sctx)->mean, y = (*sctx)->y; */
/*   PetscScalar n; */
/*   PetscInt    maxit; */
/*   (void)rnorm; */

/*   PetscFunctionBeginUser; */
/*   PetscCall(KSPGetSolution(ksp, &y)); */
/*   PetscCall(VecScale(mean, it)); */
/*   PetscCall(VecAXPY(mean, 1., y)); */
/*   PetscCall(VecScale(mean, 1. / (it + 1))); */

/*   PetscCall(VecCopy(mean, (*sctx)->tmp)); */
/*   PetscCall(VecAXPY((*sctx)->tmp, -1, (*sctx)->mean_exact)); */
/*   PetscCall(VecNorm((*sctx)->tmp, NORM_2, &n)); */

/*   PetscCall(KSPGetTolerances(ksp, NULL, NULL, NULL, &maxit)); */
/*   if (maxit - it < 10) { PetscCall(VecCopy(y, (*sctx)->samples[9 - (maxit - it)])); } */

/*   *reason = KSP_CONVERGED_ITERATING; */
/*   PetscFunctionReturn(PETSC_SUCCESS); */
/* } */

static PetscErrorCode apply(PC pc, Vec x, Vec y)
{
  AppCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MCSORApply(ctx->mc, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM             dm;
  Mat            A, B, Aop;
  Vec            S, b, x, f;
  KSP            ksp;
  PC             pc;
  MS             ms;
  AppCtx         appctx;
  PetscBool      with_lr = PETSC_FALSE;
  const PetscInt nobs    = 3;
  PetscScalar    obs[3 * nobs], radii[nobs], obsvals[nobs];

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, &A));
  PetscCall(MSGetDM(ms, &dm));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-with_lr", &with_lr, NULL));
  if (with_lr) {
    PetscReal obsval = 1;

    obs[0] = 0.25;
    obs[1] = 0.25;
    obs[2] = 0.75;
    obs[3] = 0.75;
    obs[4] = 0.25;
    obs[5] = 0.75;

    PetscCall(PetscOptionsGetReal(NULL, NULL, "-obsval", &obsval, NULL));
    obsvals[0] = obsval;
    radii[0]   = 0.1;
    obsvals[1] = obsval;
    radii[1]   = 0.1;
    obsvals[2] = obsval;
    radii[2]   = 0.1;

    PetscCall(MakeObservationMats(dm, nobs, 1e-3, obs, radii, obsvals, &B, &S, &f));
    PetscCall(MatCreateLRC(A, B, S, B, &Aop));
  } else Aop = A;

  PetscCall(PetscNew(&appctx));
  PetscCall(MCSORCreate(Aop, 1., &appctx->mc));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, Aop, Aop));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetApply(pc, apply));
  PetscCall(PCShellSetContext(pc, appctx));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &x));

  if (with_lr) {
    b = f;
  } else {
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &f));
    PetscCall(VecSetRandom(f, NULL));
    PetscCall(MatMult(Aop, f, b));
    PetscCall(VecDestroy(&f));
  }

  PetscCall(KSPSolve(ksp, b, x));

  {
    PetscViewer viewer;
    char        filename[512] = "solution.vtu";

    PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, 512, NULL));
    PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));

    PetscCall(PetscObjectSetName((PetscObject)(x), "solution"));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(MCSORDestroy(&appctx->mc));
  PetscCall(PetscFree(appctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));
  PetscCall(MatDestroy(&A));
  if (with_lr) PetscCall(MatDestroy(&Aop));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}
