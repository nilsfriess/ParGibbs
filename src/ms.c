/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/ms.h"
#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/parmgmc.h"

#include <petscdmlabel.h>
#include <petscdmplex.h>
#include <petscdm.h>
#include <petscdmtypes.h>
#include <petscds.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsnes.h>
#include <petscsys.h>
#include <petscvec.h>

typedef struct _MSCtx {
  MPI_Comm    comm;
  DM          dm;
  Mat         A, Id;
  KSP         ksp, addksp;
  Vec         b, mean, var;
  PetscInt    alpha, cnt;
  PetscScalar kappa, nu, tau;
  PetscBool   addksp_setup_called, save_samples, assemble_only;
  Vec        *samples;
} *MSCtx;

/** @file ms.c
    @brief A sampler to simulate Matérn random fields.

    # Notes
    This class encapsulates a sampler that can generate random samples from
    zero mean Whittle-Matérn fields using the Multigrid Monte Carlo method.

    Internally it uses the PCGAMGMC implementation.
 */

PetscErrorCode MSDestroy(MS *ms)
{
  MSCtx ctx = (*ms)->ctx;

  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->b));
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(KSPDestroy(&ctx->ksp));
  PetscCall(KSPDestroy(&ctx->addksp));
  PetscCall(VecDestroy(&ctx->mean));
  PetscCall(VecDestroy(&ctx->var));
  PetscCall(DMDestroy(&ctx->dm));

  PetscCall(PetscFree(ctx));
  PetscCall(PetscFree(*ms));
  ms = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetDM(MS ms, DM dm)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  ctx->dm = dm;
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = constants[0] * constants[0] * u[0];
}

static void f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  if (numConstants > 0) g0[0] = constants[0] * constants[0];
}

static void g3(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

#pragma GCC diagnostic pop

static PetscErrorCode MS_AssembleMat(MS ms)
{
  MSCtx          ctx = ms->ctx;
  SNES           snes;
  PetscInt       dim;
  PetscBool      simplex;
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  DM             cdm;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(SNESCreate(ctx->comm, &snes));
  PetscCall(SNESSetDM(snes, ctx->dm));
  PetscCall(SNESSetLagPreconditioner(snes, -1));
  PetscCall(SNESSetLagJacobian(snes, -2));

  PetscCall(DMGetDimension(ctx->dm, &dim));
  PetscCall(DMPlexIsSimplex(ctx->dm, &simplex));
  PetscCall(PetscFECreateLagrange(ctx->comm, dim, 1, simplex, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(ctx->dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(ctx->dm));
  PetscCall(DMGetDS(ctx->dm, &ds));
  if (ctx->kappa != 0) PetscCall(PetscDSSetConstants(ds, 1, &ctx->kappa));
  PetscCall(PetscDSSetResidual(ds, 0, f0, f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0, NULL, NULL, g3));

  PetscCall(DMGetLabel(ctx->dm, "marker", &label));
  if (!label) {
    PetscCall(DMCreateLabel(ctx->dm, "boundary"));
    PetscCall(DMGetLabel(ctx->dm, "boundary", &label));
    PetscCall(DMPlexMarkBoundaryFaces(ctx->dm, PETSC_DETERMINE, label));
  }
  PetscCall(DMPlexLabelComplete(ctx->dm, label));
  PetscCall(DMAddBoundary(ctx->dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, NULL, NULL, NULL, NULL));

  cdm = ctx->dm;
  while (cdm) {
    /* PetscCall(DMCreateLabel(cdm, "boundary")); */
    /* PetscCall(DMGetLabel(cdm, "boundary", &label)); */
    /* PetscCall(DMPlexMarkBoundaryFaces(cdm, PETSC_DETERMINE, label)); */
    PetscCall(DMCopyDisc(ctx->dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }

  PetscCall(DMPlexSetSNESLocalFEM(ctx->dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetUp(snes));
  PetscCall(DMCreateMatrix(ctx->dm, &ctx->A));
  PetscCall(MatCreateVecs(ctx->A, &ctx->b, NULL));
  PetscCall(SNESComputeJacobian(snes, ctx->b, ctx->A, ctx->A));

  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSample(MS ms, Vec x)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCall(KSPSolve(ctx->ksp, ctx->b, x));

  if (ctx->alpha != 1) {
    if (!ctx->addksp_setup_called) {
      PetscCall(KSPCreate(ctx->comm, &ctx->addksp));
      PetscCall(KSPSetOperators(ctx->addksp, ctx->A, ctx->A));
      PetscCall(KSPSetOptionsPrefix(ctx->addksp, "matern_"));
      PetscCall(KSPSetFromOptions(ctx->addksp));
      PetscCall(KSPSetUp(ctx->addksp));
      ctx->addksp_setup_called = PETSC_TRUE;
    }

    PetscInt addsolves = ctx->alpha % 2 == 0 ? ctx->alpha / 2 : (ctx->alpha - 1) / 2;
    for (PetscInt i = 0; i < addsolves; ++i) PetscCall(KSPSolve(ctx->addksp, x, x));
  }

  if (PetscAbsReal(ctx->nu) > 1e-10) PetscCall(VecScale(x, ctx->tau));

  if (ctx->save_samples) {
    PetscCall(VecCopy(x, ctx->samples[ctx->cnt]));
    ctx->cnt++;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSBeginSaveSamples(MS ms, PetscInt n)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  if (ctx->samples) PetscCall(PetscFree(ctx->samples));
  PetscCall(PetscMalloc1(n, &ctx->samples));
  for (PetscInt i = 0; i < n; ++i) { PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->samples[i])); }

  ctx->save_samples = PETSC_TRUE;
  ctx->cnt          = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MS_ComputeMeanAndVar(MS ms)
{
  MSCtx ctx = ms->ctx;
  Vec   tmp;

  PetscFunctionBeginUser;
  if (!ctx->mean) {
    PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->mean));
    PetscCall(VecDuplicate(ctx->mean, &ctx->var));
  }

  PetscCall(VecZeroEntries(ctx->mean));
  for (PetscInt i = 0; i < ctx->cnt; ++i) PetscCall(VecAXPY(ctx->mean, 1. / ctx->cnt, ctx->samples[i]));

  PetscCall(VecZeroEntries(ctx->var));
  PetscCall(VecDuplicate(ctx->var, &tmp));
  for (PetscInt i = 0; i < ctx->cnt; ++i) {
    PetscCall(VecCopy(ctx->samples[i], tmp));
    PetscCall(VecAXPY(tmp, -1, ctx->mean));
    PetscCall(VecPointwiseMult(tmp, tmp, tmp));
    PetscCall(VecAXPY(ctx->var, 1. / (ctx->cnt - 1), tmp));
  }

  PetscCall(PetscObjectSetName((PetscObject)ctx->mean, "mean"));
  PetscCall(PetscObjectSetName((PetscObject)ctx->var, "var"));

  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSEndSaveSamples(MS ms, PetscInt n, const Vec **samples)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCall(MS_ComputeMeanAndVar(ms));
  ctx->save_samples = PETSC_FALSE;

  if (!samples) {
    for (PetscInt i = 0; i < n; ++i) PetscCall(VecDestroy(&ctx->samples[i]));
    PetscCall(PetscFree(ctx->samples));
  } else *samples = ctx->samples;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetMeanAndVar(MS ms, Vec *mean, Vec *var)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  if (mean) *mean = ctx->mean;
  if (var) *var = ctx->var;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetAlpha(MS ms, PetscInt alpha)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCheck(alpha >= 1, ctx->comm, PETSC_ERR_SUP, "alpha must be >= 1");
  ctx->alpha = alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetKappa(MS ms, PetscScalar kappa)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCheck(kappa >= 0, ctx->comm, PETSC_ERR_SUP, "Range parameter kappa must be nonnegative");
  ctx->kappa = kappa;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetDM(MS ms, DM *dm)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  *dm = ctx->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMeshDefault(MPI_Comm comm, DM *dm)
{
  DM       distdm;
  PetscInt faces[2];

  PetscFunctionBeginUser;
  faces[0] = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-box_faces", &faces[0], NULL));
  faces[1] = faces[0];
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMPlexDistribute(*dm, 0, NULL, &distdm));
  if (distdm) {
    PetscCall(DMDestroy(dm));
    *dm = distdm;
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetUp(MS ms)
{
  MSCtx       ctx = ms->ctx;
  PC          pc, mgmc;
  PetscInt    levels, dim;
  PetscScalar sigma2;

  PetscFunctionBeginUser;
  if (!ctx->dm) PetscCall(CreateMeshDefault(ctx->comm, &ctx->dm));
  PetscCall(MS_AssembleMat(ms));

  if (!ctx->assemble_only) {
    PetscCall(KSPCreate(ctx->comm, &ctx->ksp));
    PetscCall(KSPSetOperators(ctx->ksp, ctx->A, ctx->A));
    PetscCall(KSPSetType(ctx->ksp, KSPRICHARDSON));
    PetscCall(KSPSetDM(ctx->ksp, ctx->dm));
    PetscCall(KSPSetDMActive(ctx->ksp, PETSC_FALSE));
    PetscCall(KSPGetPC(ctx->ksp, &pc));
    PetscCall(PCSetType(pc, "gamgmc"));
    PetscCall(PCGAMGGetInternalPC(pc, &mgmc));
    PetscCall(KSPSetOptionsPrefix(ctx->ksp, "ms_"));
    PetscCall(KSPSetFromOptions(ctx->ksp));
    PetscCall(KSPSetUp(ctx->ksp));
    PetscCall(KSPSetTolerances(ctx->ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));

    PetscCall(PCMGGetLevels(mgmc, &levels));
    for (PetscInt i = 0; i < levels; ++i) {
      KSP ksps;
      PC  pcs;
      PetscCall(PCMGGetSmoother(mgmc, i, &ksps));
      PetscCall(KSPSetType(ksps, KSPRICHARDSON));
      PetscCall(KSPGetPC(ksps, &pcs));
      PetscCall(PCSetType(pcs, PCGIBBS));
      PetscCall(KSPSetTolerances(ksps, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 4));
    }

    PetscCall(DMGetDimension(ctx->dm, &dim));
    ctx->nu  = ctx->alpha - dim / 2;
    sigma2   = PetscTGamma(ctx->nu) / (PetscTGamma(ctx->alpha) * PetscPowScalarReal(4 * PETSC_PI, dim / 2.) * PetscPowScalarReal(ctx->kappa, ctx->nu));
    ctx->tau = PetscSqrtReal(sigma2);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetPrecisionMatrix(MS ms, Mat *A)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  *A = ctx->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetAssemblyOnly(MS ms, PetscBool flag)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  ctx->assemble_only = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetFromOptions(MS ms)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscOptionsBegin(ctx->comm, NULL, "Options for the Matern sampler", NULL);
  PetscCall(PetscOptionsInt("-matern_alpha", "Set power of Matern precision operator", NULL, ctx->alpha, &ctx->alpha, NULL));
  PetscCall(PetscOptionsReal("-matern_kappa", "Set the range parameter of the Matern covariance", NULL, ctx->kappa, &ctx->kappa, NULL));
  PetscCall(PetscOptionsBool("-matern_assemble_only", "If true, does not setup the sampler, only assembles the precision matrix", NULL, ctx->assemble_only, &ctx->assemble_only, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSCreate(MPI_Comm comm, MS *ms)
{
  MSCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(ms));
  PetscCall(PetscNew(&ctx));
  (*ms)->ctx = ctx;

  ctx->dm                  = NULL;
  ctx->comm                = comm;
  ctx->alpha               = 1;
  ctx->kappa               = 1;
  ctx->addksp_setup_called = PETSC_FALSE;
  ctx->assemble_only       = PETSC_FALSE;
  ctx->samples             = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
