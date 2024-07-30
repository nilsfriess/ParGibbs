/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/obs.h"

#include <petscdm.h>
#include <petscis.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

typedef struct {
  Mat              M;
  PetscReal        radius, vol;
  const PetscReal *coords;
} *ObsCtx;

static PetscErrorCode VolumeOfSphere(DM dm, PetscScalar r, PetscScalar *v)
{
  PetscInt cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == 2 || cdim == 3, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only dim=2 and dim=3 supported");
  if (cdim == 2) *v = PETSC_PI * r * r;
  else *v = 4 * PETSC_PI / 3. * r * r * r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode f(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)time;
  (void)Nc;
  ObsCtx      octx = ctx;
  PetscScalar diff = 0;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < dim; ++i) diff += PetscSqr(x[i] - octx->coords[i]);
  if (diff < PetscSqr(octx->radius)) *u = 1 / octx->vol;
  else *u = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddObservationToVec_Plex(DM dm, Vec vec, ObsCtx ctx)
{
  Vec   u;
  void *octx = ctx;

  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {f};

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(VolumeOfSphere(dm, ctx->radius, &ctx->vol));
  PetscCall(DMProjectFunction(dm, 0, funcs, &octx, INSERT_VALUES, u));
  PetscCall(MatMult(ctx->M, u, vec));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TODO: This does not do exactly the same as the version for Plex, this one basically assumes that the vector is zeroed.
/* static PetscErrorCode AddObservationToVec_DA(DM dm, const PetscScalar *p, PetscScalar r, Vec vec) */
/* { */
/*   PetscInt           cdim, ncoords; */
/*   Vec                allCoordsVec; */
/*   const PetscScalar *allCoords; */
/*   PetscScalar        vol; */

/*   PetscFunctionBeginUser; */
/*   PetscCall(VolumeOfSphere(dm, r, &vol)); */
/*   PetscCall(PetscPrintf(MPI_COMM_WORLD, "Volume: %.5f\n", vol)); */
/*   PetscCall(DMGetCoordinateDim(dm, &cdim)); */
/*   PetscCall(DMGetCoordinatesLocal(dm, &allCoordsVec)); */
/*   PetscCall(VecGetLocalSize(allCoordsVec, &ncoords)); */
/*   PetscCall(VecGetArrayRead(allCoordsVec, &allCoords)); */

/*   for (PetscInt i = 0; i < ncoords; i += cdim) { */
/*     PetscScalar dist = 0; */
/*     for (PetscInt c = 0; c < cdim; ++c) dist += PetscSqr(p[c] - allCoords[i + c]); */
/*     dist = PetscSqrtReal(dist); */
/*     if (dist < r) { */
/*       PetscInt idx = i / cdim; */
/*       PetscCall(VecSetValueLocal(vec, idx, 1. / vol, INSERT_VALUES)); */
/*     } */
/*   } */
/*   PetscCall(VecRestoreArrayRead(allCoordsVec, &allCoords)); */
/*   PetscFunctionReturn(PETSC_SUCCESS); */
/* } */

/* PetscErrorCode AddObservationToVec(DM dm, const PetscScalar *p, PetscScalar r, Vec vec) */
/* { */
/*   DMType    type; */
/*   PetscBool flag_plex, flag_dmda; */

/*   PetscFunctionBeginUser; */
/*   PetscCall(DMGetType(dm, &type)); */

/*   PetscCall(PetscStrcmp(type, DMDA, &flag_dmda)); */
/*   if (flag_dmda) PetscCall(AddObservationToVec_DA(dm, p, r, vec)); */

/*   PetscCall(PetscStrcmp(type, DMPLEX, &flag_plex)); */
/*   if (flag_plex) PetscCall(AddObservationToVec_Plex(dm, p, r, vec)); */

/*   PetscCheck(flag_dmda || flag_plex, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only DMDA and DMPlex supported"); */
/*   PetscFunctionReturn(PETSC_SUCCESS); */
/* } */

/**
   @brief Construct the "observation" matrix that can be used with the Gibbs and MGMC sampler to simulate Bayesian linear inverse problems. The observations correspond to constant measurments in balls around given points in the domain.

   # Input Parameters
   - `dm` - The DM on which the observations are defined (currently, only DMPLEX can be used)
   - `nobs` - The number of observations
   - `sigma2` - The noise variance (current the same variance is used for all observations)
   - `coords` - An array of length `dim` x `nobs` that contains the coordinates of the centres of the observations, ordered as {x_0, y_0, x_1, y_1, ...} in 2D and analogously in 3D.
   - `radii` - The radii of the balls that correspond to the measurements (must be of length `nobs`).

   # Output parameters
   - `B` - The observation matrix of size `# grid points` x `nobs`
   - `S` - The inverse diagonal noise matrix, represented as a vector
   - `f` - The "right hand side" vector that cen be used in the Gibbs/MGMC sampler and which will lead to the correct posterior mean

   # TODOs
   - Allow DMDA as DM
   - We do some work twice, this should be avoided
 */

PetscErrorCode MakeObservationMats(DM dm, PetscInt nobs, PetscScalar sigma2, const PetscScalar *coords, PetscScalar *radii, const PetscScalar *obsvals, Mat *B, Vec *S, Vec *f)
{
  Vec      meas, g, y;
  PetscInt lsize, gsize, cdim;
  ObsCtx   ctx;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMCreateGlobalVector(dm, &g));
  PetscCall(VecGetSize(g, &gsize));
  PetscCall(VecGetLocalSize(g, &lsize));
  PetscCall(VecDestroy(&g));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)dm), lsize, PETSC_DECIDE, gsize, nobs, NULL, B));
  PetscCall(MatCreateVecs(*B, S, f));
  PetscCall(VecSet(*S, 1. / sigma2));
  PetscCall(VecDuplicate(*S, &y));

  PetscCall(DMCreateGlobalVector(dm, &meas));
  PetscCall(PetscNew(&ctx));
  PetscCall(DMCreateMassMatrix(dm, dm, &ctx->M));
  for (PetscInt i = 0; i < nobs; ++i) {
    PetscCall(VecZeroEntries(meas));
    ctx->coords = &(coords[cdim * i]);
    ctx->radius = radii[i];
    PetscCall(AddObservationToVec_Plex(dm, meas, ctx));
    PetscCall(MatDenseGetColumnVec(*B, i, &g));
    PetscCall(VecCopy(meas, g));
    PetscCall(MatDenseRestoreColumnVec(*B, i, &g));
    PetscCall(VecSetValue(y, i, obsvals[i], INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(VecPointwiseMult(y, y, *S));
  PetscCall(MatMult(*B, y, *f));

  PetscCall(MatDestroy(&ctx->M));
  PetscCall(PetscFree(ctx));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&meas));
  PetscFunctionReturn(PETSC_SUCCESS);
}
