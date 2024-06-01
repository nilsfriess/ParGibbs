#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_gibbs.h"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <stdio.h>
#include <time.h>

typedef struct _SampleCtx {
  PetscReal norm_ex;
  Vec       mean;
} *SampleCtx;

PetscErrorCode SampleCallback(PetscInt it, Vec y, void *ctx)
{
  SampleCtx *sctx    = ctx;
  Vec        mean    = (*sctx)->mean;
  PetscReal  norm_ex = (*sctx)->norm_ex;

  PetscFunctionBeginUser;
  PetscCall(VecScale(mean, it));
  PetscCall(VecAXPY(mean, 1., y));
  PetscCall(VecScale(mean, 1. / (it + 1)));

  PetscScalar norm;
  PetscCall(VecNorm(mean, NORM_2, &norm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "%f\n", fabs(norm - norm_ex) / norm_ex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

#define N_OBS 5

PetscErrorCode MatCreateObservationMat(DM dm, PetscReal obs_coords[][2], PetscInt nobs, Mat A, Mat *O)
{
  PetscInt      ii, jj, lobs = 0, Arows;
  PetscInt     *II, *JJ, *oi;
  DMDALocalInfo info;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(nobs, &II));
  PetscCall(PetscCalloc1(nobs, &JJ));

  for (PetscInt i = 0; i < nobs; ++i) {
    PetscCall(DMDAGetLogicalCoordinate(dm, obs_coords[i][0], obs_coords[i][1], 0, &ii, &jj, NULL, NULL, NULL, NULL));
    if (ii != -1 && jj != -1) {
      II[lobs] = ii;
      JJ[lobs] = jj;
      lobs++;
    }
  }

  PetscCall(PetscMalloc1(lobs, &oi));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  for (PetscInt i = 0; i < lobs; ++i) oi[i] = II[i] + JJ[i] * info.mx;

  PetscCall(MatGetLocalSize(A, &Arows, NULL));
  PetscCall(MatCreateDense(MPI_COMM_WORLD, Arows, lobs, PETSC_DECIDE, PETSC_DECIDE, NULL, O));
  for (PetscInt i = 0; i < lobs; ++i) PetscCall(MatSetValue(*O, oi[i], i, 1., INSERT_VALUES));
  PetscCall(MatAssemblyBegin(*O, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*O, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree(oi));
  PetscCall(PetscFree(II));
  PetscCall(PetscFree(JJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  Mat A, B, ALR;
  DM  da;
  Vec Sinv, x, b, f;
  KSP ksp;
  PC  pc;
  /* PetscRandom pr; */
  SampleCtx ctx;
  PetscReal obs[N_OBS][2] = {
    {0.25, 0.25},
    {0.25, 0.75},
    {0.5,  0.5 },
    {0.75, 0.25},
    {0.75, 0.75}
  };

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 1, A));

  PetscCall(MatCreateObservationMat(da, obs, N_OBS, A, &B));
  PetscCall(MatCreateVecs(B, &Sinv, NULL));
  PetscCall(VecSet(Sinv, 1));
  PetscCall(VecReciprocal(Sinv));
  PetscCall(MatCreateLRC(A, B, Sinv, NULL, &ALR));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, ALR, ALR));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  PetscCall(KSPGetPC(ksp, &pc));
  /* PetscCall(PCGibbsGetPetscRandom(pc, &pr)); */
  /* PetscCall(PetscRandomSetSeed(pr, 1)); */
  /* PetscCall(PetscRandomSeed(pr)); */

  Vec o;
  PetscCall(MatCreateVecs(B, &o, NULL));
  PetscCall(VecSet(o, 1));
  PetscCall(MatCreateVecs(ALR, &x, &b));
  PetscCall(VecSet(b, 0));
  PetscCall(MatMultAdd(B, o, b, b));
  PetscCall(VecSet(x, 0));
  PetscCall(VecDuplicate(b, &f));
  PetscCall(MatMult(ALR, b, f));

  PetscCall(PetscNew(&ctx));
  PetscCall(MatCreateVecs(A, &(ctx->mean), NULL));
  PetscCall(VecNorm(b, NORM_2, &ctx->norm_ex));
  PetscCall(PCGibbsSetSampleCallback(pc, SampleCallback, &ctx));

  PetscCall(KSPSolve(ksp, f, x));

  // Clean up
  PetscCall(VecDestroy(&o));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&f));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx->mean));
  PetscCall(PetscFree(ctx));
  PetscCall(VecDestroy(&Sinv));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&ALR));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
}
