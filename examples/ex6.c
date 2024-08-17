#include "parmgmc/parmgmc.h"
#include "parmgmc/stats.h"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscvec.h>

// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t %opts

typedef struct {
  Vec     *samples;
  PetscInt idx;
  PetscInt chains;
} *SampleCtx;

static PetscErrorCode SampleCtxDestroy(void *ctx)
{
  SampleCtx sctx = ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(sctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SampleCallback(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx)
{
  (void)rnorm;
  SampleCtx sctx = ctx;
  Vec       y;

  PetscFunctionBeginUser;
  PetscCall(KSPGetSolution(ksp, &y));
  PetscCall(VecCopy(y, sctx->samples[it * sctx->chains + sctx->idx]));
  *reason = KSP_CONVERGED_ITERATING;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleMatrix(Mat *A)
{
  PetscInt      n     = 10;
  PetscScalar   kappa = 1, values[5];
  MatStencil    rowstencil, colstencil[5];
  DM            da;
  DMDALocalInfo info;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-kappa", &kappa, NULL));
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateMatrix(da, A));
  for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
    for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
      PetscInt k = 0;

      rowstencil.i = i;
      rowstencil.j = j;
      if (i != 0) {
        values[k]       = -1;
        colstencil[k].i = i - 1;
        colstencil[k].j = j;
        ++k;
      }
      if (i != info.mx - 1) {
        values[k]       = -1;
        colstencil[k].i = i + 1;
        colstencil[k].j = j;
        ++k;
      }
      if (j != 0) {
        values[k]       = -1;
        colstencil[k].i = i;
        colstencil[k].j = j - 1;
        ++k;
      }
      if (j != info.my - 1) {
        values[k]       = -1;
        colstencil[k].i = i;
        colstencil[k].j = j + 1;
        ++k;
      }
      colstencil[k].i = i;
      colstencil[k].j = j;
      values[k]       = k + kappa;
      ++k;

      PetscCall(MatSetValuesStencil(*A, 1, &rowstencil, k, colstencil, values, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(*A, NULL, "-prec_mat_view"));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscInt  chains = 1, seed = 1, samples_per_chain = 1, n;
  Mat       A;
  KSP       ksp;
  Vec      *samples, b;
  SampleCtx ctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(AssembleMatrix(&A));
  PetscCall(MatCreateVecs(A, &b, NULL));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Degrees of freedom: %d\n", n));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-chains", &chains, NULL));
  for (PetscInt i = 0; i < chains; ++i) {
    PC          pc;
    PetscRandom pr;

    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetUp(ksp));

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCGetPetscRandom(pc, &pr));
    PetscCall(PetscRandomSetSeed(pr, seed + 13 * (i + 1)));
    PetscCall(PetscRandomSeed(pr));
  }

  PetscCall(KSPGetTolerances(ksp, NULL, NULL, NULL, &samples_per_chain));
  samples_per_chain++;
  PetscCall(PetscMalloc1(samples_per_chain * chains, &samples));
  for (PetscInt i = 0; i < samples_per_chain * chains; ++i) PetscCall(VecDuplicate(b, &samples[i]));

  PetscCall(PetscNew(&ctx));
  ctx->samples = samples;
  ctx->chains  = chains;
  PetscCall(KSPSetConvergenceTest(ksp, SampleCallback, ctx, SampleCtxDestroy));

  for (PetscInt i = 0; i < chains; ++i) {
    Vec x;

    ctx->idx = i;
    PetscCall(VecDuplicate(b, &x));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(VecDestroy(&x));
  }

  {
    PetscReal *errs;

    PetscCall(PetscMalloc1(samples_per_chain, &errs));
    PetscCall(EstimateCovarianceMatErrors(A, chains, samples_per_chain, samples, errs));
    for (PetscInt i = 0; i < samples_per_chain; ++i) { PetscCall(PetscPrintf(MPI_COMM_WORLD, "%.8f\n", errs[i])); }
    PetscCall(PetscFree(errs));
  }

  for (PetscInt i = 0; i < samples_per_chain * chains; ++i) PetscCall(VecDestroy(&samples[i]));
  PetscCall(PetscFree(samples));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
}
