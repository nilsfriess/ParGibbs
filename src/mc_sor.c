#include "parmgmc/mc_sor.h"

#include <stdbool.h>

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

PetscErrorCode ContextDestroy_MPIAIJ(void *ctx)
{
  SORCtx_MPIAIJ *mpictx = ctx;
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < mpictx->ncolors; ++i) {
    PetscCall(VecScatterDestroy(&mpictx->scatters[i]));
    PetscCall(VecDestroy(&mpictx->ghostvecs[i]));
  }
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ContextDestroy_LRC(void *ctx)
{
  SORCtx_LRC    *lrcctx = ctx;
  SORCtx_MPIAIJ *mpictx = lrcctx->basectx;
  PetscFunctionBeginUser;
  PetscCall(ContextDestroy_MPIAIJ(mpictx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultiColorSOR_MPIAIJ(Mat mat, const PetscInt *diagptrs, Vec idiag, Vec b, PetscReal omega, ISColoring ic, void *sor_ctx, Vec y)
{
  Mat              ad, ao; // Local and off-processor parts of mat
  PetscInt         nind, gcnt, ncolors;
  const PetscInt  *rowptr, *colptr, *bRowptr, *bColptr, *rowind;
  const PetscReal *idiagarr, *barr, *ghostarr;
  PetscReal       *matvals, *bMatvals, *yarr;
  IS              *isc;
  SORCtx_MPIAIJ   *ctx = sor_ctx;

  PetscFunctionBeginUser;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &ad, &ao, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(ad, &rowptr, &colptr, &matvals, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(ao, &bRowptr, &bColptr, &bMatvals, NULL));

  PetscCall(ISColoringGetIS(ic, PETSC_USE_POINTER, &ncolors, &isc));

  PetscCall(VecGetArrayRead(idiag, &idiagarr));
  PetscCall(VecGetArrayRead(b, &barr));

  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscCall(VecScatterBegin(ctx->scatters[color], y, ctx->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx->scatters[color], y, ctx->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayRead(ctx->ghostvecs[color], &ghostarr));

    PetscCall(ISGetLocalSize(isc[color], &nind));
    PetscCall(ISGetIndices(isc[color], &rowind));
    PetscCall(VecGetArray(y, &yarr));

    gcnt = 0;
    for (PetscInt i = 0; i < nind; ++i) {
      PetscReal sum = 0;

      for (PetscInt k = rowptr[rowind[i]]; k < diagptrs[rowind[i]]; ++k) sum -= matvals[k] * yarr[colptr[k]];
      for (PetscInt k = diagptrs[rowind[i]] + 1; k < rowptr[rowind[i] + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];
      for (PetscInt k = bRowptr[rowind[i]]; k < bRowptr[rowind[i] + 1]; ++k) sum -= bMatvals[k] * ghostarr[gcnt++];

      yarr[rowind[i]] = (1 - omega) * yarr[rowind[i]] + omega * idiagarr[rowind[i]] * (sum + barr[rowind[i]]);
    }

    PetscCall(VecRestoreArray(y, &yarr));
    PetscCall(VecRestoreArrayRead(ctx->ghostvecs[color], &ghostarr));
    PetscCall(ISRestoreIndices(isc[color], &rowind));
  }

  PetscCall(VecRestoreArrayRead(b, &barr));
  PetscCall(VecRestoreArrayRead(idiag, &idiagarr));
  PetscCall(ISColoringRestoreIS(ic, PETSC_USE_POINTER, &isc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultiColorSOR_SEQAIJ(Mat mat, const PetscInt *diagptrs, Vec idiag, Vec b, PetscReal omega, ISColoring ic, void *sor_ctx, Vec y)
{
  (void)sor_ctx;
  PetscInt         nind, ncolors;
  const PetscInt  *rowptr, *colptr, *rowind;
  const PetscReal *idiagarr, *barr;
  PetscReal       *matvals, *yarr;
  IS              *isc;

  PetscFunctionBeginUser;
  PetscCall(MatSeqAIJGetCSRAndMemType(mat, &rowptr, &colptr, &matvals, NULL));

  PetscCall(ISColoringGetIS(ic, PETSC_USE_POINTER, &ncolors, &isc));

  PetscCall(VecGetArrayRead(idiag, &idiagarr));
  PetscCall(VecGetArrayRead(b, &barr));

  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscCall(ISGetLocalSize(isc[color], &nind));
    PetscCall(ISGetIndices(isc[color], &rowind));
    PetscCall(VecGetArray(y, &yarr));

    for (PetscInt i = 0; i < nind; ++i) {
      PetscReal sum = 0;

      for (PetscInt k = rowptr[rowind[i]]; k < diagptrs[rowind[i]]; ++k) sum -= matvals[k] * yarr[colptr[k]];
      for (PetscInt k = diagptrs[rowind[i]] + 1; k < rowptr[rowind[i] + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];

      yarr[rowind[i]] = (1 - omega) * yarr[rowind[i]] + omega * idiagarr[rowind[i]] * (sum + barr[rowind[i]]);
    }

    PetscCall(VecRestoreArray(y, &yarr));
    PetscCall(ISRestoreIndices(isc[color], &rowind));
  }

  PetscCall(VecRestoreArrayRead(b, &barr));
  PetscCall(VecRestoreArrayRead(idiag, &idiagarr));
  PetscCall(ISColoringRestoreIS(ic, PETSC_USE_POINTER, &isc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultiColorSOR_LRC(Mat mat, const PetscInt *diagptrs, Vec idiag, Vec b, PetscReal omega, ISColoring ic, void *sor_ctx, Vec y)
{
  SORCtx_LRC *ctx = sor_ctx;
  Mat         A;

  PetscFunctionBeginUser;
  PetscCall(MatLRCGetMats(mat, &A, NULL, NULL, NULL));
  PetscCall(ctx->basesor(A, diagptrs, idiag, b, omega, ic, ctx->basectx, y));
  // TODO: Handle low rank part
  PetscFunctionReturn(PETSC_SUCCESS);
}
