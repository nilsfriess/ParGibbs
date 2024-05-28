#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/coloring.h"
#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petscis.h>
#include <petscistypes.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsf.h>
#include <petscsys.h>
#include <petscpc.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/matimpl.h>
#include <petscvec.h>

#include <stdbool.h>
#include <string.h>

typedef struct {
  PetscRandom prand;
  ISColoring  ic;
  Vec         idiag, sqrtdiag; // Both include omega
  PetscInt   *diagptrs;        // Index of the diagonal entry in the csr array for each row
  PetscReal   omega;

  PetscErrorCode (*sor)(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec); // The multicolor SOR implementation (can be different for different matrix types)
  void *sor_ctx;                                                                              // Context that can be passed to the multicolor SOR routine
  PetscErrorCode (*sor_ctx_destroy)(void **);
} PC_Gibbs;

static PetscErrorCode PCDestroy_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  if (pg->sor_ctx_destroy) PetscCall(pg->sor_ctx_destroy(&pg->sor_ctx));
  PetscCall(PetscRandomDestroy(&pg->prand));
  PetscCall(ISColoringDestroy(&pg->ic));
  PetscCall(VecDestroy(&pg->idiag));
  PetscCall(VecDestroy(&pg->sqrtdiag));
  PetscCall(PetscFree(pg->diagptrs));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TODO: The addition of random noise to the rhs is missing here
static PetscErrorCode PCApply_Gibbs(PC pc, Vec x, Vec y)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(MULTICOL_SOR, pc, x, y, NULL));
  PetscCall(pg->sor(pc->pmat, pg->diagptrs, pg->idiag, x, pg->omega, pg->ic, pg->sor_ctx, y));
  PetscCall(PetscLogEventEnd(MULTICOL_SOR, pc, x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_Gibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(VecSetRandom(w, pg->prand));
    PetscCall(VecAXPY(w, 1., b)); // TODO: For omega != 1 this is not doing the right thing
    PetscCall(PetscLogEventBegin(MULTICOL_SOR, pc, b, y, NULL));
    PetscCall(pg->sor(pc->pmat, pg->diagptrs, pg->idiag, w, pg->omega, pg->ic, pg->sor_ctx, y));
    PetscCall(PetscLogEventEnd(MULTICOL_SOR, pc, b, y, NULL));
  }
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateScatters(Mat mat, ISColoring isc, SORCtx_MPIAIJ *ctx)
{
  PetscInt ncolors;
  IS      *iss;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(ctx));
  PetscCall(ISColoringGetIS(isc, PETSC_USE_POINTER, &ncolors, &iss));
  PetscCall(PetscMalloc1(ncolors, &(*ctx)->scatters));
  PetscCall(PetscMalloc1(ncolors, &(*ctx)->ghostvecs));
  (*ctx)->ncolors = ncolors;

  Mat             ao; // off-processor part of matrix
  const PetscInt *colmap, *rowptr, *colptr;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, NULL, &ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(ao, &rowptr, &colptr, NULL, NULL));

  Vec      gvec;
  PetscInt localRows, globalRows;
  PetscCall(MatGetSize(mat, &globalRows, NULL));
  PetscCall(MatGetLocalSize(mat, &localRows, NULL));
  PetscCall(VecCreateMPIWithArray(MPI_COMM_WORLD, 1, localRows, globalRows, NULL, &gvec));

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // First count the total number of off-processor values for each color
  PetscInt *nTotalOffProc;
  PetscCall(PetscCalloc1(ncolors, &nTotalOffProc));
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(ISGetLocalSize(iss[color], &nCurCol));
    PetscCall(ISGetIndices(iss[color], &curidxs));
    for (PetscInt i = 0; i < nCurCol; ++i) nTotalOffProc[color] += rowptr[curidxs[i] + 1] - rowptr[curidxs[i]];
    PetscCall(ISRestoreIndices(iss[color], &curidxs));
  }

  // Now we again loop over all colors and create the required VecScatters
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt       *offProcIdx;
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(PetscMalloc1(nTotalOffProc[color], &offProcIdx));
    PetscCall(ISGetLocalSize(iss[color], &nCurCol));
    PetscCall(ISGetIndices(iss[color], &curidxs));
    PetscInt cnt = 0;
    for (PetscInt i = 0; i < nCurCol; ++i)
      for (PetscInt k = rowptr[curidxs[i]]; k < rowptr[curidxs[i] + 1]; ++k) offProcIdx[cnt++] = colmap[colptr[k]];
    PetscCall(ISRestoreIndices(iss[color], &curidxs));

    IS is;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nTotalOffProc[color], offProcIdx, PETSC_USE_POINTER, &is));
    PetscCall(VecCreateSeq(MPI_COMM_SELF, nTotalOffProc[color], &(*ctx)->ghostvecs[color]));
    PetscCall(VecScatterCreate(gvec, is, (*ctx)->ghostvecs[color], NULL, &(*ctx)->scatters[color]));
    PetscCall(ISDestroy(&is));
    PetscCall(PetscFree(offProcIdx));
  }

  PetscCall(ISColoringRestoreIS(isc, PETSC_USE_POINTER, &iss));
  PetscCall(PetscFree(nTotalOffProc));
  PetscCall(VecDestroy(&gvec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalPointers(Mat mat, PetscInt **diagptrs)
{
  PetscInt rows;
  PetscFunctionBeginUser;
  PetscCall(MatGetSize(mat, &rows, NULL));
  PetscCall(PetscMalloc1(rows, diagptrs));

  // TODO: Make sure the matrix does not contain any zeros on the diagonal
  const PetscInt *i, *j;
  PetscReal      *a;
  PetscCall(MatSeqAIJGetCSRAndMemType(mat, &i, &j, &a, NULL));
  for (PetscInt row = 0; row < rows; ++row) {
    for (PetscInt k = i[row]; k < i[row + 1]; ++k) {
      PetscInt col = j[k];
      if (col == row) (*diagptrs)[row] = k;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;
  Mat       A;
  MatType   type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(pc->pmat, &type));
  if (strcmp(type, MATMPIAIJ) == 0) {
    SORCtx_MPIAIJ ctx;
    PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &A, NULL, NULL));
    PetscCall(MatCreateISColoring_AIJ(pc->pmat, &pg->ic));
    PetscCall(MatCreateScatters(pc->pmat, pg->ic, &ctx));
    pg->sor_ctx         = ctx;
    pg->sor             = MatMultiColorSOR_MPIAIJ;
    pg->sor_ctx_destroy = ContextDestroy_MPIAIJ;
  } else if (strcmp(type, MATSEQAIJ) == 0) {
    A = pc->pmat;
    PetscCall(MatCreateISColoring_AIJ(pc->pmat, &pg->ic));
    pg->sor = MatMultiColorSOR_SEQAIJ;
  } else if (strcmp(type, MATLRC) == 0) {
    SORCtx_LRC ctx;
    MatType    Atype;

    PetscCall(MatLRCGetMats(pc->pmat, &A, NULL, NULL, NULL));
    PetscCall(MatCreateISColoring_AIJ(A, &pg->ic));
    PetscCall(MatGetType(A, &Atype));
    PetscCall(PetscNew(&ctx));

    if (strcmp(Atype, MATMPIAIJ) == 0) {
      SORCtx_MPIAIJ mpictx;
      PetscCall(MatCreateScatters(A, pg->ic, &mpictx));
      ctx->basectx = mpictx;
      ctx->basesor = MatMultiColorSOR_MPIAIJ;
    } else if (strcmp(Atype, MATSEQAIJ) == 0) {
      ctx->basectx = NULL;
      ctx->basesor = MatMultiColorSOR_SEQAIJ;
    } else {
      PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Matrix type not supported");
    }

    pg->sor             = MatMultiColorSOR_LRC;
    pg->sor_ctx         = ctx;
    pg->sor_ctx_destroy = ContextDestroy_LRC;
  } else {
    PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Matrix type not supported");
  }

  PetscCall(MatCreateVecs(A, &pg->idiag, NULL));
  PetscCall(VecDuplicate(pg->idiag, &pg->sqrtdiag));

  PetscCall(MatGetDiagonal(A, pg->sqrtdiag));
  PetscCall(VecSqrtAbs(pg->sqrtdiag));
  PetscCall(VecScale(pg->sqrtdiag, PetscSqrtReal((2 - pg->omega) / pg->omega)));

  PetscCall(MatGetDiagonal(A, pg->idiag));
  PetscCall(VecReciprocal(pg->idiag));
  PetscCall(VecScale(pg->idiag, pg->omega));

  PetscCall(MatGetDiagonalPointers(A, &pg->diagptrs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Gibbs(PC pc)
{
  PC_Gibbs *gibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&gibbs));
  gibbs->omega           = 1.0; // TODO: Allow user to change omega
  gibbs->sor_ctx         = NULL;
  gibbs->sor_ctx_destroy = NULL;

  // TODO: Allow user to pass own PetscRandom
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &gibbs->prand));
  PetscCall(PetscRandomSetType(gibbs->prand, PARMGMC_ZIGGURAT));

  pc->data                 = gibbs;
  pc->ops->setup           = PCSetUp_Gibbs;
  pc->ops->destroy         = PCDestroy_Gibbs;
  pc->ops->applyrichardson = PCApplyRichardson_Gibbs;
  pc->ops->apply           = PCApply_Gibbs;
  PetscFunctionReturn(PETSC_SUCCESS);
}
