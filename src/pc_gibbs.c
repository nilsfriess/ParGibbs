#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petscis.h>
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
} PC_Gibbs;

static PetscErrorCode PCDestroy_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;
  PetscInt  ncolors;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&pg->prand));

  PetscCall(ISColoringGetIS(pg->ic, PETSC_USE_POINTER, &ncolors, NULL));
  if (pg->sor_ctx) {
    CTX_SOR *ctx = pg->sor_ctx;
    for (PetscInt i = 0; i < ncolors; ++i) {
      PetscCall(VecScatterDestroy(&ctx->scatters[i]));
      PetscCall(VecDestroy(&ctx->ghostvecs[i]));
    }
  }
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
  // PetscCall(MatMultiColorSOR(pc->pmat, pg->diagptrs, pg->idiag, x, pg->omega, ncolors, isc, pg->scatters, pg->ghostvecs, y));
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

static PetscErrorCode PCGibbs_CreateColoring(PC pc)
{
  PC_Gibbs   *pg = pc->data;
  MatColoring mc;

  PetscFunctionBeginUser;
  PetscCall(MatColoringCreate(pc->pmat, &mc));
  PetscCall(MatColoringSetDistance(mc, 1));
  PetscCall(MatColoringSetType(mc, MATCOLORINGGREEDY));
  PetscCall(MatColoringApply(mc, &pg->ic));
  PetscCall(ISColoringSetType(pg->ic, IS_COLORING_LOCAL));
  PetscCall(MatColoringDestroy(&mc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGibbs_CreateScatters(PC pc)
{
  PC_Gibbs *pg = pc->data;
  PetscInt  ncolors;
  IS       *isc;
  Mat       P = pc->pmat;
  CTX_SOR  *ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&ctx));
  PetscCall(ISColoringGetIS(pg->ic, PETSC_USE_POINTER, &ncolors, &isc));
  PetscCall(PetscMalloc1(ncolors, &ctx->scatters));
  PetscCall(PetscMalloc1(ncolors, &ctx->ghostvecs));

  Mat             ao; // off-processor part of matrix
  const PetscInt *colmap, *rowptr, *colptr;
  PetscCall(MatMPIAIJGetSeqAIJ(P, NULL, &ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(ao, &rowptr, &colptr, NULL, NULL));

  Vec      gvec;
  PetscInt localRows, globalRows;
  PetscCall(MatGetSize(P, &globalRows, NULL));
  PetscCall(MatGetLocalSize(P, &localRows, NULL));
  PetscCall(VecCreateMPIWithArray(MPI_COMM_WORLD, 1, localRows, globalRows, NULL, &gvec));

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // First count the total number of off-processor values for each color
  PetscInt *nTotalOffProc;
  PetscCall(PetscCalloc1(ncolors, &nTotalOffProc));
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(ISGetLocalSize(isc[color], &nCurCol));
    PetscCall(ISGetIndices(isc[color], &curidxs));
    for (PetscInt i = 0; i < nCurCol; ++i) nTotalOffProc[color] += rowptr[curidxs[i] + 1] - rowptr[curidxs[i]];
    PetscCall(ISRestoreIndices(isc[color], &curidxs));
  }

  // Now we again loop over all colors and create the required VecScatters
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt       *offProcIdx;
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(PetscMalloc1(nTotalOffProc[color], &offProcIdx));
    PetscCall(ISGetLocalSize(isc[color], &nCurCol));
    PetscCall(ISGetIndices(isc[color], &curidxs));
    PetscInt cnt = 0;
    for (PetscInt i = 0; i < nCurCol; ++i)
      for (PetscInt k = rowptr[curidxs[i]]; k < rowptr[curidxs[i] + 1]; ++k) offProcIdx[cnt++] = colmap[colptr[k]];
    PetscCall(ISRestoreIndices(isc[color], &curidxs));

    IS is;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nTotalOffProc[color], offProcIdx, PETSC_USE_POINTER, &is));
    PetscCall(VecCreateSeq(MPI_COMM_SELF, nTotalOffProc[color], &ctx->ghostvecs[color]));
    PetscCall(VecScatterCreate(gvec, is, ctx->ghostvecs[color], NULL, &ctx->scatters[color]));
    PetscCall(ISDestroy(&is));
    PetscCall(PetscFree(offProcIdx));
  }

  PetscCall(ISColoringRestoreIS(pg->ic, PETSC_USE_POINTER, &isc));
  PetscCall(PetscFree(nTotalOffProc));
  PetscCall(VecDestroy(&gvec));

  pg->sor_ctx = ctx;
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
  MatType   type;

  PetscFunctionBeginUser;
  PetscCall(PCGibbs_CreateColoring(pc));

  PetscCall(MatCreateVecs(pc->pmat, &pg->idiag, NULL));
  PetscCall(VecDuplicate(pg->idiag, &pg->sqrtdiag));

  PetscCall(MatGetDiagonal(pc->pmat, pg->sqrtdiag));
  PetscCall(VecSqrtAbs(pg->sqrtdiag));
  PetscCall(VecScale(pg->sqrtdiag, PetscSqrtReal((2 - pg->omega) / pg->omega)));

  PetscCall(MatGetDiagonal(pc->pmat, pg->idiag));
  PetscCall(VecReciprocal(pg->idiag));
  PetscCall(VecScale(pg->idiag, pg->omega));

  PetscCall(MatGetType(pc->pmat, &type));
  if (strcmp(type, MATMPIAIJ) == 0) {
    Mat mat;
    PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &mat, NULL, NULL));
    PetscCall(MatGetDiagonalPointers(mat, &pg->diagptrs));
    PetscCall(PCGibbs_CreateScatters(pc));
    pg->sor = MatMultiColorSOR_MPIAIJ;
  } else if (strcmp(type, MATSEQAIJ) == 0) {
    PetscCall(MatGetDiagonalPointers(pc->pmat, &pg->diagptrs));
    pg->sor = MatMultiColorSOR_SEQAIJ;
  } else {
    PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only MATMPIAIJ and MATSEQAIJ types are supported");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Gibbs(PC pc)
{
  PC_Gibbs *gibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&gibbs));
  gibbs->omega   = 1.0; // TODO: Allow user to change omega
  gibbs->sor_ctx = NULL;

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
