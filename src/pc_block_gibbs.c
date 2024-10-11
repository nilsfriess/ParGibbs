#include "parmgmc/pc/pc_block_gibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>

typedef struct {
  PetscInt   *blockcolmap, ncolors, *diagptrs;
  VecScatter *scatters;
  Vec        *ghostvecs, idiag, sqrtdiag;
  PetscRandom prand;

  /* PetscErrorCode (*prepare_rhs)(PC, Vec, Vec); */
} *PCBlockGibbs;

static PetscErrorCode PCDestroy_BlockGibbs(PC pc)
{
  PCBlockGibbs pcg = pc->data;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < pcg->ncolors; ++i) {
    PetscCall(VecScatterDestroy(&pcg->scatters[i]));
    PetscCall(VecDestroy(&pcg->ghostvecs[i]));
  }
  PetscCall(PetscFree(pcg->scatters));
  PetscCall(PetscFree(pcg->ghostvecs));
  PetscCall(PetscFree(pcg->blockcolmap));
  PetscCall(PetscFree(pcg->diagptrs));
  PetscCall(VecDestroy(&pcg->idiag));
  PetscCall(VecDestroy(&pcg->sqrtdiag));
  PetscCall(PetscRandomDestroy(&pcg->prand));
  PetscCall(PetscFree(pcg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrepareRHS(PC pc, Vec b, Vec c)
{
  PCBlockGibbs pcg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecSetRandom(c, pcg->prand));
  PetscCall(VecPointwiseMult(c, c, pcg->sqrtdiag));
  PetscCall(VecAXPY(c, 1., b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ColourBlockessors(PC pc)
{
  PCBlockGibbs     pcg = pc->data;
  PetscMPIInt     size, rank;
  Mat             Ap, Ao, P;
  const PetscInt *colmap, *ii, *jj;
  PetscInt        n, N, *blockmap, n_nb_total = 0, *block_cols;
  PetscScalar    *block_cols_vals;
  PetscLayout     layout;
  MatColoring     block_coloring;
  ISColoring      pcol;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(PetscCalloc1(size, &blockmap));

  PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &Ap, &Ao, &colmap));
  PetscCall(MatGetSize(Ao, &n, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetLayouts(pc->pmat, NULL, &layout));
  PetscCall(MatGetSize(pc->pmat, &N, NULL));

  /* Create map (of size `size` = no. of MPI ranks) that stores if a certain index
     (= MPI rank) has values that our blockess needs during Gauss-Seidel sweeps.
  */
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = ii[i]; j < ii[i + 1]; ++j) {
      PetscInt    c = colmap[jj[j]];
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, c, &owner));
      blockmap[owner] = 1;
    }
  }

  /* Find out how many MPI neighbours we have in total. */
  for (PetscInt i = 0; i < size; ++i)
    if (blockmap[i] > 0) n_nb_total++;

  /* Say the map above looks like this for rank 0:
        blockmap = {0, 1, 0, 1}.
     This means that rank 1 and rank 4 are neighbours. Below we transform this
     into the form
        block_cols = {1, 4}.
  */
  PetscCall(PetscCalloc1(n_nb_total, &block_cols));
  PetscCall(PetscCalloc1(n_nb_total, &block_cols_vals));
  for (PetscInt i = 0, j = 0; i < size; ++i) {
    if (blockmap[i] > 0) {
      block_cols[j]      = i;
      block_cols_vals[j] = 1;
      ++j;
    }
  }

  /* Finally we create a matrix whose matrix graphs represents the "MPI
     neighbouring layout" that we determined above. This matrix is always
     of size `#MPI ranks x #MPI ranks` and each blockess owns exactly one row.
     The columns contain a non-zero value (the value is irrelevant, we use 1)
     if the MPI rank corresponding to the column is a MPI neighbour of
     the current MPI rank. We do all this so that we can use PETSc's
     MatColoring to create a colouring of the blockessors.
   */
  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)pc), 1, 1, size, size, 0, NULL, n_nb_total, NULL, &P));
  for (PetscInt i = 0; i < size; ++i) {
    const PetscInt row = rank;
    PetscCall(MatSetValues(P, 1, &row, n_nb_total, block_cols, block_cols_vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(P, NULL, "-mat_blocks_view"));

  PetscCall(MatColoringCreate(P, &block_coloring));
  PetscCall(MatColoringSetDistance(block_coloring, 1));
  PetscCall(MatColoringSetType(block_coloring, MATCOLORINGJP));
  PetscCall(MatColoringApply(block_coloring, &pcol));
  PetscCall(ISColoringSetType(pcol, IS_COLORING_GLOBAL));
  PetscCall(ISColoringViewFromOptions(pcol, NULL, "-block_coloring_view"));
  PetscCall(ISColoringGetColors(pcol, NULL, &pcg->ncolors, NULL));

  /* Lastly, create a map (in the form of an array of length `size`) that maps
     MPI ranks to colour indices. */
  PetscCall(PetscMalloc1(size, &pcg->blockcolmap));
  for (PetscInt i = 0; i < size; ++i) {
    IS      *iss;
    PetscInt ncols;

    PetscCall(ISColoringGetIS(pcol, PETSC_USE_POINTER, &ncols, &iss));
    for (PetscInt c = 0; c < ncols; ++c) {
      const PetscInt *idxs;
      IS              gis;

      PetscCall(ISAllGather(iss[c], &gis));
      PetscCall(ISGetSize(gis, &n));
      PetscCall(ISGetIndices(gis, &idxs));
      for (PetscInt j = 0; j < n; ++j) {
        if (idxs[j] == i) pcg->blockcolmap[i] = c; // Blockess i has color c
      }
      PetscCall(ISRestoreIndices(gis, &idxs));
      PetscCall(ISDestroy(&gis));
    }
    PetscCall(ISColoringRestoreIS(pcol, PETSC_USE_POINTER, &iss));
  }

  PetscCall(MatColoringDestroy(&block_coloring));
  PetscCall(ISColoringDestroy(&pcol));
  PetscCall(PetscFree(block_cols));
  PetscCall(PetscFree(block_cols_vals));
  PetscCall(PetscFree(blockmap));
  PetscCall(MatDestroy(&P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalPointers(Mat A, PetscInt **diagptrs)
{
  PetscInt        rows;
  const PetscInt *i, *j;
  PetscReal      *a;
  Mat             P;
  MatType         type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &type));
  if (strcmp(type, MATSEQAIJ) == 0) P = A;
  else PetscCall(MatMPIAIJGetSeqAIJ(A, &P, NULL, NULL));

  PetscCall(MatGetSize(P, &rows, NULL));
  PetscCall(PetscMalloc1(rows, diagptrs));

  PetscCall(MatSeqAIJGetCSRAndMemType(P, &i, &j, &a, NULL));
  for (PetscInt row = 0; row < rows; ++row) {
    for (PetscInt k = i[row]; k < i[row + 1]; ++k) {
      PetscInt col = j[k];
      if (col == row) (*diagptrs)[row] = k;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Create the VecScatter contexts to get the off-blockessor values that are needed
  for colour i, i=1,...,#colors. Since all nodes of a given blockess have the same 
  colour, all but one VecScatter objects are actually "empty", in the sense that
  they might send values, but don't receive any.
 */
static PetscErrorCode CreateScatters(PC pc)
{
  PCBlockGibbs     pcg = pc->data;
  Mat             A, Ao;
  PetscInt        grows, lrows, noffblock = 0, *offblockidx, cnt = 0;
  const PetscInt *colmap, *rowptr, *colptr;
  Vec             gvec;
  IS              is;
  PetscMPIInt     rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(PCGetOperators(pc, NULL, &A));
  PetscCall(MatGetSize(A, &grows, NULL));
  PetscCall(MatGetLocalSize(A, &lrows, NULL));
  PetscCall(VecCreateMPIWithArray(MPI_COMM_WORLD, 1, lrows, grows, NULL, &gvec));
  PetscCall(MatMPIAIJGetSeqAIJ(A, NULL, &Ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &rowptr, &colptr, NULL, NULL));

  PetscCall(PetscCalloc1(pcg->ncolors, &pcg->scatters));
  PetscCall(PetscCalloc1(pcg->ncolors, &pcg->ghostvecs));
  for (PetscInt color = 0; color < pcg->ncolors; ++color) {
    if (pcg->blockcolmap[rank] == color) {
      for (PetscInt i = 0; i < lrows; ++i) noffblock += rowptr[i + 1] - rowptr[i];
      PetscCall(PetscCalloc1(noffblock, &offblockidx));
      for (PetscInt i = 0; i < lrows; ++i)
        for (PetscInt k = rowptr[i]; k < rowptr[i + 1]; ++k) offblockidx[cnt++] = colmap[colptr[k]];
    } else {
      noffblock   = 0;
      offblockidx = NULL;
    }

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, noffblock, offblockidx, PETSC_USE_POINTER, &is));
    PetscCall(VecCreateSeq(MPI_COMM_SELF, noffblock, &pcg->ghostvecs[color]));
    PetscCall(VecScatterCreate(gvec, is, pcg->ghostvecs[color], NULL, &pcg->scatters[color]));
    PetscCall(ISDestroy(&is));

    if (pcg->blockcolmap[rank] == color) { PetscCall(PetscFree(offblockidx)); }
  }
  PetscCall(VecDestroy(&gvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static PetscErrorCode PCApplyRichardson_BlockGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PCBlockGibbs      pcg = pc->data;
  PetscMPIInt      rank;
  Mat              Ad, Ao;
  PetscInt         gcnt, lsize;
  const PetscInt  *rowptr, *colptr, *bRowptr, *bColptr;
  const PetscReal *idiagarr, *warr, *ghostarr;
  PetscReal       *matvals, *bMatvals, *yarr;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &Ad, &Ao, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ad, &rowptr, &colptr, &matvals, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &bRowptr, &bColptr, &bMatvals, NULL));
  PetscCall(VecGetArrayRead(pcg->idiag, &idiagarr));
  PetscCall(MatGetLocalSize(pc->pmat, &lsize, NULL));

  for (PetscInt it = 0; it < its; ++it) {
    for (PetscInt color = 0; color < pcg->ncolors; ++color) {
      PetscCall(VecScatterBegin(pcg->scatters[color], y, pcg->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(PrepareRHS(pc, b, w));
      PetscCall(VecScatterEnd(pcg->scatters[color], y, pcg->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecGetArrayRead(w, &warr));
      if (pcg->blockcolmap[rank] == color) {
        PetscCall(VecGetArrayRead(pcg->ghostvecs[color], &ghostarr));
        PetscCall(VecGetArray(y, &yarr));

        gcnt = 0;
        for (PetscInt i = 0; i < lsize; ++i) {
          PetscReal sum = 0;

          for (PetscInt k = rowptr[i]; k < pcg->diagptrs[i]; ++k) sum -= matvals[k] * yarr[colptr[k]];
          for (PetscInt k = pcg->diagptrs[i] + 1; k < rowptr[i + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];
          for (PetscInt k = bRowptr[i]; k < bRowptr[i + 1]; ++k) sum -= bMatvals[k] * ghostarr[gcnt++];

          yarr[i] = idiagarr[i] * (sum + warr[i]);
        }

        PetscCall(VecRestoreArray(y, &yarr));
        PetscCall(VecRestoreArrayRead(pcg->ghostvecs[color], &ghostarr));

      } else {
        // Could do other work here
      }
      PetscCall(VecRestoreArrayRead(w, &warr));
    }
  }
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic pop

static PetscErrorCode PCSetUp_BlockGibbs(PC pc)
{
  PCBlockGibbs pcg = pc->data;
  PetscInt    size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCall(ColourBlockessors(pc));
  PetscCall(CreateScatters(pc));
  PetscCall(MatCreateVecs(pc->pmat, &pcg->idiag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, pcg->idiag));
  PetscCall(VecReciprocal(pcg->idiag));
  PetscCall(MatGetDiagonalPointers(pc->pmat, &(pcg->diagptrs)));
  PetscCall(VecDuplicate(pcg->idiag, &pcg->sqrtdiag));
  PetscCall(MatGetDiagonal(pc->pmat, pcg->sqrtdiag));
  PetscCall(VecSqrtAbs(pcg->sqrtdiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetPetscRandom_BlockGibbs(PC pc, PetscRandom pr)
{
  PCBlockGibbs pcg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&pcg->prand));
  pcg->prand = pr;
  PetscCall(PetscObjectReference((PetscObject)pcg->prand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGetPetscRandom_BlockGibbs(PC pc, PetscRandom *pr)
{
  PCBlockGibbs pcg = pc->data;

  PetscFunctionBeginUser;
  *pr = pcg->prand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_BlockGibbs(PC pc)
{
  PCBlockGibbs pcg;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&pcg));
  pc->data                 = pcg;
  pc->ops->setup           = PCSetUp_BlockGibbs;
  pc->ops->applyrichardson = PCApplyRichardson_BlockGibbs;
  pc->ops->destroy         = PCDestroy_BlockGibbs;

  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &pcg->prand));
  PetscCall(PetscRandomSetType(pcg->prand, PARMGMC_ZIGGURAT));
  PetscCall(RegisterPCSetGetPetscRandom(pc, PCSetPetscRandom_BlockGibbs, PCGetPetscRandom_BlockGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
