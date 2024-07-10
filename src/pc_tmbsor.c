#include "parmgmc/pc/pc_tmbsor.h"

#include <petsc/private/pcimpl.h>
#include <petscis.h>
#include <petscmat.h>

typedef struct {
  ISColoring pcol;
} *TMBSOR;

static PetscErrorCode PCDestroy_TMBSOR(PC pc)
{
  TMBSOR tmb = pc->data;

  PetscFunctionBeginUser;
  PetscCall(ISColoringDestroy(&tmb->pcol));
  PetscCall(PetscFree(tmb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TMBSOR_ColorProcessors(PC pc)
{
  TMBSOR          tmb = pc->data;
  PetscMPIInt     size, rank;
  Mat             Ap, Ao, P;
  const PetscInt *colmap, *ii, *jj;
  PetscInt        n, N, *procmap, n_nb_total = 0, *proc_cols;
  PetscScalar    *proc_cols_vals;
  PetscLayout     layout;
  MatColoring     proc_coloring;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCall(PetscCalloc1(size, &procmap));

  PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &Ap, &Ao, &colmap));
  PetscCall(MatGetSize(Ao, &n, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetLayouts(pc->pmat, NULL, &layout));
  PetscCall(MatGetSize(pc->pmat, &N, NULL));

  /* Create map (of size `size` = no. of MPI ranks) that stores if a certain index
     (= MPI rank) has values that our process needs during Gauss-Seidel sweeps.
  */
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = ii[i]; j < ii[i + 1]; ++j) {
      PetscInt    c = colmap[jj[j]];
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, c, &owner));
      procmap[owner] = 1;
    }
  }

  /* Find out how many MPI neighbours we have in total. */
  for (PetscInt i = 0; i < size; ++i)
    if (procmap[i] > 0) n_nb_total++;

  /* Say the map above looks like this for rank 0:
        procmap = {0, 1, 0, 1}.
     This means that rank 1 and rank 4 are neighbours. Below we transform this
     into the form
        proc_cols = {1, 4}.
  */
  PetscCall(PetscCalloc1(n_nb_total, &proc_cols));
  PetscCall(PetscCalloc1(n_nb_total, &proc_cols_vals));
  for (PetscInt i = 0, j = 0; i < size; ++i) {
    if (procmap[i] > 0) {
      proc_cols[j]      = i;
      proc_cols_vals[j] = 1;
      ++j;
    }
  }

  /* Finally we create a matrix whose matrix graphs represents the "MPI
     neighbouring layout" that we determined above. This matrix is always
     of size `#MPI ranks x #MPI ranks` and each process owns exactly one row.
     The columns contain a non-zero value (the value is irrelevant, we use 1)
     if the MPI rank corresponding to the column is a MPI neighbour of
     the current MPI rank. We do all this so that we can use PETSc's
     MatColoring to create a colouring of the processors.
   */
  PetscCall(MatCreateAIJ(MPI_COMM_WORLD, 1, 1, size, size, 0, NULL, n_nb_total, NULL, &P));
  for (PetscInt i = 0; i < size; ++i) {
    const PetscInt row = rank;
    PetscCall(MatSetValues(P, 1, &row, n_nb_total, proc_cols, proc_cols_vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  PetscCall(MatColoringCreate(P, &proc_coloring));
  PetscCall(MatColoringSetDistance(proc_coloring, 1));
  PetscCall(MatColoringSetType(proc_coloring, MATCOLORINGJP));
  PetscCall(MatColoringApply(proc_coloring, &tmb->pcol));
  PetscCall(ISColoringViewFromOptions(tmb->pcol, NULL, "-proc_coloring_view"));

  PetscCall(MatColoringDestroy(&proc_coloring));
  PetscCall(PetscFree(proc_cols));
  PetscCall(PetscFree(proc_cols_vals));
  PetscCall(PetscFree(procmap));
  PetscCall(MatDestroy(&P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_TMBSOR(PC pc)
{
  PetscFunctionBeginUser;
  PetscCall(TMBSOR_ColorProcessors(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_TMBSOR(PC pc)
{
  TMBSOR sor;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sor));
  pc->data = sor;

  pc->ops->setup   = PCSetUp_TMBSOR;
  pc->ops->destroy = PCDestroy_TMBSOR;
  PetscFunctionReturn(PETSC_SUCCESS);
}
