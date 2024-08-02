#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/parmgmc.h"

#include <petscmacros.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petsc/private/pcimpl.h>
#include <petscvec.h>

#include <mpi.h>

#include <time.h>

typedef struct {
  Vec           r, v;
  Mat           F;
  PetscRandom   pr;
  MatSolverType st;
} *PC_CholSampler;

static PetscErrorCode PCDestroy_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&chol->F));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  PetscCall(PetscRandomDestroy(&chol->pr));
  PetscCall(PetscFree(chol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;
  Mat            S, P;
  PetscMPIInt    size;
  MatType        type;
  PetscBool      flag;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(pc->pmat, &type));
  PetscCall(PetscStrcmp(type, MATLRC, &flag));
  if (flag) {
    Mat A, B, Bs, Bs_S, BSBt;
    Vec D;

    PetscCall(MatLRCGetMats(pc->pmat, &A, &B, &D, NULL));
    PetscCall(MatConvert(B, MATAIJ, MAT_INITIAL_MATRIX, &Bs));
    PetscCall(MatDuplicate(Bs, MAT_COPY_VALUES, &Bs_S));
    PetscCall(MatDiagonalScale(Bs_S, NULL, D));
    PetscCall(MatMatTransposeMult(Bs_S, Bs, MAT_INITIAL_MATRIX, PETSC_DECIDE, &BSBt));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &P));
    PetscCall(MatAXPY(P, 1., BSBt, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Bs));
    PetscCall(MatDestroy(&Bs_S));
    PetscCall(MatDestroy(&BSBt));
  } else {
    P = pc->pmat;
  }

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (size != 1) PetscCall(MatConvert(P, MATSBAIJ, MAT_INITIAL_MATRIX, &S));
  else S = P;
  PetscCall(MatSetOption(S, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCreateVecs(S, &chol->r, &chol->v));
  PetscCall(MatGetFactor(S, chol->st, MAT_FACTOR_CHOLESKY, &chol->F));

  PetscCall(MatCholeskyFactorSymbolic(chol->F, S, NULL, NULL));
  PetscCall(MatCholeskyFactorNumeric(chol->F, S, NULL));
  if (size != 1 || flag) PetscCall(MatDestroy(&S));

  pc->setupcalled         = PETSC_TRUE;
  pc->reusepreconditioner = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_CholSampler(PC pc, Vec x, Vec y)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatForwardSolve(chol->F, x, chol->v));
  PetscCall(VecSetRandom(chol->r, chol->pr));
  PetscCall(VecAXPY(chol->v, 1., chol->r));
  PetscCall(MatBackwardSolve(chol->F, chol->v, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCholSamplerGetPetscRandom(PC pc, PetscRandom *pr)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  *pr = chol->pr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_CholSampler(PC pc)
{
  PC_CholSampler chol;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&chol));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &chol->pr));
  PetscCall(PetscRandomSetType(chol->pr, PARMGMC_ZIGGURAT));
  PetscCall(PetscRandomSetSeed(chol->pr, time(0)));
  PetscCall(PetscRandomSeed(chol->pr));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (size == 1) chol->st = MATSOLVERMKL_PARDISO;
  else chol->st = MATSOLVERMKL_CPARDISO;

  pc->data         = chol;
  pc->ops->destroy = PCDestroy_CholSampler;
  pc->ops->setup   = PCSetUp_CholSampler;
  pc->ops->apply   = PCApply_CholSampler;
  PetscFunctionReturn(PETSC_SUCCESS);
}
