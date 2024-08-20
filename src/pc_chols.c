#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsftypes.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>
#include <mpi.h>

typedef struct {
  Vec           r, v;
  Mat           F;
  PetscRandom   prand;
  MatSolverType st;
  PetscBool     prand_is_initial_prand, richardson;

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_CholSampler;

static PetscErrorCode PCDestroy_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&chol->F));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  if (chol->prand_is_initial_prand) PetscCall(PetscRandomDestroy(&chol->prand));
  PetscCall(PetscFree(chol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&chol->F));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  if (chol->prand_is_initial_prand) PetscCall(PetscRandomDestroy(&chol->prand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;
  Mat            S, P;
  PetscMPIInt    size;
  MatType        type;
  PetscBool      flag;
  IS             rowperm, colperm;
  MatFactorInfo  info;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &chol->prand));
  PetscCall(PetscRandomSetType(chol->prand, PARMGMC_ZIGGURAT));

  PetscCall(MatGetType(pc->pmat, &type));
  PetscCall(PetscStrcmp(type, MATLRC, &flag));
  if (flag) {
    Mat A, B, Bs, Bs_S, BSBt;
    Vec D;

    PetscCall(MatLRCGetMats(pc->pmat, &A, &B, &D, NULL));
    PetscCall(MatConvert(B, MATAIJ, MAT_INITIAL_MATRIX, &Bs));
    PetscCall(MatDuplicate(Bs, MAT_COPY_VALUES, &Bs_S));

    { // Scatter D into a distributed vector
      PetscInt   sctsize;
      IS         sctis;
      Vec        Sd;
      VecScatter sct;

      PetscCall(VecGetSize(D, &sctsize));
      PetscCall(ISCreateStride(MPI_COMM_WORLD, sctsize, 0, 1, &sctis));
      PetscCall(MatCreateVecs(Bs_S, &Sd, NULL));
      PetscCall(VecScatterCreate(D, sctis, Sd, NULL, &sct));
      PetscCall(VecScatterBegin(sct, D, Sd, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sct, D, Sd, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterDestroy(&sct));
      PetscCall(ISDestroy(&sctis));
      D = Sd;
    }

    PetscCall(MatDiagonalScale(Bs_S, NULL, D));
    PetscCall(MatMatTransposeMult(Bs_S, Bs, MAT_INITIAL_MATRIX, PETSC_DECIDE, &BSBt));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &P));
    PetscCall(MatAXPY(P, 1., BSBt, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Bs));
    PetscCall(MatDestroy(&Bs_S));
    PetscCall(MatDestroy(&BSBt));
    PetscCall(VecDestroy(&D));
  } else {
    P = pc->pmat;
  }

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (size != 1) PetscCall(MatConvert(P, MATSBAIJ, MAT_INITIAL_MATRIX, &S));
  else S = P;
  PetscCall(MatSetOption(S, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCreateVecs(S, &chol->r, &chol->v));
  PetscCall(MatGetFactor(S, chol->st, MAT_FACTOR_CHOLESKY, &chol->F));

  if (size == 1) PetscCall(MatGetOrdering(S, MATORDERINGMETISND, &rowperm, &colperm));
  else PetscCall(MatGetOrdering(S, MATORDERINGEXTERNAL, &rowperm, &colperm));
  PetscCall(MatCholeskyFactorSymbolic(chol->F, S, rowperm, &info));
  PetscCall(MatCholeskyFactorNumeric(chol->F, S, &info));
  if (size != 1) PetscCall(MatDestroy(&S));
  if (flag) PetscCall(MatDestroy(&P));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));

  pc->setupcalled         = PETSC_TRUE;
  pc->reusepreconditioner = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_CholSampler(PC pc, Vec x, Vec y)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCheck(chol->richardson || !chol->scb, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Setting a sample callback is not supported for Cholesky sampler in PREONLY mode. Use KSPRICHARDSON instead");
  PetscCall(MatForwardSolve(chol->F, x, chol->v));
  PetscCall(VecSetRandom(chol->r, chol->prand));
  PetscCall(VecAXPY(chol->v, 1., chol->r));
  PetscCall(MatBackwardSolve(chol->F, chol->v, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_CholSampler(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;
  (void)w;

  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  chol->richardson = PETSC_TRUE;
  for (PetscInt it = 0; it < its; ++it) {
    if (chol->scb) PetscCall(chol->scb(it, y, chol->cbctx));
    PetscCall(PCApply_CholSampler(pc, b, y));
  }
  if (chol->scb) PetscCall(chol->scb(its, y, chol->cbctx));
  *outits          = its;
  *reason          = PCRICHARDSON_CONVERGED_ITS;
  chol->richardson = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCholSamplerGetPetscRandom(PC pc, PetscRandom *pr)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  *pr = chol->prand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCCholSamplerSetPetscRandom(PC pc, PetscRandom pr)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  if (chol->prand_is_initial_prand) {
    PetscCall(PetscRandomDestroy(&chol->prand));
    chol->prand_is_initial_prand = PETSC_FALSE;
  }
  chol->prand = pr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_Cholsampler(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  if (chol->del_scb) PetscCall(chol->del_scb(chol->cbctx));
  chol->scb     = cb;
  chol->cbctx   = ctx;
  chol->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_CholSampler(PC pc, PetscViewer viewer)
{
  PC_CholSampler chol = pc->data;
  MatInfo        info;

  PetscFunctionBeginUser;
  PetscCall(MatGetInfo(chol->F, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Nonzeros in factored matrix: allocated %f\n", info.nz_allocated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_CholSampler(PC pc)
{
  PC_CholSampler chol;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&chol));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (size == 1) chol->st = MATSOLVERMKL_PARDISO;
  else chol->st = MATSOLVERMKL_CPARDISO;

  pc->data                     = chol;
  pc->ops->destroy             = PCDestroy_CholSampler;
  pc->ops->reset               = PCReset_CholSampler;
  pc->ops->setup               = PCSetUp_CholSampler;
  pc->ops->apply               = PCApply_CholSampler;
  pc->ops->applyrichardson     = PCApplyRichardson_CholSampler;
  pc->ops->view                = PCView_CholSampler;
  chol->prand_is_initial_prand = PETSC_TRUE;
  PetscCall(RegisterPCSetGetPetscRandom(pc, PCCholSamplerSetPetscRandom, PCCholSamplerGetPetscRandom));
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_Cholsampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}
