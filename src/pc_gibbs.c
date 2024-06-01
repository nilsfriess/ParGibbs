#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petscdm.h>
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

#include <petscsystypes.h>
#include <petscvec.h>

#include <stdbool.h>
#include <string.h>

typedef struct {
  Mat         A, Asor;
  PetscRandom prand;
  Vec         sqrtdiag; // Both include omega
  PetscReal   omega;
  PetscBool   omega_changed;
  MCSOR       mc;

  PetscErrorCode (*prepare_rhs)(PC, Vec, Vec, Vec);
  PetscErrorCode (*sample_callback)(PetscInt, Vec, void *);
  void *callbackctx;
} PC_Gibbs;

static PetscErrorCode PCDestroy_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&pg->prand));
  PetscCall(VecDestroy(&pg->sqrtdiag));
  PetscCall(MCSORDestroy(&pg->mc));
  PetscCall(PetscFree(pg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrepareRHS_Default(PC pc, Vec y, Vec rhsin, Vec rhsout)
{
  (void)y;
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecSetRandom(rhsout, pg->prand));
  PetscCall(VecPointwiseMult(rhsout, rhsout, pg->sqrtdiag));
  PetscCall(VecAXPY(rhsout, 1., rhsin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGibbsUpdateSqrtDiag(PC pc)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MCSORSetOmega(pg->mc, pg->omega));
  PetscCall(MatGetDiagonal(pg->Asor, pg->sqrtdiag));
  PetscCall(VecSqrtAbs(pg->sqrtdiag));
  PetscCall(VecScale(pg->sqrtdiag, PetscSqrtReal((2 - pg->omega) / pg->omega)));
  pg->omega_changed = PETSC_FALSE;
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
  if (pg->omega_changed) PetscCall(PCGibbsUpdateSqrtDiag(pc));

  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(pg->prepare_rhs(pc, y, b, w));
    PetscCall(MCSORApply(pg->mc, w, y));
    if (pg->sample_callback) PetscCall(pg->sample_callback(it, y, pg->callbackctx));
  }
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_Gibbs(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_Gibbs *pg = pc->data;
  PetscBool flag;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "Gibbs options");
  PetscCall(PetscOptionsRangeReal("-pc_gibbs_omega", "Gibbs SOR parameter", NULL, pg->omega, &pg->omega, &flag, 0.0, 2.0));
  if (flag) pg->omega_changed = PETSC_TRUE;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;
  MatType   type;

  PetscFunctionBeginUser;
  PetscCall(MCSORCreate(pc->pmat, pg->omega, &pg->mc));

  PetscCall(MatGetType(pc->pmat, &type));
  if (strcmp(type, MATSEQAIJ) == 0) {
    pg->A    = pc->pmat;
    pg->Asor = pg->A;
  } else if (strcmp(type, MATMPIAIJ) == 0) {
    pg->A    = pc->pmat;
    pg->Asor = pg->A;
  } else if (strcmp(type, MATLRC) == 0) {
    pg->A = pc->pmat;
    PetscCall(MatLRCGetMats(pg->A, &pg->Asor, NULL, NULL, NULL));
  } else {
    PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Matrix type not supported");
  }

  PetscCall(MatCreateVecs(pg->A, &pg->sqrtdiag, NULL));
  pg->omega_changed = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGibbsSetSampleCallback(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  pg->sample_callback = cb;
  pg->callbackctx     = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGibbsGetPetscRandom(PC pc, PetscRandom *pr)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  *pr = pg->prand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGibbsSetOmega(PC pc, PetscReal omega)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  pg->omega         = omega;
  pg->omega_changed = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Gibbs(PC pc)
{
  PC_Gibbs *gibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&gibbs));
  gibbs->omega       = 1; // TODO: Allow user to change omega
  gibbs->prepare_rhs = PrepareRHS_Default;

  // TODO: Allow user to pass own PetscRandom
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &gibbs->prand));
  PetscCall(PetscRandomSetType(gibbs->prand, PARMGMC_ZIGGURAT));

  pc->data                 = gibbs;
  pc->ops->setup           = PCSetUp_Gibbs;
  pc->ops->destroy         = PCDestroy_Gibbs;
  pc->ops->applyrichardson = PCApplyRichardson_Gibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_Gibbs;
  /* pc->ops->apply           = PCApply_Gibbs; */
  PetscFunctionReturn(PETSC_SUCCESS);
}
