/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stddef.h>

typedef struct {
  Vec         sqrtdiag;
  PetscRandom prand;
  PetscBool   prand_is_initial_prand;

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_Hogwild;

static PetscErrorCode PCApplyRichardson_Hogwild(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  for (PetscInt it = 0; it < its; ++it) {
    if (hw->scb) PetscCall(hw->scb(it, y, hw->cbctx));

    PetscCall(VecSetRandom(w, hw->prand));
    PetscCall(VecPointwiseMult(w, w, hw->sqrtdiag));
    PetscCall(VecAXPY(w, 1., b));
    PetscCall(MatSOR(pc->pmat, w, 1., SOR_LOCAL_FORWARD_SWEEP, 0., 1., 1., y));
  }
  if (hw->scb) PetscCall(hw->scb(its, y, hw->cbctx));

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_Hogwild(PC pc)
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&hw->sqrtdiag));
  if (hw->del_scb) {
    PetscCall(hw->del_scb(hw->cbctx));
    hw->del_scb = NULL;
  }
  if (hw->prand_is_initial_prand) PetscCall(PetscRandomDestroy(&hw->prand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Hogwild(PC pc)
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&hw->sqrtdiag));
  if (hw->del_scb) {
    PetscCall(hw->del_scb(hw->cbctx));
    hw->del_scb = NULL;
  }
  if (hw->prand_is_initial_prand) PetscCall(PetscRandomDestroy(&hw->prand));
  PetscCall(PetscFree(hw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Hogwild(PC pc)
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(pc->pmat, &hw->sqrtdiag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, hw->sqrtdiag));
  PetscCall(VecSqrtAbs(hw->sqrtdiag));

  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &hw->prand));
  PetscCall(PetscRandomSetType(hw->prand, PARMGMC_ZIGGURAT));
  hw->prand_is_initial_prand = PETSC_TRUE; // TODO: Use PETSc's reference counting instead
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_Hogwild(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  if (hw->del_scb) {
    PetscCall(hw->del_scb(hw->cbctx));
    hw->del_scb = NULL;
  }
  hw->scb     = cb;
  hw->cbctx   = ctx;
  hw->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHogwildGetPetscRandom(PC pc, PetscRandom *pr)
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  *pr = hw->prand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHogwildSetPetscRandom(PC pc, PetscRandom pr)
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  if (hw->prand_is_initial_prand) {
    PetscCall(PetscRandomDestroy(&hw->prand));
    hw->prand_is_initial_prand = PETSC_FALSE;
  }
  hw->prand = pr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Hogwild(PC pc)
{
  PC_Hogwild hw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&hw));
  pc->data = hw;

  pc->ops->applyrichardson = PCApplyRichardson_Hogwild;
  pc->ops->destroy         = PCDestroy_Hogwild;
  pc->ops->reset           = PCReset_Hogwild;
  pc->ops->setup           = PCSetUp_Hogwild;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_Hogwild));
  PetscCall(RegisterPCSetGetPetscRandom(pc, PCHogwildSetPetscRandom, PCHogwildGetPetscRandom));
  PetscFunctionReturn(PETSC_SUCCESS);
}
