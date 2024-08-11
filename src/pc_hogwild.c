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
    PetscCall(VecSetRandom(w, hw->prand));
    PetscCall(VecPointwiseMult(w, w, hw->sqrtdiag));
    PetscCall(VecAXPY(w, 1., b));
    PetscCall(MatSOR(pc->pmat, w, 1., SOR_LOCAL_FORWARD_SWEEP, 0., 1., 1., y));
  }

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Hogwild(PC pc)
{
  PC_Hogwild hw = pc->data;
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&hw->sqrtdiag));
  PetscCall(PetscRandomDestroy(&hw->prand));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Hogwild(PC pc)
{
  PC_Hogwild hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(pc->pmat, &hw->sqrtdiag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, hw->sqrtdiag));
  PetscCall(VecSqrtAbs(hw->sqrtdiag));

  // TODO: Allow user to pass own PetscRandom
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &hw->prand));
  PetscCall(PetscRandomSetType(hw->prand, PARMGMC_ZIGGURAT));
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
  pc->ops->setup           = PCSetUp_Hogwild;
  PetscFunctionReturn(PETSC_SUCCESS);
}
