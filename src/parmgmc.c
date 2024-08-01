/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/ksp/cgs.h"
#include "parmgmc/random/ziggurat.h"

#include <petscerror.h>
#include <petsclog.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petsc/private/pcimpl.h>

/** @file
    @brief This file contains general purpose functions for the ParMGMC library.
*/

PetscClassId  PARMGMC_CLASSID;
PetscLogEvent MULTICOL_SOR;

static PetscErrorCode ParMGMCRegisterPCAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PCRegister(PCHOGWILD, PCCreate_Hogwild));
  PetscCall(PCRegister(PCGIBBS, PCCreate_Gibbs));
  PetscCall(PCRegister(PCGAMGMC, PCCreate_GAMGMC));
  PetscCall(PCRegister(PCCHOLSAMPLER, PCCreate_CholSampler));

  PetscCall(KSPRegister(KSPCGSAMPLER, KSPCreate_CGSampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParMGMCRegisterPetscRandomAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PetscRandomRegister(PARMGMC_ZIGGURAT, PetscRandomCreate_Ziggurat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCInitialize(void)
{
  PetscFunctionBeginUser;
  PetscCall(ParMGMCRegisterPCAll());
  PetscCall(ParMGMCRegisterPetscRandomAll());

  PetscCall(PetscClassIdRegister("ParMGMC", &PARMGMC_CLASSID));
  PetscCall(PetscLogEventRegister("MulticolSOR", PARMGMC_CLASSID, &MULTICOL_SOR));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Set a callback function that is called everytime a sampler generated a new sample.

   For PETSc's own PC's this does nothing. This works by abusing the `void* user` field
   in the `PC` class which also means that the `user` field cannot be used for anything
   else.
*/
PetscErrorCode PCSetSampleCallback(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx)
{
  SampleCallbackCtx cbctx;

  PetscFunctionBeginUser;
  if (!pc->user) PetscFunctionReturn(PETSC_SUCCESS);

  cbctx      = pc->user;
  cbctx->cb  = cb;
  cbctx->ctx = ctx;

  PetscFunctionReturn(PETSC_SUCCESS);
}
