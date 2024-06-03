/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/pc/pc_gmgmc.h"
#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/random/ziggurat.h"

#include <petscerror.h>
#include <petsclog.h>
#include <petscpc.h>
#include <petscsys.h>

PetscClassId  PARMGMC_CLASSID;
PetscLogEvent MULTICOL_SOR;

static PetscErrorCode ParMGMCRegisterPCAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PCRegister("hogwild", PCCreate_Hogwild));
  PetscCall(PCRegister("gibbs", PCCreate_Gibbs));
  PetscCall(PCRegister("gmgmc", PCCreate_GMGMC));
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
