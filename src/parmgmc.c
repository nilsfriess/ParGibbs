#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/random/ziggurat.h"

#include <petscerror.h>
#include <petsclog.h>
#include <petscpc.h>
#include <petscsys.h>

PetscClassId PARMGMC_CLASSID;
PetscLogEvent MULTICOL_SOR;

static PetscErrorCode ParMGMCRegisterPCAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PCRegister("hogwild", PCCreate_Hogwild));
  PetscCall(PCRegister("gibbs", PCCreate_Gibbs));
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
