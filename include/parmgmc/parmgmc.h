#pragma once

#include <petsclog.h>
#include <petscmacros.h>
#include <petscsys.h>

#define PARMGMC_ZIGGURAT "ziggurat"

PETSC_EXTERN PetscClassId  PARMGMC_CLASSID;
PETSC_EXTERN PetscLogEvent MULTICOL_SOR;

PETSC_EXTERN PetscErrorCode ParMGMCInitialize(void);
