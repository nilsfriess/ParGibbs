#pragma once

#include <petscpc.h>

PETSC_EXTERN PetscErrorCode PCGMGMCSetLevels(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCGMGMCSetSampleCallback(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode PCCreate_GMGMC(PC);
