#pragma once

#include <petscpc.h>

PETSC_EXTERN PetscErrorCode PCCreate_Gibbs(PC);
PETSC_EXTERN PetscErrorCode PCGibbsSetSampleCallback(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode PCGibbsGetPetscRandom(PC, PetscRandom *);
PETSC_EXTERN PetscErrorCode PCGibbsSetOmega(PC, PetscReal);
