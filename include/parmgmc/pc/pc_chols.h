#pragma once

#include <petscmacros.h>
#include <petscpctypes.h>
#include <petscsystypes.h>

PETSC_EXTERN PetscErrorCode PCCreate_CholSampler(PC);
PETSC_EXTERN PetscErrorCode PCCholSamplerGetPetscRandom(PC, PetscRandom *);
PETSC_EXTERN PetscErrorCode PCCholSamplerSetIsCoarseGAMG(PC, PetscBool);
