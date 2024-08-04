/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscpc.h>

PETSC_EXTERN PetscErrorCode PCCreate_Gibbs(PC);
PETSC_EXTERN PetscErrorCode PCGibbsGetPetscRandom(PC, PetscRandom *);
PETSC_EXTERN PetscErrorCode PCGibbsSetOmega(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCGibbsSetSweepType(PC, MatSORType);
