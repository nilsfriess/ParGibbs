/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petsclog.h>
#include <petscmacros.h>
#include <petscsys.h>

#define PARMGMC_ZIGGURAT "ziggurat"

PETSC_EXTERN PetscClassId  PARMGMC_CLASSID;
PETSC_EXTERN PetscLogEvent MULTICOL_SOR;

PETSC_EXTERN PetscErrorCode ParMGMCInitialize(void);
