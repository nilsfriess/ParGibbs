/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscmat.h>

typedef struct _MCSOR {
  void *ctx;
} *MCSOR;

PETSC_EXTERN PetscErrorCode MCSORCreate(Mat, PetscReal, MCSOR *);
PETSC_EXTERN PetscErrorCode MCSORDestroy(MCSOR *);
PETSC_EXTERN PetscErrorCode MCSORApply(MCSOR, Vec, Vec);
PETSC_EXTERN PetscErrorCode MCSORSetOmega(MCSOR, PetscReal);
