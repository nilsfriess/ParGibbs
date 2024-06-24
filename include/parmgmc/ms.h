/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscdm.h>
#include <petscsys.h>
#include <mpi.h>

typedef struct _p_MS {
  void *ctx;
} *MS;

PETSC_EXTERN PetscErrorCode MSCreate(MPI_Comm, MS *);
PETSC_EXTERN PetscErrorCode MSDestroy(MS *);
PETSC_EXTERN PetscErrorCode MSSetFromOptions(MS);
PETSC_EXTERN PetscErrorCode MSSetDM(MS, DM);
PETSC_EXTERN PetscErrorCode MSSetUp(MS);

PETSC_EXTERN PetscErrorCode MSGetDM(MS, DM *);
PETSC_EXTERN PetscErrorCode MSSetAlpha(MS, PetscInt);
PETSC_EXTERN PetscErrorCode MSSetKappa(MS, PetscScalar);

PETSC_EXTERN PetscErrorCode MSSample(MS, Vec);
PETSC_EXTERN PetscErrorCode MSBeginSaveSamples(MS, PetscInt);
PETSC_EXTERN PetscErrorCode MSEndSaveSamples(MS, PetscInt);
PETSC_EXTERN PetscErrorCode MSGetMeanAndVar(MS, Vec *, Vec *);
