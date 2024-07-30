/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscsystypes.h>
#include <petscvec.h>

PETSC_EXTERN PetscErrorCode AddObservationToVec(DM dm, const PetscScalar *p, PetscScalar r, Vec vec);
PETSC_EXTERN PetscErrorCode MakeObservationMats(DM, PetscInt, PetscScalar, const PetscScalar *, PetscScalar *, const PetscScalar *, Mat *, Vec *, Vec *);
