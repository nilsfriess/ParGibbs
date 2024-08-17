#pragma once

#include <petscmat.h>
#include <petscsystypes.h>

PETSC_EXTERN PetscErrorCode EstimateCovarianceMatErrors(Mat, PetscInt, PetscInt, Vec *, PetscScalar *);
