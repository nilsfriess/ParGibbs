#pragma once

#include <petscmat.h>

typedef struct _MCSOR {
  void *ctx;
} *MCSOR;

PETSC_EXTERN PetscErrorCode MCSORCreate(Mat, PetscReal, MCSOR *);
PETSC_EXTERN PetscErrorCode MCSORDestroy(MCSOR *);
PETSC_EXTERN PetscErrorCode MCSORApply(MCSOR, Vec, Vec);
PETSC_EXTERN PetscErrorCode MCSORSetOmega(MCSOR, PetscReal);
