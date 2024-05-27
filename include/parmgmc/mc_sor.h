#pragma once

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

typedef struct {
  VecScatter *scatters;  // A VecScatter context for scattering the boundary nodes for each color
  Vec        *ghostvecs; // A Vec of the correct size to scatter the boundary values for each color into
} CTX_SOR;

PETSC_EXTERN PetscErrorCode MatMultiColorSOR_MPIAIJ(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec);
PETSC_EXTERN PetscErrorCode MatMultiColorSOR_SEQAIJ(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec);
