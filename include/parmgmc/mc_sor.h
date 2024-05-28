#pragma once

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

typedef struct {
  PetscInt    ncolors;
  VecScatter *scatters;  // A VecScatter context for scattering the boundary nodes for each color
  Vec        *ghostvecs; // A Vec of the correct size to scatter the boundary values for each color into
} SORCtx_MPIAIJ;

typedef struct {
  SORCtx_MPIAIJ *basectx;
  PetscErrorCode (*basesor)(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec);
} SORCtx_LRC;

PETSC_EXTERN PetscErrorCode ContextDestroy_MPIAIJ(void *);
PETSC_EXTERN PetscErrorCode ContextDestroy_LRC(void *);

PETSC_EXTERN PetscErrorCode MatMultiColorSOR_MPIAIJ(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec);
PETSC_EXTERN PetscErrorCode MatMultiColorSOR_SEQAIJ(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec);
PETSC_EXTERN PetscErrorCode MatMultiColorSOR_LRC(Mat, const PetscInt *, Vec, Vec, PetscReal, ISColoring, void *, Vec);
