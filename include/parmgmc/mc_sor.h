#pragma once

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

PETSC_EXTERN PetscErrorCode MatMultiColorSOR(Mat, const PetscInt *, Vec, Vec, PetscReal, PetscInt, const IS *, const VecScatter *, const Vec *, Vec);
