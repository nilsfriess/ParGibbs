#pragma once

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

PetscErrorCode MatMultiColorSOR(Mat, const PetscInt *, Vec, Vec, PetscReal, PetscInt, const IS *, const VecScatter *, const Vec *, Vec);
