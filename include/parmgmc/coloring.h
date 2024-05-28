#pragma once

#include <petscis.h>
#include <petscmat.h>
#include <petscmacros.h>

PETSC_EXTERN PetscErrorCode MatCreateISColoring_AIJ(Mat, ISColoring *);
