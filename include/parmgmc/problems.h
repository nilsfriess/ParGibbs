#pragma once

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode MatAssembleShiftedLaplaceFD(DM, PetscReal, Mat);
