#include "parmgmc/coloring.h"
#include <petscmat.h>

PetscErrorCode MatCreateISColoring_AIJ(Mat mat, ISColoring *isc)
{
  MatColoring mc;

  PetscFunctionBeginUser;
  PetscCall(MatColoringCreate(mat, &mc));
  PetscCall(MatColoringSetDistance(mc, 1));
  PetscCall(MatColoringSetType(mc, MATCOLORINGGREEDY));
  PetscCall(MatColoringApply(mc, isc));
  PetscCall(ISColoringSetType(*isc, IS_COLORING_LOCAL));
  PetscCall(MatColoringDestroy(&mc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
