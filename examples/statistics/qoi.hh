#pragma once

#include <mpi.h>

#include <petscerror.h>
#include <petscsystypes.h>
#include <petscvec.h>

/* Example quantity of interest that computes Q = ||v||,
 * where v is the sample. */
struct NormQOI {
  using DataT = PetscScalar;

  PetscErrorCode operator()(Vec sample, PetscScalar *res) const {
    PetscFunctionBeginUser;

    PetscCall(VecNorm(sample, NORM_2, res));

    PetscFunctionReturn(PETSC_SUCCESS);
  }
};
