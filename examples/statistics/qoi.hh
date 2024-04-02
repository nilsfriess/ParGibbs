#pragma once

#include <mpi.h>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscerror.h>
#include <petscsystypes.h>
#include <petscvec.h>

/* Example quantity of interest that computes Q(v) = <v,r> for some fixed vector r. */
class TestQOI {
public:
  using DataT = PetscScalar;

  TestQOI(Vec direction) {
    PetscFunctionBeginUser;
    PetscCallVoid(VecDuplicate(direction, &r));
    PetscCallVoid(VecCopy(direction, r));
    PetscFunctionReturnVoid();
  }

  PetscErrorCode operator()(Vec sample, PetscScalar *res) const {
    PetscFunctionBeginUser;

    PetscCall(VecDot(sample, r, res));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~TestQOI() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&r));

    PetscFunctionReturnVoid();
  }

private:
  Vec r;
};
