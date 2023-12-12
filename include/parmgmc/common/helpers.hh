#pragma once

#include <algorithm>
#include <random>

#include <mpi.h>
#include <petscerror.h>
#include <petsclog.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine>
PetscErrorCode fill_vec_rand(Vec vec, PetscInt size, Engine &engine) {
  static std::normal_distribution<PetscReal> dist;

  PetscFunctionBeginUser;
  PetscScalar *r_arr;
  PetscCall(VecGetArray(vec, &r_arr));
  std::generate_n(r_arr, size, [&]() { return dist(engine); });
  PetscCall(VecRestoreArray(vec, &r_arr));

  // Estimated using perf
  PetscCall(size * PetscLogFlops(27));

  PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace parmgmc
