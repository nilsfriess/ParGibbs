#pragma once

#include "parmgmc/common/petsc_helper.hh"
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

  PetscLogEvent rng_event;
  PetscCall(PetscHelper::get_rng_event(&rng_event));
  PetscCall(PetscLogEventBegin(rng_event, NULL, NULL, NULL, NULL));

  PetscScalar *r_arr;
  PetscCall(VecGetArray(vec, &r_arr));
  std::generate_n(r_arr, size, [&]() { return dist(engine); });
  PetscCall(VecRestoreArray(vec, &r_arr));

  // Estimated using perf
  PetscCall(PetscLogFlops(size * 27));

  PetscCall(PetscLogEventEnd(rng_event, NULL, NULL, NULL, NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace parmgmc
