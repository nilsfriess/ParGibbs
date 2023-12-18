#pragma once

#include <iostream>
#include <petsclog.h>
#include <petscsys.h>

namespace parmgmc {
struct PetscHelper {
  PetscHelper(int *argc, char ***argv) {
    PetscInitialize(argc, argv, NULL, NULL);
  }

  static PetscErrorCode get_rng_event(PetscLogEvent *evt) {
    static PetscLogEvent rng_event;

    PetscFunctionBeginUser;
    if (!rng_event_registered) {
      PetscClassId classid;
      PetscCall(PetscClassIdRegister("ParMGMC", &classid));
      PetscCall(PetscLogEventRegister("RNG", classid, &rng_event));

      rng_event_registered = true;
    }

    *evt = rng_event;

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~PetscHelper() { PetscFinalize(); }

private:
  inline static bool rng_event_registered = false;
};
} // namespace parmgmc
