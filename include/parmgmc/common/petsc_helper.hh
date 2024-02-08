#pragma once

#include <iostream>
#include <petsclog.h>
#include <petscsys.h>

namespace parmgmc {
struct PetscHelper {
  static void init(int &argc, char **&argv, const char *file = nullptr,
                   const char *help = nullptr) {
    static PetscHelper helper(&argc, &argv, file, help);
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

  static PetscErrorCode get_gibbs_event(PetscLogEvent *evt) {
    static PetscLogEvent gibbs_event;

    PetscFunctionBeginUser;
    if (!gibbs_event_registered) {
      PetscClassId classid;
      PetscCall(PetscClassIdRegister("ParMGMC", &classid));
      PetscCall(PetscLogEventRegister("Gibbs", classid, &gibbs_event));

      gibbs_event_registered = true;
    }

    *evt = gibbs_event;

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~PetscHelper() { PetscFinalize(); }

private:
  PetscHelper(int *argc, char ***argv, const char *file, const char *help) {
    PetscFunctionBeginUser;

    PetscCallVoid(PetscInitialize(argc, argv, file, help));

    PetscFunctionReturnVoid();
  }

  inline static bool rng_event_registered = false;
  inline static bool gibbs_event_registered = false;
};
} // namespace parmgmc
