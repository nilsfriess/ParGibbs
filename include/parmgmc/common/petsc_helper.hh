#pragma once

#include <petsclog.h>
#include <petscsys.h>
#include <petscsystypes.h>

namespace parmgmc {
struct PetscHelper {
  static void init(int &argc, char **&argv, const char *file = nullptr,
                   const char *help = nullptr) {
    static PetscHelper helper(&argc, &argv, file, help);
  }

  static PetscErrorCode begin_rng_event() {
    PetscFunctionBeginUser;

    PetscLogEvent rng_event = PetscHelper::get_rng_event();
    PetscCall(
        PetscLogEventBegin(rng_event, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode end_rng_event() {
    PetscFunctionBeginUser;

    PetscLogEvent rng_event = PetscHelper::get_rng_event();
    PetscCall(PetscLogEventEnd(rng_event, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode begin_gibbs_event() {
    PetscFunctionBeginUser;

    PetscLogEvent gibbs_event = PetscHelper::get_gibbs_event();
    PetscCall(
        PetscLogEventBegin(gibbs_event, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode end_gibbs_event() {
    PetscFunctionBeginUser;

    PetscLogEvent gibbs_event = PetscHelper::get_gibbs_event();
    PetscCall(
        PetscLogEventEnd(gibbs_event, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscLogEvent get_rng_event() {
    static PetscLogEvent rng_event = []() {
      PetscClassId classid;
      PetscLogEvent event;
      PetscClassIdRegister("ParMGMC", &classid);
      PetscLogEventRegister("RNG", classid, &event);

      return event;
    }();

    return rng_event;
  }

  static PetscLogEvent get_gibbs_event() {
    static PetscLogEvent gibbs_event = []() {
      PetscClassId classid;
      PetscLogEvent event;
      PetscClassIdRegister("ParMGMC", &classid);
      PetscLogEventRegister("Gibbs", classid, &event);

      return event;
    }();

    return gibbs_event;
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
