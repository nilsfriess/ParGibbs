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

  static PetscErrorCode beginRngEvent() {
    PetscFunctionBeginUser;

    PetscLogEvent rngEvent = PetscHelper::getRngEvent();
    PetscCall(
        PetscLogEventBegin(rngEvent, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode endRngEvent() {
    PetscFunctionBeginUser;

    PetscLogEvent rngEvent = PetscHelper::getRngEvent();
    PetscCall(PetscLogEventEnd(rngEvent, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode beginGibbsEvent() {
    PetscFunctionBeginUser;

    PetscLogEvent gibbsEvent = PetscHelper::getGibbsEvent();
    PetscCall(
        PetscLogEventBegin(gibbsEvent, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode endGibbsEvent() {
    PetscFunctionBeginUser;

    PetscLogEvent gibbsEvent = PetscHelper::getGibbsEvent();
    PetscCall(
        PetscLogEventEnd(gibbsEvent, nullptr, nullptr, nullptr, nullptr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscLogEvent getRngEvent() {
    static PetscLogEvent rngEvent = []() {
      PetscClassId classid;
      PetscLogEvent event;
      PetscClassIdRegister("ParMGMC", &classid);
      PetscLogEventRegister("RNG", classid, &event);

      return event;
    }();

    return rngEvent;
  }

  static PetscLogEvent getGibbsEvent() {
    static PetscLogEvent gibbsEvent = []() {
      PetscClassId classid;
      PetscLogEvent event;
      PetscClassIdRegister("ParMGMC", &classid);
      PetscLogEventRegister("Gibbs", classid, &event);

      return event;
    }();

    return gibbsEvent;
  }

  ~PetscHelper() { PetscFinalize(); }

private:
  PetscHelper(int *argc, char ***argv, const char *file, const char *help) {
    PetscFunctionBeginUser;

    PetscCallVoid(PetscInitialize(argc, argv, file, help));

    PetscFunctionReturnVoid();
  }

  inline static bool rngEventRegistered = false;
  inline static bool gibbsEventRegistered = false;
};
} // namespace parmgmc
