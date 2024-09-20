#pragma once

#include <petscsys.h>
#include <petscerror.h>
#include <petscsystypes.h>
#include <petscviewer.h>

typedef struct {
  PetscBool measure_sampling_time, measure_iact;
  PetscBool with_lr, est_mean_and_var;
  PetscInt  n_burnin, n_samples;
} *Parameters;

inline PetscErrorCode ParametersCreate(Parameters *params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscNew(params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode ParametersDestroy(Parameters *params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(*params));
  params = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode ParametersRead(Parameters params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-n_burnin", &params->n_burnin, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-n_samples", &params->n_samples, nullptr));

  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-measure_sampling_time", &params->measure_sampling_time, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-measure_iact", &params->measure_iact, nullptr));

  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-with_lr", &params->with_lr, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-est_mean_and_var", &params->est_mean_and_var, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PRINTBOOL(x) x ? "true" : "false"

inline PetscErrorCode ParametersView(Parameters params, PetscViewer viewer)
{
  PetscFunctionBeginUser;
  PetscCheck(viewer == PETSC_VIEWER_STDOUT_WORLD || viewer == PETSC_VIEWER_STDOUT_SELF, MPI_COMM_WORLD, PETSC_ERR_SUP, "Viewer not supported");

  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of samples\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Burn-in: %" PetscInt_FMT "\n", params->n_burnin));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Actual:  %" PetscInt_FMT "\n", params->n_samples));

  PetscFunctionReturn(PETSC_SUCCESS);
}
