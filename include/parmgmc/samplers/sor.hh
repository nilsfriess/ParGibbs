#pragma once

#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/sor_preconditioner.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class SORSampler {
public:
  SORSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
             PetscReal omega = 1.) {
    PetscFunctionBeginUser;

    using Context = SORRichardsonContext<Engine>;

    Context *context;

    if (grid_operator->has_lowrank_update) {
      context = new Context(
          engine, grid_operator->mat, grid_operator->lowrank_factor, omega);
    } else {
      context = new Context(engine, grid_operator->mat, omega);
    }

    PetscCallAbort(MPI_COMM_WORLD, init(grid_operator, context));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < n_samples; ++n)
      PetscCall(KSPSolve(ksp, rhs, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~SORSampler() { KSPDestroy(&ksp); }

private:
  PetscErrorCode init(std::shared_ptr<GridOperator> grid_operator,
                      SORRichardsonContext<Engine> *context) {
    PetscFunctionBeginUser;

    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetType(ksp, KSPRICHARDSON));
    PetscCall(
        KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCall(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));

    PetscCall(PCShellSetContext(pc, context));
    PetscCall(PCShellSetApplyRichardson(pc, sor_pc_richardson_apply<Engine>));
    PetscCall(PCShellSetDestroy(pc, sor_pc_richardson_destroy<Engine>));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  KSP ksp;
};
} // namespace parmgmc
