#pragma once

#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/gibbs.hh"

#include <iostream>
#include <memory>
#include <petscerror.h>
#include <random>

#include <mpi.h>

#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class MultigridSampler {
public:
  MultigridSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
                   std::size_t n_levels)
      : ops(n_levels) {
    PetscFunctionBeginUser;

    /* Create a full KSP solver but set it to only run the (Multigrid)
     * preconditioner. */
    PetscCallVoid(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCallVoid(KSPSetType(ksp, KSPRICHARDSON));
    PetscCallVoid(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));
    PetscCallVoid(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCallVoid(
        KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));

    PC pc;
    PetscCallVoid(KSPGetPC(ksp, &pc));
    PetscCallVoid(PCSetType(pc, PCMG));

    PetscCallVoid(PCMGSetLevels(pc, n_levels, NULL));

    // PetscCallVoid(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCallVoid(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    PetscCallVoid(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    PetscCallVoid(PCMGSetNumberSmooth(pc, 2));

    PetscCallVoid(PCSetDM(pc, grid_operator->dm));
    PetscCallVoid(PCSetOperators(pc, grid_operator->mat, grid_operator->mat));
    PetscCallVoid(PCSetUp(pc));

    for (std::size_t level = 0; level < n_levels; ++level) {
      KSP ksp_level;
      PetscCallVoid(PCMGGetSmoother(pc, level, &ksp_level));

      ops[level] = std::make_shared<GridOperator>();
      PetscCallVoid(KSPGetOperators(ksp_level, &ops[level]->mat, NULL));
      PetscCallVoid(KSPGetDM(ksp_level, &ops[level]->dm));

      PetscCallVoid(ops[level]->color_general());
      MatType type;
      PetscCallVoid(MatGetType(ops[level]->mat, &type));
      if (std::strcmp(type, MATMPIAIJ) == 0) {
        PetscCallVoid(ops[level]->create_rb_scatter());
      }
    }

    smoothers.resize(n_levels);
    for (std::size_t level = 0; level < n_levels; ++level) {
      KSP ksp_level;

      { // Pre smoothers/samplers
        PetscCallVoid(PCMGGetSmootherDown(pc, level, &ksp_level));
        PetscCallVoid(KSPSetType(ksp_level, KSPRICHARDSON));
        PetscCallVoid(KSPSetInitialGuessNonzero(ksp_level, PETSC_TRUE));

        // Set preconditioner to be stochastic SOR
        PC pc_level;
        PetscCallVoid(KSPGetPC(ksp_level, &pc_level));
        PetscCallVoid(PCSetType(pc_level, PCSHELL));

        smoothers[level].first = std::make_shared<GibbsSampler<Engine>>(
            ops[level], engine, 1., GibbsSweepType::FORWARD);

        PetscCallVoid(
            PCShellSetContext(pc_level, smoothers[level].first.get()));
        PetscCallVoid(
            PCShellSetApplyRichardson(pc_level, PCShellCallback_Gibbs<Engine>));
      }

      if (level > 0) { // Post smoothers/samplers
        PetscCallVoid(PCMGGetSmootherUp(pc, level, &ksp_level));
        PetscCallVoid(KSPSetType(ksp_level, KSPRICHARDSON));
        PetscCallVoid(
            KSPSetOperators(ksp_level, ops[level]->mat, ops[level]->mat));
        PetscCallVoid(KSPSetInitialGuessNonzero(ksp_level, PETSC_TRUE));
        PetscCallVoid(KSPSetTolerances(
            ksp_level, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));

        // Set preconditioner to be stochastic SOR
        PC pc_level;
        PetscCallVoid(KSPGetPC(ksp_level, &pc_level));
        PetscCallVoid(PCSetType(pc_level, PCSHELL));

        smoothers[level].second = std::make_shared<GibbsSampler<Engine>>(
            ops[level], engine, 1., GibbsSweepType::BACKWARD);

        PetscCallVoid(
            PCShellSetContext(pc_level, smoothers[level].second.get()));
        PetscCallVoid(
            PCShellSetApplyRichardson(pc_level, PCShellCallback_Gibbs<Engine>));
      }
    }

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < n_samples; ++n)
      PetscCall(KSPSolve(ksp, rhs, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~MultigridSampler() { KSPDestroy(&ksp); }

private:
  std::vector<std::shared_ptr<GridOperator>> ops;
  std::vector<std::pair<std::shared_ptr<GibbsSampler<Engine>>,
                        std::shared_ptr<GibbsSampler<Engine>>>>
      smoothers;

  KSP ksp;
};
} // namespace parmgmc
