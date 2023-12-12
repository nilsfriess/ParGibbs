#pragma once

#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/sor.hh"
#include "parmgmc/samplers/sor_preconditioner.hh"

#include <iostream>
#include <memory>
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
  template <class MatAssembler>
  MultigridSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
                   std::size_t n_levels, MatAssembler &&mat_assembler)
      : ops(n_levels) {
    auto call = [&](auto err) { PetscCallAbort(MPI_COMM_WORLD, err); };

    PetscFunctionBeginUser;

    /* As in the SORSampler, we create a full Krylov solver but set it to only
     * run the (Multigrid) preconditioner. */
    call(KSPCreate(MPI_COMM_WORLD, &ksp));
    call(KSPSetType(ksp, KSPRICHARDSON));
    call(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));
    call(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    call(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));

    PC pc;
    call(KSPGetPC(ksp, &pc));
    call(PCSetType(pc, PCMG));

    call(PCMGSetLevels(pc, n_levels, NULL));

    // Don't coarsen operators using Galerkin product, but rediscretize (see
    // below)
    call(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    call(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE));
    call(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    call(PCMGSetNumberSmooth(pc, 2));

    // Create hierachy of meshes and operators
    for (std::size_t level = 0; level < n_levels - 1; ++level)
      ops[level] = std::make_shared<GridOperator>();

    ops[n_levels - 1] = grid_operator;

    for (std::size_t level = n_levels - 1; level > 0; --level) {
      call(DMCoarsen(ops[level]->dm, MPI_COMM_NULL, &(ops[level - 1]->dm)));

      call(DMCreateMatrix(ops[level - 1]->dm, &(ops[level - 1]->mat)));
      call(mat_assembler(ops[level - 1]->mat, ops[level - 1]->dm));
    }

    // Setup multigrid sampler
    for (std::size_t level = 0; level < n_levels; ++level) {
      KSP ksp_level;

      /* We configure the smoother on each level to be a preconditioned
         Richardson smoother with a (stochastic) Gauss-Seidel preconditioner. */
      call(PCMGGetSmoother(pc, level, &ksp_level));
      call(KSPSetType(ksp_level, KSPRICHARDSON));
      call(KSPSetOperators(ksp_level, ops[level]->mat, ops[level]->mat));
      call(KSPSetInitialGuessNonzero(ksp_level, PETSC_TRUE));

      // Set preconditioner to be stochastic SOR
      PC pc_level;
      call(KSPGetPC(ksp_level, &pc_level));
      call(PCSetType(pc_level, PCSHELL));

      auto *context =
          new SORRichardsonContext<Engine>(engine, ops[level]->mat, 1.);

      call(PCShellSetContext(pc_level, context));
      call(
          PCShellSetApplyRichardson(pc_level, sor_pc_richardson_apply<Engine>));
      call(PCShellSetDestroy(pc_level, sor_pc_richardson_destroy<Engine>));

      if (level > 0) {
        Mat grid_transfer;
        DM dm_fine = ops[level]->dm;
        DM dm_coarse = ops[level - 1]->dm;

        call(DMCreateInterpolation(dm_coarse, dm_fine, &grid_transfer, NULL));
        call(PCMGSetInterpolation(pc, level, grid_transfer));
        call(MatDestroy(&grid_transfer));
      }
    }

    call(PCSetFromOptions(pc));

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

  KSP ksp;
};
} // namespace parmgmc
