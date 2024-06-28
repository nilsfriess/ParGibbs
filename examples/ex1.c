/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Samples from a Gaussian random field with Matern covariance using standalone
 *  Gibbs and Cholesky samplers, and the GAMGMC Multigrid Monte Carlo sampler.
 *  The precision operator is discretised using using finite differences.
 *  For GAMGMC, this tests both the fully algrabic variant and the geometric
 *  variant.
 */

/**************************** Test specification ****************************/
// Gibbs with default omega
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gibbs -ksp_initial_guess_nonzero true

// Gibbs with custom omega
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gibbs -pc_gibbs_omega 1.4 -ksp_initial_guess_nonzero true

// Cholesky
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type preonly -pc_type cholsampler

// Algebraic MGMC with coarse Gibbs
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_coarse_pc_type gibbs -gamgmc_mg_coarse_ksp_max_it 2 -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -ksp_initial_guess_nonzero true

// Algebraic MGMC with coarse Cholesky
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type preonly -gamgmc_mg_coarse_pc_type cholsampler -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -ksp_initial_guess_nonzero true

// Geometric MGMC with coarse Gibbs
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_pc_mg_levels 3 -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_coarse_pc_type gibbs -gamgmc_mg_coarse_ksp_max_it 2 -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -ksp_initial_guess_nonzero true

// Geometric MGMC with coarse Cholesky
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_pc_mg_levels 3 -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type preonly -gamgmc_mg_coarse_pc_type cholsampler -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -ksp_initial_guess_nonzero true
/****************************************************************************/

#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>

#include <petsc.h>
#include <petscmath.h>
#include <petscsystypes.h>
#include <petscksp.h>

int main(int argc, char *argv[])
{
  DM             da;
  Mat            A;
  Vec            b, x, f, mean;
  KSP            ksp;
  PetscReal      err;
  const PetscInt n_samples = 5000000;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 1, A));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(KSPSetTolerances(ksp, 0, 0, 0, 1));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(MatCreateVecs(A, &mean, NULL));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(VecSet(b, 1));
  PetscCall(VecSet(x, 1));
  PetscCall(MatMult(A, b, f));

  for (PetscInt i = 0; i < n_samples; ++i) {
    PetscCall(KSPSolve(ksp, f, x));
    PetscCall(VecAXPY(mean, 1. / n_samples, x));
  }

  PetscCall(VecAXPY(mean, -1, b));
  PetscCall(VecNorm(mean, NORM_2, &err));

  PetscCheck(PetscIsCloseAtTol(err, 0, 0.01, 0.01), MPI_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Sample mean has not converged: got %.4f, expected %.4f", err, 0.f);

  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&da));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
}
