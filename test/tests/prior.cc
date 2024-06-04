/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

// RUN: %cxx %s -o %t %flags && %t -ksp_type richardson -pc_type gibbs -ksp_norm_type none -ksp_max_it 1000000
// RUN: %cxx %s -o %t %flags && %t -ksp_type richardson -pc_type gibbs -pc_gibbs_omega 1.4 -ksp_norm_type none -ksp_max_it 1000000
// RUN: %cxx %s -o %t %flags && %t -ksp_type richardson -pc_type gmgmc -da_grid_x 5 -da_grid_y 5 -gmgmc_mg_levels_ksp_type richardson -gmgmc_mg_levels_pc_type gibbs -gmgmc_mg_coarse_ksp_type richardson -gmgmc_mg_coarse_pc_type gibbs -gmgmc_mg_coarse_ksp_max_it 4 -gmgmc_mg_levels_ksp_max_it 2 -da_refine 1 -ksp_norm_type none -gmgmc_pc_mg_levels 3 -ksp_max_it 1000000

#include <petsc.h>
#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>
#include <petscmath.h>
#include <petscsystypes.h>

PetscErrorCode SampleCallback(PetscInt it, Vec y, void *ctx)
{
  Vec *mean = (Vec *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecScale(*mean, it));
  PetscCall(VecAXPY(*mean, 1., y));
  PetscCall(VecScale(*mean, 1. / (it + 1)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM        da;
  Mat       A;
  Vec       b, x, f, mean;
  KSP       ksp;
  PC        pc;
  PetscReal err;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 1, A));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(MatCreateVecs(A, &mean, NULL));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetSampleCallback(pc, SampleCallback, &mean));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(VecSet(b, 1));
  PetscCall(VecSet(x, 1));
  PetscCall(MatMult(A, b, f));

  PetscCall(KSPSolve(ksp, f, x));

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
