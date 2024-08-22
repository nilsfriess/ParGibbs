/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Asses sampler convergence using the Gelman-Rubin diagnostic.
 *
 */

#include "mpi.h"
#include "mpi_proto.h"
#include "parmgmc/ms.h"
#include "parmgmc/parmgmc.h"

#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <time.h>

typedef struct {
  PetscReal **qois; // 2d array of size chains x max_samples
  PetscInt    curr_chain, curr_first_it, check_every;
  Vec         meas_vec;
} *SampleCtx;

PetscErrorCode SampleCallback(PetscInt it, Vec y, void *ctx)
{
  SampleCtx sctx = ctx;

  PetscFunctionBeginUser;
  if (it != sctx->check_every) {
    PetscCall(VecDot(y, sctx->meas_vec, &sctx->qois[sctx->curr_chain][sctx->curr_first_it + it]));
    /* PetscCall(PetscPrintf(MPI_COMM_WORLD, "[%d : %d] %.5f\n", sctx->curr_chain, sctx->curr_first_it + it, sctx->qois[sctx->curr_chain][sctx->curr_first_it + it])); */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GelmanRubin(PetscInt chains, PetscInt n, PetscReal **vals, PetscReal *gr)
{
  PetscReal *means, mean = 0, *vars, B = 0, W = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(chains, &means));
  PetscCall(PetscCalloc1(chains, &vars));

  // Chain means
  for (PetscInt i = 0; i < chains; ++i)
    for (PetscInt j = 0; j < n; ++j) means[i] += 1. / n * vals[i][j];

  // Total mean
  for (PetscInt i = 0; i < chains; ++i) mean += 1. / chains * means[i];

  // Between-chain variance
  for (PetscInt i = 0; i < chains; ++i) B += n / (chains - 1.) * (means[i] - mean) * (means[i] - mean);

  // Within-chain variance
  for (PetscInt i = 0; i < chains; ++i)
    for (PetscInt j = 0; j < n; ++j) vars[i] += 1. / (n - 1.) * (vals[i][j] - means[i]) * (vals[i][j] - means[i]);

  // Mean of vars
  for (PetscInt i = 0; i < chains; ++i) W += 1. / chains * vars[i];

  *gr = ((n - 1.) / n * W + 1. / n * B) / W;

  PetscCall(PetscFree(means));
  PetscCall(PetscFree(vars));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  MS             ms;
  Mat            A;
  MatInfo        info;
  PetscInt       check_every = 50, chains = 8, max_samples = 10000, burn_in = 1000, n;
  PetscLogDouble elapsed = 0, *times;
  PetscScalar   *grs, R_crit = 1.05;
  KSP           *samplers;
  Vec           *x, b;
  SampleCtx      ctx;
  PetscRandom    pr;
  PetscBool      write_to_file;
  PetscMPIInt    rank;
  unsigned long  seed = 0xCAFECAFE;
  PetscScalar    gr   = -1;
  DM             dm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetDM(ms, &dm));
  PetscCall(MSGetPrecisionMatrix(ms, &A));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Degrees of freedom: %d (nnz %.0f)\n", n, info.nz_used));

  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &pr));
  PetscCall(PetscRandomSetType(pr, PARMGMC_ZIGGURAT));
  PetscCall(PetscRandomSetSeed(pr, seed + rank));
  PetscCall(PetscRandomSeed(pr));

  PetscCall(PetscMalloc1(chains, &samplers));
  for (PetscInt i = 0; i < chains; ++i) {
    PC pc;

    PetscCall(KSPCreate(MPI_COMM_WORLD, &samplers[i]));
    PetscCall(KSPSetFromOptions(samplers[i]));
    PetscCall(KSPSetOperators(samplers[i], A, A));
    PetscCall(KSPSetDM(samplers[i], dm));
    PetscCall(KSPSetDMActive(samplers[i], PETSC_FALSE));
    PetscCall(KSPSetUp(samplers[i]));
    PetscCall(KSPSetNormType(samplers[i], KSP_NORM_NONE));
    PetscCall(KSPSetConvergenceTest(samplers[i], KSPConvergedSkip, NULL, NULL));
    PetscCall(KSPSetInitialGuessNonzero(samplers[i], PETSC_TRUE));
    PetscCall(KSPGetPC(samplers[i], &pc));
    PetscCall(PCSetPetscRandom(pc, pr));
  }

  // Set up sample context
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_samples", &max_samples, NULL));
  PetscCall(PetscNew(&ctx));
  PetscCall(PetscCalloc1(chains, &ctx->qois));
  for (PetscInt i = 0; i < chains; ++i) { PetscCall(PetscCalloc1(max_samples, &ctx->qois[i])); }
  ctx->check_every = check_every;

  // Set up sample vectors
  PetscCall(MatCreateVecs(A, &ctx->meas_vec, NULL));
  PetscCall(VecDuplicate(ctx->meas_vec, &b));
  PetscCall(VecSetRandom(ctx->meas_vec, pr));
  PetscCall(VecNormalize(ctx->meas_vec, NULL));
  PetscCall(PetscMalloc1(chains, &x));
  for (PetscInt i = 0; i < chains; ++i) {
    PetscReal scl;

    PetscCall(VecDuplicate(ctx->meas_vec, &x[i]));
    PetscCall(VecSetRandom(x[i], pr));
    PetscCall(PetscRandomGetValue(pr, &scl));
    PetscCall(VecScale(x[i], 1e6 * scl)); // Initial samples should be overdispersed for GR diagnostic
  }

  // Burn-in
  for (PetscInt i = 0; i < chains; ++i) {
    PetscCall(KSPSetTolerances(samplers[i], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, burn_in));
    PetscCall(KSPSolve(samplers[i], b, x[i]));
    PetscCall(KSPSetTolerances(samplers[i], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, check_every));
  }

  // Set sample callback for each sampler
  for (PetscInt i = 0; i < chains; ++i) {
    PC pc;

    PetscCall(KSPGetPC(samplers[i], &pc));
    PetscCall(PCSetSampleCallback(pc, SampleCallback, ctx, NULL));
  }

  // Sample
  PetscCall(PetscCalloc1(max_samples / check_every, &times));
  PetscCall(PetscCalloc1(max_samples / check_every, &grs));
  for (PetscInt i = 0; i < max_samples; i += check_every) {
    PetscLogDouble start = MPI_Wtime();

    ctx->curr_first_it = i;
    for (PetscInt j = 0; j < chains; ++j) {
      ctx->curr_chain = j;
      PetscCall(KSPSolve(samplers[j], b, x[j]));
    }

    elapsed += MPI_Wtime() - start;
    PetscCall(GelmanRubin(chains, i + check_every, ctx->qois, &gr));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Gelman-Rubin (%d samples, %.5fs): %.5f\n", i + check_every, elapsed, gr));
    times[i / check_every] = elapsed;
    grs[i / check_every]   = gr;

    if (gr < R_crit) {
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Converged! Took %.5f seconds\n", elapsed));
      break;
    }
  }
  if (gr >= R_crit) PetscCall(PetscPrintf(MPI_COMM_WORLD, "Did not converge in %.5f seconds\n", elapsed));

  // Print results to file
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-write_results", &write_to_file, NULL));
  if (write_to_file) {
    FILE  *fptr;
    char   filename[256];
    PCType pctype;
    PC     pc;

    PetscCall(KSPGetPC(samplers[0], &pc));
    PetscCall(PCGetType(pc, &pctype));
    PetscCall(PetscSNPrintf(filename, 256, "gelman_rubin_%s.txt", pctype));
    fptr = fopen(filename, "w");
    for (PetscInt i = 0; i < max_samples / check_every; i++) PetscCall(PetscFPrintf(MPI_COMM_WORLD, fptr, "%.6f %.6f\n", times[i] - times[0], grs[i]));
    fclose(fptr);
  }

  PetscCall(PetscFree(times));
  PetscCall(PetscFree(grs));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&ctx->meas_vec));
  for (PetscInt i = 0; i < chains; ++i) {
    PetscCall(KSPDestroy(&samplers[i]));
    PetscCall(PetscFree(ctx->qois[i]));
    PetscCall(VecDestroy(&x[i]));
  }
  PetscCall(PetscFree(x));
  PetscCall(PetscFree(ctx->qois));
  PetscCall(PetscFree(ctx));
  PetscCall(PetscFree(samplers));
  PetscCall(MSDestroy(&ms));
  PetscCall(PetscRandomDestroy(&pr));
  PetscCall(PetscFinalize());
}
