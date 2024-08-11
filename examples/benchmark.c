#include <math.h>
#include <mpi.h>

#include <petscdm.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewertypes.h>
#include <time.h>

#include "parmgmc/iact.h"
#include "parmgmc/ms.h"
#include "parmgmc/obs.h"
#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/pc/pc_gibbs.h"

typedef struct {
  PetscBool run_gibbs, run_mgmc_coarse_chol, run_mgmc_coarse_gibbs, run_cholsampler;
  PetscBool measure_sampling_time, measure_iact;
  PetscBool with_lr;
  PetscInt  n_burnin, n_samples;
} *Parameters;

static PetscErrorCode ParametersCreate(Parameters *params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscNew(params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParametersDestroy(Parameters *params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(*params));
  params = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParametersRead(Parameters params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_gibbs", &params->run_gibbs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_mgmc_coarse_gibbs", &params->run_mgmc_coarse_gibbs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_mgmc_coarse_chol", &params->run_mgmc_coarse_chol, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_cholsampler", &params->run_cholsampler, NULL));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_burnin", &params->n_burnin, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_samples", &params->n_samples, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-measure_sampling_time", &params->measure_sampling_time, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-measure_iact", &params->measure_iact, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-with_lr", &params->with_lr, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PRINTBOOL(x) x ? "true" : "false"

static PetscErrorCode ParametersView(Parameters params, PetscViewer viewer)
{
  PetscFunctionBeginUser;
  PetscCheck(viewer == PETSC_VIEWER_STDOUT_WORLD || viewer == PETSC_VIEWER_STDOUT_SELF, MPI_COMM_WORLD, PETSC_ERR_SUP, "Viewer not supported");

  PetscCall(PetscViewerASCIIPrintf(viewer, "Samplers to run\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Gibbs:                 %s\n", PRINTBOOL(params->run_gibbs)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t MGMC (coarse Gibbs)    %s\n", PRINTBOOL(params->run_mgmc_coarse_gibbs)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t MGMC (coarse Cholesky) %s\n", PRINTBOOL(params->run_mgmc_coarse_chol)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Cholesky:              %s\n", PRINTBOOL(params->run_cholsampler)));

  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of samples\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Burn-in: %d\n", params->n_burnin));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Actual:  %d\n", params->n_samples));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InfoView(Mat A, Parameters params, PetscViewer viewer)
{
  PetscInt    n;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCheck(viewer == PETSC_VIEWER_STDOUT_WORLD || viewer == PETSC_VIEWER_STDOUT_SELF, MPI_COMM_WORLD, PETSC_ERR_SUP, "Viewer not supported");
  PetscCall(PetscViewerASCIIPrintf(viewer, "################################################################################\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "                              Benchmark parameters\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "################################################################################\n"));
  PetscCall(ParametersView(params, viewer));

  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Problem size (degrees of freedom): %d\n\n", n));
  PetscCall(MPI_Comm_size(MPI_COMM_WORLD, &size));

  PetscCall(PetscOptionsView(NULL, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GibbsSamplerCreate(Mat A, PetscRandom pr, Parameters params, KSP *ksp)
{
  (void)params;

  PC pc;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(MPI_COMM_WORLD, ksp));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCGIBBS));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetUp(*ksp));
  PetscCall(PCGibbsSetSweepType(pc, SOR_FORWARD_SWEEP));
  PetscCall(PCSetPetscRandom(pc, pr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CholeskySamplerCreate(Mat A, PetscRandom pr, Parameters params, KSP *ksp)
{
  (void)params;

  PC pc;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(MPI_COMM_WORLD, ksp));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCCHOLSAMPLER));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetUp(*ksp));
  PetscCall(PCSetPetscRandom(pc, pr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Note: This must be set from the options databse using the prefix mgmc
static PetscErrorCode MGMCSamplerCreate(Mat A, DM dm, PetscRandom pr, Parameters params, PCType coarse, KSP *ksp)
{
  (void)params;

  PC        pc;
  PetscBool flag;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(MPI_COMM_WORLD, ksp));
  PetscCall(KSPSetDM(*ksp, dm));
  PetscCall(KSPSetDMActive(*ksp, PETSC_FALSE));
  PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCGAMGMC));
  PetscCall(PetscStrcmp(coarse, PCGIBBS, &flag));
  if (flag) {
    PetscCall(PetscOptionsSetValue(NULL, "-gamgmc_mg_coarse_ksp_type", "richardson"));
    PetscCall(PetscOptionsSetValue(NULL, "-gamgmc_mg_coarse_pc_type", "gibbs"));
    PetscCall(PetscOptionsSetValue(NULL, "-gamgmc_mg_coarse_ksp_max_it", "2"));
  } else {
    PetscCall(PetscStrcmp(coarse, PCCHOLSAMPLER, &flag));
    if (flag) {
      PetscCall(PetscOptionsSetValue(NULL, "-gamgmc_mg_coarse_ksp_type", "preonly"));
      PetscCall(PetscOptionsSetValue(NULL, "-gamgmc_mg_coarse_pc_type", "cholsampler"));
    } else PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Unknown coarse sampler. Using default"));
  }
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetUp(*ksp));
  PetscCall(PCSetPetscRandom(pc, pr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MGMCSamplerGetNumLevels(KSP ksp, PetscInt *levels)
{
  PC pc, mg;

  PetscFunctionBeginUser;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGAMGMCGetInternalPC(pc, &mg));
  PetscCall(PCMGGetLevels(mg, levels));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Burnin(KSP ksp, Vec b, Parameters params)
{
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, params->n_burnin));
  PetscCall(VecDuplicate(b, &x));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Sample(KSP ksp, Vec b, Parameters params)
{
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, params->n_samples));
  PetscCall(VecDuplicate(b, &x));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SaveSample(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx)
{
  (void)rnorm;

  PetscScalar norm, *qois = (PetscScalar *)ctx;
  Vec         x;

  PetscFunctionBeginUser;
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(VecNorm(x, NORM_2, &norm));
  qois[it] = norm;
  *reason  = KSP_CONVERGED_ITERATING;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TIME(functioncall, name, time) \
  do { \
    double _starttime, _endtime; \
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting %s... ", name)); \
    _starttime = MPI_Wtime(); \
    PetscCall(functioncall); \
    _endtime = MPI_Wtime(); \
    PetscCall(PetscPrintf(MPI_COMM_WORLD, " done. Took %.4fs.\n", _endtime - _starttime)); \
    *time = _endtime - _starttime; \
  } while (0);

int main(int argc, char *argv[])
{
  Parameters  params;
  MS          ms;
  Mat         A;
  Vec         b;
  DM          dm;
  PetscRandom pr;
  double      time;

  PetscCall(PetscInitialize(&argc, &argv, "../examples/benchmarkrc", NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(ParametersCreate(&params));
  PetscCall(ParametersRead(params));

  {
    double starttime, endtime;

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting setup..."));
    starttime = MPI_Wtime();
    PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
    PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
    PetscCall(MSSetFromOptions(ms));
    PetscCall(MSSetUp(ms));
    PetscCall(MSGetPrecisionMatrix(ms, &A));
    PetscCall(MSGetDM(ms, &dm));

    PetscCall(DMCreateGlobalVector(dm, &b));
    PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &pr));
    PetscCall(PetscRandomSetType(pr, PARMGMC_ZIGGURAT));
    PetscCall(PetscRandomSetSeed(pr, 2));
    PetscCall(PetscRandomSeed(pr));

    if (params->with_lr) {
      const PetscInt nobs = 3;
      PetscScalar    obs[3 * nobs], radii[nobs], obsvals[nobs], obsval = 1;
      Mat            A2, B;
      Vec            S;

      obs[0] = 0.25;
      obs[1] = 0.25;
      obs[2] = 0.75;
      obs[3] = 0.75;
      obs[4] = 0.25;
      obs[5] = 0.75;

      PetscCall(PetscOptionsGetReal(NULL, NULL, "-obsval", &obsval, NULL));
      obsvals[0] = obsval;
      radii[0]   = 0.1;
      obsvals[1] = obsval;
      radii[1]   = 0.1;
      obsvals[2] = obsval;
      radii[2]   = 0.1;

      PetscCall(MakeObservationMats(dm, nobs, 1e-6, obs, radii, obsvals, &B, &S, NULL));
      PetscCall(MatCreateLRC(A, B, S, NULL, &A2));
      PetscCall(MatDestroy(&B));
      PetscCall(VecDestroy(&S));

      A = A2;
    }

    endtime = MPI_Wtime();
    PetscCall(PetscPrintf(MPI_COMM_WORLD, " done. Took %.4fs.\n\n", endtime - starttime));
  }

  if (params->measure_sampling_time) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "                              Measure sampling time\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    if (params->run_gibbs) {
      KSP ksp;

      TIME(GibbsSamplerCreate(A, pr, params, &ksp), "Gibbs setup", &time);
      TIME(Burnin(ksp, b, params), "Gibbs burn-in", &time);
      TIME(Sample(ksp, b, params), "Gibbs sampling", &time);
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.6f\n\n", time / params->n_samples * 1000));

      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_mgmc_coarse_gibbs) {
      KSP      ksp;
      PetscInt levels;

      TIME(MGMCSamplerCreate(A, dm, pr, params, PCGIBBS, &ksp), "MGMC (Coarse Gibbs) setup", &time);
      PetscCall(MGMCSamplerGetNumLevels(ksp, &levels));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t Using %d levels\n", levels));
      TIME(Burnin(ksp, b, params), "MGMC (Coarse Gibbs) burn-in", &time);
      TIME(Sample(ksp, b, params), "MGMC (Coarse Gibbs) sampling", &time);
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.6f\n\n", time / params->n_samples * 1000));

      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_mgmc_coarse_chol) {
      KSP      ksp;
      PetscInt levels;

      TIME(MGMCSamplerCreate(A, dm, pr, params, PCCHOLSAMPLER, &ksp), "MGMC (Coarse Cholesky) setup", &time);
      PetscCall(MGMCSamplerGetNumLevels(ksp, &levels));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t Using %d levels\n", levels));
      TIME(Burnin(ksp, b, params), "MGMC (Coarse Cholesky) burn-in", &time);
      TIME(Sample(ksp, b, params), "MGMC (Coarse Cholesky) sampling", &time);
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.6f\n\n", time / params->n_samples * 1000));

      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_cholsampler) {
      KSP ksp;

      TIME(CholeskySamplerCreate(A, pr, params, &ksp), "Cholesky setup", &time);
      TIME(Burnin(ksp, b, params), "Cholesky burn-in", &time);
      TIME(Sample(ksp, b, params), "Cholesky sampling", &time);
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.6f\n\n", time / params->n_samples * 1000));

      PetscCall(KSPDestroy(&ksp));
    }
  }

  if (params->measure_iact) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "                                  Measure IACT\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

    if (params->run_gibbs) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;
      PetscBool    valid;

      TIME(GibbsSamplerCreate(A, pr, params, &ksp), "Gibbs setup", &time);
      PetscCall(PetscCalloc1(params->n_samples, &qois));

      TIME(Burnin(ksp, b, params), "Gibbs burn-in", &time);
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));
      TIME(Sample(ksp, b, params), "Gibbs sampling", &time);

      PetscCall(IACT(params->n_samples, qois, &tau, &valid));
      if (!valid) PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Chain is too short to give reliable IACT estimate (need at least %d)\n", (int)ceil(50 * tau)));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Gibbs IACT: %.5f\n", tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per independent sample [ms]: %.6f\n\n", PetscMax(tau, 1) * time / params->n_samples * 1000));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_cholsampler) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;
      PetscBool    valid;

      TIME(CholeskySamplerCreate(A, pr, params, &ksp), "Cholesky setup", &time);
      PetscCall(PetscCalloc1(params->n_samples, &qois));

      TIME(Burnin(ksp, b, params), "Cholesky burn-in", &time);
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));
      TIME(Sample(ksp, b, params), "Cholesky sampling", &time);

      PetscCall(IACT(params->n_samples, qois, &tau, &valid));
      if (!valid) PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Chain is too short to give reliable IACT estimate (need at least %d)\n", (int)ceil(50 * tau)));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Cholesky IACT: %.5f\n", tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per independent sample [ms]: %.6f\n\n", PetscMax(tau, 1) * time / params->n_samples * 1000));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_mgmc_coarse_gibbs) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;
      PetscBool    valid;
      PetscInt     levels;

      TIME(MGMCSamplerCreate(A, dm, pr, params, PCGIBBS, &ksp), "MGMC (Coarse Gibbs) setup", &time);
      PetscCall(MGMCSamplerGetNumLevels(ksp, &levels));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t Using %d levels\n", levels));
      PetscCall(PetscCalloc1(params->n_samples, &qois));

      TIME(Burnin(ksp, b, params), "MGMC (Coarse Gibbs) burn-in", &time);
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));
      TIME(Sample(ksp, b, params), "MGMC (Coarse Gibbs) sampling", &time);

      PetscCall(IACT(params->n_samples, qois, &tau, &valid));
      if (!valid) PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Chain is too short to give reliable IACT estimate (need at least %d)\n", (int)ceil(50 * tau)));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "MGMC (Coarse Gibbs) IACT: %.5f\n", tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per independent sample [ms]: %.6f\n\n", PetscMax(tau, 1) * time / params->n_samples * 1000));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_mgmc_coarse_chol) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;
      PetscBool    valid;
      PetscInt     levels;

      TIME(MGMCSamplerCreate(A, dm, pr, params, PCCHOLSAMPLER, &ksp), "MGMC (Coarse Cholesky) setup", &time);
      PetscCall(MGMCSamplerGetNumLevels(ksp, &levels));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t Using %d levels\n", levels));
      PetscCall(PetscCalloc1(params->n_samples, &qois));

      TIME(Burnin(ksp, b, params), "MGMC (Coarse Cholesky) burn-in", &time);
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));
      TIME(Sample(ksp, b, params), "MGMC (Coarse Cholesky) sampling", &time);

      PetscCall(IACT(params->n_samples, qois, &tau, &valid));
      if (!valid) PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Chain is too short to give reliable IACT estimate (need at least %d)\n", (int)ceil(50 * tau)));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "MGMC (Coarse Cholesky) IACT: %.5f\n", tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per independent sample [ms]: %.6f\n\n", PetscMax(tau, 1) * time / params->n_samples * 1000));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }
  }

  PetscCall(InfoView(A, params, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscRandomDestroy(&pr));
  PetscCall(MSDestroy(&ms));
  if (params->with_lr) PetscCall(MatDestroy(&A));
  PetscCall(ParametersDestroy(&params));
  PetscCall(PetscFinalize());
}
