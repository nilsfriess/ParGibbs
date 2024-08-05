#include "parmgmc/ms.h"
#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_gibbs.h"

#include <complex.h>
#include <petsc/private/pcmgimpl.h>
#include <petscdm.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscksp.h>
#include <mpi.h>

#include <fftw3.h>

typedef struct {
  PetscBool run_gibbs, run_mgmc, run_cholsampler;
  PetscBool measure_sampling_time, measure_iact;
  PetscInt  n_burnin, n_samples;
} *Parameters;

static PetscErrorCode ParametersCreate(Parameters *params)
{
  PetscFunctionBeginUser;
  PetscCall(PetscNew(params));
  (*params)->run_gibbs             = PETSC_FALSE;
  (*params)->run_mgmc              = PETSC_FALSE;
  (*params)->run_cholsampler       = PETSC_FALSE;
  (*params)->measure_sampling_time = PETSC_FALSE;
  (*params)->measure_iact          = PETSC_FALSE;
  (*params)->n_burnin              = 0;
  (*params)->n_samples             = 0;
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
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_mgmc", &params->run_mgmc, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_cholsampler", &params->run_cholsampler, NULL));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_burnin", &params->n_burnin, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_samples", &params->n_samples, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-measure_sampling_time", &params->measure_sampling_time, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-measure_iact", &params->measure_iact, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PRINTBOOL(x) x ? "true" : "false"

static PetscErrorCode ParametersView(Parameters params, PetscViewer viewer)
{
  PetscFunctionBeginUser;
  PetscCheck(viewer == PETSC_VIEWER_STDOUT_WORLD || viewer == PETSC_VIEWER_STDOUT_SELF, MPI_COMM_WORLD, PETSC_ERR_SUP, "Viewer not supported");

  PetscCall(PetscViewerASCIIPrintf(viewer, "Samplers to run\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Gibbs:    %s\n", PRINTBOOL(params->run_gibbs)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t MGMC:     %s\n", PRINTBOOL(params->run_mgmc)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Cholesky: %s\n", PRINTBOOL(params->run_cholsampler)));

  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of samples\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Burn-in: %d\n", params->n_burnin));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\t Actual:  %d\n", params->n_samples));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InfoView(Mat A, Parameters params, PetscViewer viewer)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCheck(viewer == PETSC_VIEWER_STDOUT_WORLD || viewer == PETSC_VIEWER_STDOUT_SELF, MPI_COMM_WORLD, PETSC_ERR_SUP, "Viewer not supported");
  PetscCall(PetscViewerASCIIPrintf(viewer, "####################\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Benchmark parameters\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "####################\n"));
  PetscCall(ParametersView(params, viewer));

  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Degrees of freedom: %d\n", n));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GibbsSamplerCreate(Mat A, Parameters params, KSP *ksp)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CholeskySamplerCreate(Mat A, Parameters params, KSP *ksp)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Note: This must be set from the options databse using the prefix mgmc
static PetscErrorCode MGMCSamplerCreate(Mat A, DM dm, Parameters params, KSP *ksp)
{
  (void)params;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(MPI_COMM_WORLD, ksp));
  PetscCall(KSPSetOptionsPrefix(*ksp, "mgmc_"));
  PetscCall(KSPSetDM(*ksp, dm));
  PetscCall(KSPSetDMActive(*ksp, PETSC_FALSE));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetUp(*ksp));
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

static PetscErrorCode Autocorrelation(PetscInt n, const PetscScalar *x, PetscScalar **acf)
{
  fftw_plan     p;
  fftw_complex *in, *out;
  PetscScalar   mean = 0;

  PetscFunctionBeginUser;
  in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);

  for (PetscInt i = 0; i < n; ++i) mean += 1. / n * x[i];
  for (PetscInt i = 0; i < n; ++i) in[i] = x[i] - mean;

  p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  for (PetscInt i = 0; i < n; ++i) out[i] = out[i] * conj(out[i]);
  p = fftw_plan_dft_1d(n, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  PetscCall(PetscMalloc1(n, acf));
  for (PetscInt i = 0; i < n; ++i) (*acf)[i] = PetscRealPart(in[i]) / PetscRealPart(in[0]);

  fftw_free(in);
  fftw_free(out);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AutoWindow(PetscInt n, const PetscScalar *taus, PetscInt c, PetscInt *w)
{
  PetscBool flag = PETSC_FALSE;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < n; ++i) {
    if (i < c * taus[i]) {
      flag = PETSC_TRUE;
      break;
    }
  }
  if (flag) {
    flag = PETSC_FALSE;
    for (PetscInt i = 0; i < n; ++i) {
      if (i >= c * taus[i]) {
        *w   = i;
        flag = PETSC_TRUE;
        break;
      }
    }
    if (flag == PETSC_FALSE) *w = 0;
  } else *w = n - 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IACT(PetscInt n, const PetscScalar *x, PetscScalar *tau)
{
  PetscScalar *out;
  PetscInt     w;

  PetscFunctionBeginUser;
  PetscCall(Autocorrelation(n, x, &out));
  for (PetscInt i = 1; i < n; ++i) out[i] = out[i] + out[i - 1];
  for (PetscInt i = 1; i < n; ++i) out[i] = 2 * out[i] - 1;
  PetscCall(AutoWindow(n, out, 5, &w));
  *tau = out[w];
  PetscCall(PetscFree(out));
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

#define START_STAGE(log) \
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting " log "...")); \
  starttime = MPI_Wtime()

#define END_STAGE() \
  endtime = MPI_Wtime(); \
  PetscCall(PetscPrintf(MPI_COMM_WORLD, " done. Took %.4fs.\n", endtime - starttime))

#define LOG_TIME_PER_SAMPLE() PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.4f\n", ((endtime - starttime) / params->n_samples * 1000)));

int main(int argc, char *argv[])
{
  Parameters params;
  MS         ms;
  Mat        A;
  Vec        b;
  DM         dm;
  double     starttime, endtime;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(ParametersCreate(&params));
  PetscCall(ParametersRead(params));

  START_STAGE("Setup");
  {
    PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
    PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
    PetscCall(MSSetFromOptions(ms));
    PetscCall(MSSetUp(ms));
    PetscCall(MSGetPrecisionMatrix(ms, &A));
    PetscCall(MSGetDM(ms, &dm));
    PetscCall(DMCreateGlobalVector(dm, &b));
  }
  END_STAGE();

  if (params->measure_sampling_time) {
    if (params->run_gibbs) {
      KSP ksp;

      PetscCall(GibbsSamplerCreate(A, params, &ksp));

      START_STAGE("Gibbs burn-in");
      PetscCall(Burnin(ksp, b, params));
      END_STAGE();

      START_STAGE("Gibbs sampling");
      PetscCall(Sample(ksp, b, params));
      END_STAGE();
      LOG_TIME_PER_SAMPLE();

      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_mgmc) {
      KSP ksp;

      PetscCall(MGMCSamplerCreate(A, dm, params, &ksp));

      START_STAGE("MGMC burn-in");
      PetscCall(Burnin(ksp, b, params));
      END_STAGE();

      START_STAGE("MGMC sampling");
      PetscCall(Sample(ksp, b, params));
      END_STAGE();
      LOG_TIME_PER_SAMPLE();

      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_cholsampler) {
      KSP ksp;

      PetscCall(CholeskySamplerCreate(A, params, &ksp));

      START_STAGE("Cholesky burn-in");
      PetscCall(Burnin(ksp, b, params));
      END_STAGE();

      START_STAGE("Cholesky sampling");
      PetscCall(Sample(ksp, b, params));
      END_STAGE();
      LOG_TIME_PER_SAMPLE();

      PetscCall(KSPDestroy(&ksp));
    }
  }

  if (params->measure_iact) {
    if (params->run_gibbs) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;

      PetscCall(GibbsSamplerCreate(A, params, &ksp));
      START_STAGE("Gibbs burn-in");
      PetscCall(Burnin(ksp, b, params));
      END_STAGE();

      PetscCall(PetscCalloc1(params->n_samples, &qois));
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));

      START_STAGE("Gibbs sampling");
      PetscCall(Sample(ksp, b, params));
      END_STAGE();
      LOG_TIME_PER_SAMPLE();

      PetscCall(IACT(params->n_samples, qois, &tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "TAU = %.5f\n", tau));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_cholsampler) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;

      PetscCall(CholeskySamplerCreate(A, params, &ksp));
      START_STAGE("Cholesky burn-in");
      PetscCall(Burnin(ksp, b, params));
      END_STAGE();

      PetscCall(PetscCalloc1(params->n_samples, &qois));
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));

      START_STAGE("Cholesky sampling");
      PetscCall(Sample(ksp, b, params));
      END_STAGE();
      LOG_TIME_PER_SAMPLE();

      PetscCall(IACT(params->n_samples, qois, &tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "TAU = %.5f\n", tau));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }

    if (params->run_mgmc) {
      KSP          ksp;
      PetscScalar *qois, tau = 0;

      PetscCall(MGMCSamplerCreate(A, dm, params, &ksp));
      START_STAGE("MGMC burn-in");
      PetscCall(Burnin(ksp, b, params));
      END_STAGE();

      PetscCall(PetscCalloc1(params->n_samples, &qois));
      PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));

      START_STAGE("MGMC sampling");
      PetscCall(Sample(ksp, b, params));
      END_STAGE();
      LOG_TIME_PER_SAMPLE();

      PetscCall(IACT(params->n_samples, qois, &tau));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "TAU = %.5f\n", tau));

      PetscCall(PetscFree(qois));
      PetscCall(KSPDestroy(&ksp));
    }
  }

  PetscCall(InfoView(A, params, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ParametersDestroy(&params));
  PetscCall(MSDestroy(&ms));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
}
