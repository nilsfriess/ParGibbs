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
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>

#include "parmgmc/iact.h"
#include "parmgmc/parmgmc.h"

#include "params.hh"
#include "problems.hh"

#if __has_include(<mfem.hpp>)
  #define PARMGMC_HAVE_MFEM
#endif

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
  PetscCall(MatGetSize(A, &n, nullptr));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Problem size (degrees of freedom): %d\n\n", n));
  PetscCall(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Running on %d MPI ranks\n\n", size));

  PetscCall(PetscOptionsView(nullptr, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SamplerCreate(Mat A, DM dm, PetscRandom pr, Parameters params, KSP *ksp)
{
  (void)params;

  PC pc;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(MPI_COMM_WORLD, ksp));
  PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetConvergenceTest(*ksp, KSPSkipConverged, nullptr, nullptr));
  PetscCall(KSPSetOperators(*ksp, A, A));
  if (dm) {
    PetscCall(KSPSetDM(*ksp, dm));
    PetscCall(KSPSetDMActive(*ksp, PETSC_FALSE));
  }
  PetscCall(KSPSetUp(*ksp));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetPetscRandom(pc, pr));
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

struct SampleCtx {
  PetscScalar *qois;
  Vec          meas_vec;
};

static PetscErrorCode SaveSample(PetscInt it, Vec y, void *ctx)
{
  SampleCtx  *sctx = (SampleCtx *)ctx;
  PetscScalar qoi, *qois = sctx->qois;
  Vec         meas_vec = sctx->meas_vec;

  PetscFunctionBeginUser;
  PetscCall(VecDot(y, meas_vec, &qoi));
  qois[it] = qoi;
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
  Mat         A;
  DM          dm = nullptr; // Only set when building Mat with PETSc
  KSP         ksp;
  PC          pc;
  Vec         b, meas_vec;
  PetscRandom pr;
  double      time;
  PetscMPIInt rank;
#ifdef PARMGMC_HAVE_MFEM
  PetscBool mfem = PETSC_FALSE;
#endif
  PetscBool seed_from_dev_random;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(ParMGMCInitialize());

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "#############                Benchmark Test Program                #############\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

  PetscCall(ParametersCreate(&params));
  PetscCall(ParametersRead(params));

  PetscCheck(params->measure_iact || params->measure_sampling_time, MPI_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Pass at least one of -measure_sampling_time or -measure_iact");

  PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &pr));
  PetscCall(PetscRandomSetType(pr, PARMGMC_ZIGGURAT));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-seed_from_dev_random", &seed_from_dev_random, nullptr));
  if (seed_from_dev_random) {
    int           dr = open("/dev/random", O_RDONLY);
    unsigned long seed;
    read(dr, &seed, sizeof(seed));
    close(dr);
    PetscCall(PetscRandomSetSeed(pr, seed));
  } else {
    PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    PetscCall(PetscRandomSetSeed(pr, 1 + rank));
  }
  PetscCall(PetscRandomSeed(pr));

#ifdef PARMGMC_HAVE_MFEM
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-mfem", &mfem, nullptr));
  if (mfem) {
    TIME(CreateMatrixMFEM(params, &A), "Assembling matrix", &time);
  } else
#endif
    TIME(CreateMatrixPetsc(params, &A, &meas_vec, &dm), "Assembling matrix", &time);
  PetscCall(MatCreateVecs(A, nullptr, &b));

  TIME(SamplerCreate(A, dm, pr, params, &ksp), "Setup sampler", &time);
  PetscCall(KSPGetPC(ksp, &pc));

  if (params->measure_sampling_time) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "                              Measure sampling time\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

    TIME(Burnin(ksp, b, params), "Burn-in", &time);
    TIME(Sample(ksp, b, params), "Sampling", &time);

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.6f\n\n", time / params->n_samples * 1000));
  }

  if (params->measure_iact) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "                                  Measure IACT\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

    PetscScalar tau = 0;
    PetscBool   valid;

    SampleCtx *ctx = new SampleCtx;
    PetscCall(PetscCalloc1(params->n_samples + 1, &ctx->qois));
    ctx->meas_vec = meas_vec;

    TIME(Burnin(ksp, b, params), "Burn-in", &time);

    PetscCall(PCSetSampleCallback(pc, SaveSample, ctx, nullptr));
    // PetscCall(KSPSetConvergenceTest(ksp, SaveSample, qois, NULL));
    TIME(Sample(ksp, b, params), "Sampling", &time);

    PetscBool    print_acf;
    PetscScalar *acf;

    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-print_acf", &print_acf, nullptr));
    PetscCall(IACT(params->n_samples, ctx->qois, &tau, print_acf ? &acf : nullptr, &valid));
    if (!valid) PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Chain is too short to give reliable IACT estimate (need at least %d)\n", (int)ceil(50 * tau)));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "IACT: %.5f\n", tau));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per independent sample [ms]: %.6f\n\n", PetscMax(tau, 1) * time / params->n_samples * 1000));
    if (print_acf) {
      FILE *fptr;
      fptr = fopen("acf.txt", "w");
      for (PetscInt i = 0; i < params->n_samples; i++) PetscCall(PetscFPrintf(MPI_COMM_WORLD, fptr, "%.6f\n", acf[i]));
      fclose(fptr);
    }

    PetscCall(PetscFree(ctx->qois));
    PetscCall(VecDestroy(&meas_vec));
    delete ctx;
  }

  PetscCall(InfoView(A, params, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PCView(pc, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&b));
  PetscCall(PetscRandomDestroy(&pr));
  PetscCall(MatDestroy(&A));
  if (dm) PetscCall(DMDestroy(&dm));
  PetscCall(ParametersDestroy(&params));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
}
