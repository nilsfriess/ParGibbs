#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"

#include "parmgmc/samplers/pc_cholsampler.hh"
#include "parmgmc/samplers/pc_gibbs.hh"
#include "problems.hh"

#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <random>

int main(int argc, char *argv[]) {
  parmgmc::PetscHelper::init(argc, argv);

  PetscCall(PCRegister("gibbs", parmgmc::PCCreate_Gibbs));
  PetscCall(PCRegister("cholsampler", parmgmc::PCCreate_CholeskySampler));

  // Assemble precision matrix
  PetscInt size = 9;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-problem_size", &size, nullptr));

  SquaredShiftedLaplaceFD problem(Dim{2}, size, 1);
  Mat mat = problem.getFineOperator()->getMat();
  // Just to make sure we're not using any geometric information below
  PetscCall(MatSetDM(mat, nullptr));

  // Setup sampler using PETSc's PCGAMG algebraic multigrid preconditioner
  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  PetscCall(KSPSetOperators(ksp, mat, mat));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  PetscCall(KSPSetFromOptions(ksp));

  // We always want to run a fixed number of iterations (= samples)
  PetscInt nSamples = 100;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-samples", &nSamples, nullptr));
  PetscCall(KSPSetTolerances(ksp, 0, 0, 0, nSamples));
  PetscCall(KSPSetMinimumIterations(ksp, nSamples));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE)); // No need to compute residuals

  /* All the setup code below can also be configured from the command line as:
       ./pcmg -pc_type gamg
              -mg_levels_ksp_type richardson
              -mg_levels_pc_type gibbs
              -mg_levels_ksp_max_it 2
              -mg_coarse_ksp_type preonly
              -mg_coarse_pc_type cholsampler

     To use Gibbs also on the coarsest level:
       -mg_coarse_ksp_type richardson
       -mg_coarse_pc_type gibbs
       -mg_coarse_ksp_max_it 4

     Additional options:
       -mg_levels_pc_gibbs_symmetric (use symmetric Gibbs sweeps)
       -pc_mg_cycle_type w (use W cycles instead of V cycles)
       -pc_mg_type full (use a full multigrid scheme)
  */
#if 0
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCGAMG));
  PetscCall(PCSetFromOptions(pc));

  PetscCall(KSPSetUp(ksp));

  PetscInt levels;
  PetscCall(PCMGGetLevels(pc, &levels));
  
  KSP smoothksp;
  PC smoothpc;
  for (PetscInt l = 1; l < levels; ++l) {
    PetscCall(PCMGGetSmoother(pc, l, &smoothksp));
    PetscCall(KSPSetType(smoothksp, KSPRICHARDSON));
    PetscCall(KSPSetTolerances(smoothksp, 0, 0, 0, 2));

    PetscCall(KSPGetPC(smoothksp, &smoothpc));
    PetscCall(PCSetType(smoothpc, "gibbs"));

    // Allow user to change options
    PetscCall(KSPSetFromOptions(smoothksp));
    PetscCall(PCSetFromOptions(smoothpc));
  }

  // Handle coarse level separately
  PetscCall(PCMGGetSmoother(pc, 0, &smoothksp));
  PetscCall(KSPSetType(smoothksp, KSPPREONLY));
  PetscCall(KSPGetPC(smoothksp, &smoothpc));
  PetscCall(PCSetType(smoothpc, "cholsampler"));
  PetscCall(KSPSetFromOptions(smoothksp));
  PetscCall(PCSetFromOptions(smoothpc));
#endif

  // We use KSPMonitorSet to access the current sample at each iteration.
  // Here we compute the sample mean as an example.
  Vec mean;
  PetscCall(MatCreateVecs(mat, &mean, nullptr));
  PetscCall(KSPSetConvergenceTest(
      ksp,
      [](KSP ksp, PetscInt it, PetscReal, KSPConvergedReason *reason, void *ctx) {
        PetscFunctionBeginUser;

        auto *m = (Vec *)ctx;

        Vec s;
        PetscCall(KSPGetSolution(ksp, &s));

        PetscCall(VecScale(*m, it / (it + 1.)));
        PetscCall(VecAXPY(*m, 1. / (it + 1.), s));

        PetscScalar xnorm;
        PetscCall(VecNorm(s, NORM_2, &xnorm));

        PetscScalar mnorm;
        PetscCall(VecNorm(*m, NORM_2, &mnorm));

        PetscCall(PetscPrintf(MPI_COMM_WORLD, "%d: %f, %f\n", it, xnorm, mnorm));

        *reason = KSP_CONVERGED_ITERATING;

        PetscFunctionReturn(PETSC_SUCCESS);
      },
      &mean, nullptr));

  Vec x, b, f;
  PetscCall(MatCreateVecs(mat, &x, &b));
  std::mt19937 engine{};
  parmgmc::fillVecRand(x, engine);

  PetscCall(VecDuplicate(x, &f));

  // Set target mean and compute "right hand side"
  PetscCall(VecSet(f, 0.));
  PetscCall(MatMult(mat, f, b));

  // Sample
  PetscCall(KSPSolve(ksp, b, x));

  // Clean up
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
}
