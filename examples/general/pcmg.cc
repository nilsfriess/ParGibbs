#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"

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

  using Engine = std::mt19937_64;
  Engine engine{std::random_device{}()};

  PetscCall(PCRegister("gibbs", parmgmc::PCCreate_Gibbs<Engine>));

  // Assemble precision matrix
  PetscInt size = 9;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-problem_size", &size, nullptr));

  ShiftedLaplaceFD problem(Dim{2}, size, 1, 100);
  Mat mat = problem.getFineOperator()->getMat();

  // Setup sampler by abusing PETSc's PCMG multigrid preconditioner
  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  PetscCall(KSPSetOperators(ksp, mat, mat));
  PetscCall(KSPSetDM(ksp, problem.getFineDM()));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  PetscCall(KSPSetFromOptions(ksp));

  // We always want to run a fixed number of iterations (= samples)
  PetscInt nSamples = 100;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-samples", &nSamples, nullptr));
  PetscCall(KSPSetTolerances(ksp, 0, 0, 0, nSamples));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE)); // No need to compute residuals

  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCSetFromOptions(pc));

  PetscInt levels;
  PetscCall(PCMGGetLevels(pc, &levels));

  KSP smoothksp;
  PC smoothpc;
  for (PetscInt l = 0; l < levels; ++l) {
    PetscCall(PCMGGetSmoother(pc, l, &smoothksp));
    PetscCall(KSPSetType(smoothksp, KSPRICHARDSON));
    PetscCall(KSPSetInitialGuessNonzero(smoothksp, PETSC_TRUE));
    PetscCall(KSPSetTolerances(smoothksp, 0, 0, 0, 1));

    PetscCall(KSPGetPC(smoothksp, &smoothpc));
    PetscCall(PCSetType(smoothpc, "gibbs"));
    PetscCall(PCSetApplicationContext(smoothpc, (void *)&engine));
  }

  PetscCall(KSPSetUp(ksp));

  // We use KSPMonitorSet to access the current sample at each iteration
  Vec mean;
  PetscCall(MatCreateVecs(mat, &mean, nullptr));
  PetscCall(KSPMonitorSet(
      ksp,
      [](KSP ksp, PetscInt it, PetscReal, void *ctx) {
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

        PetscFunctionReturn(PETSC_SUCCESS);
      },
      &mean, nullptr));

  Vec x, b, f;
  PetscCall(MatCreateVecs(mat, &x, &b));
  PetscCall(VecDuplicate(x, &f));

  // Set target mean and compute "right hand side"
  PetscCall(VecSet(f, 1.));
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
