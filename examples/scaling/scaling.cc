#include "problems.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/samplers/hogwild.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <random>

#include <mpi.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscerror.h>
#include <petscsystypes.h>

#include <pcg_random.hpp>

using namespace parmgmc;

struct TimingResult {
  TimingResult operator+=(const TimingResult &other) {
    setupTime += other.setupTime;
    sampleTime += other.sampleTime;
    return *this;
  }

  TimingResult operator/=(double d) {
    setupTime /= d;
    sampleTime /= d;
    return *this;
  }

  double setupTime = 0;
  double sampleTime = 0;
};

template <typename Engine>
PetscErrorCode testGibbsSampler(const ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                                PetscScalar omega, GibbsSweepType sweepType, bool fixRhs,
                                TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(MatZeroRowsColumns(problem.getOperator()->getMat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., sample, rhs));

  PetscCall(fillVecRand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MulticolorGibbsSampler sampler(problem.getOperator(), &engine, omega, sweepType);

  if (fixRhs)
    sampler.setFixedRhs(rhs);

  timingResult.setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  timingResult.sampleTime = timer.elapsed();
  // Sampling done

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Engine>
PetscErrorCode testHogwildGibbsSampler(const ShiftedLaplaceFD &problem, PetscInt nSamples,
                                       Engine &engine, bool fixRhs, TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(MatZeroRowsColumns(problem.getOperator()->getMat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., sample, rhs));

  PetscCall(fillVecRand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  HogwildGibbsSampler sampler(problem.getOperator(), &engine);

  if (fixRhs)
    sampler.fixRhs(rhs);

  timingResult.setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  timingResult.sampleTime = timer.elapsed();
  // Sampling done

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Engine>
PetscErrorCode testMGMCSampler(const ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                               const MGMCParameters &params, TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(MatZeroRowsColumns(problem.getOperator()->getMat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., sample, rhs));

  PetscCall(fillVecRand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), &engine, params);
  timingResult.setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  timingResult.sampleTime = timer.elapsed();
  // Sampling done

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResult(const std::string &name, TimingResult timing) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\n+++-------------------------------------------------"
                                        "-----------+++\n\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Name: %s\n", name.c_str()));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Timing [s]:\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Setup time:    %.4f\n", timing.setupTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Sampling time: %.4f\n", timing.sampleTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   -----------------------\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Total:         %.4f\n",
                        timing.setupTime + timing.sampleTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\n+++-------------------------------------------------"
                                        "-----------+++\n"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscFunctionBeginUser;

  PetscInt size = 9;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-size", &size, nullptr));
  PetscInt nSamples = 1000;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-samples", &nSamples, nullptr));
  PetscInt nRuns = 5;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-runs", &nRuns, nullptr));
  PetscInt nRefine = 3;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-refine", &nRefine, nullptr));

  // PetscBool weak = PETSC_TRUE;
  // PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-weak", &weak, nullptr));

  PetscBool runGibbs = PETSC_FALSE, runMGMC = PETSC_FALSE, runCholesky = PETSC_FALSE,
            runHogwild = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-gibbs", &runGibbs, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-mgmc", &runMGMC, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-cholesky", &runCholesky, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-hogwild", &runHogwild, nullptr));

  PetscMPIInt mpisize, mpirank;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &mpisize));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpirank));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "####               Running scaling test suite               ######\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));

  if (!(runGibbs || runMGMC || runCholesky || runHogwild)) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "No sampler selected, not running any tests.\n"
                                          "Pass at least one of\n"
                                          "     -gibbs     -mgmc     -cholesky     -hogwild\n"
                                          "to run the test with the respective sampler.\n"));
    return 0;
  }

  ShiftedLaplaceFD problem(size, nRefine, 1., true);
  DMDALocalInfo fineInfo, coarseInfo;
  PetscCall(DMDAGetLocalInfo(problem.getFineDM(), &fineInfo));
  PetscCall(DMDAGetLocalInfo(problem.getCoarseDM(), &coarseInfo));

  pcg32 engine;

  int seed;
  if (mpirank == 0) {
    seed = std::random_device{}();
    PetscOptionsGetInt(nullptr, nullptr, "-seed", &seed, nullptr);
  }

  // Send seed to all other processes
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  engine.seed(seed);
  engine.set_stream(mpirank);

  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "Configuration: \n"
                        "\tMPI rank(s):           %d\n"
                        "\tProblem size (coarse): %dx%d = %d\n"
                        "\tProblem size (fine):   %dx%d = %d\n"
                        "\tLevels:                %d\n"
                        "\tSamples:               %d\n"
                        "\tRuns:                  %d\n"
                        "\tRandom Seed:           %d\n",
                        mpisize, coarseInfo.mx, coarseInfo.mx, (coarseInfo.mx * coarseInfo.mx),
                        fineInfo.mx, fineInfo.mx, (fineInfo.mx * fineInfo.mx), nRefine, nSamples,
                        nRuns, seed));

  if (runGibbs) {
    TimingResult avg;

    for (int i = 0; i < nRuns; ++i) {
      TimingResult timing;
      PetscCall(
          testGibbsSampler(problem, nSamples, engine, 1., GibbsSweepType::Forward, true, timing));

      avg += timing;
    }

    avg /= nRuns;

    PetscCall(printResult("Gibbs sampler, forward sweep, fixed rhs", avg));
  }

  if (runMGMC) {
    TimingResult avg;

    MGMCParameters params = MGMCParameters::defaultParams();

    for (int i = 0; i < nRuns; ++i) {
      TimingResult timing;
      PetscCall(testMGMCSampler(problem, nSamples, engine, params, timing));

      avg += timing;
    }

    avg /= nRuns;

    PetscCall(printResult("MGMC sampler", avg));
  }

  if (runHogwild) {
    TimingResult avg;

    for (int i = 0; i < nRuns; ++i) {
      TimingResult timing;
      PetscCall(testHogwildGibbsSampler(problem, nSamples, engine, true, timing));

      avg += timing;
    }

    avg /= nRuns;

    PetscCall(printResult("Hogwild Gibbs sampler, forward sweep, fixed rhs", avg));
  }

  // {
  //   TimingResult timing;
  //   PetscCall(testGibbsSampler(problem,
  //                              n_samples,
  //                              engine,
  //                              1.,
  //                              GibbsSweepType::Forward,
  //                              false,
  //                              timing));

  //   PetscCall(
  //       printResult("Gibbs sampler, forward sweep, nonfixed rhs", timing));
  // }

  // {
  //   TimingResult timing;
  //   PetscCall(testGibbsSampler(problem,
  //                              n_samples,
  //                              engine,
  //                              1.,
  //                              GibbsSweepType::Symmetric,
  //                              true,
  //                              timing));

  //   PetscCall(
  //       printResult("Gibbs sampler, symmetric sweep, nonfixed rhs", timing));
  // }
}
