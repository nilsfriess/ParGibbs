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
  TimingResult() = default;
  TimingResult(std::size_t nRuns) {
    setupTimes.reserve(nRuns);
    samplingTimes.reserve(nRuns);
  }

  std::pair<double, double> getSetupTime() const { return meanAndStdDev(setupTimes); }

  std::pair<double, double> getSamplingTime() const { return meanAndStdDev(samplingTimes); }

  std::pair<double, double> getTotalTime() const {
    std::vector<double> totalTimes(setupTimes.size());

    for (std::size_t i = 0; i < totalTimes.size(); ++i)
      totalTimes[i] = setupTimes[i] + samplingTimes[i];

    return meanAndStdDev(totalTimes);
  }

  void add(double setupTime, double samplingTime) {
    setupTimes.push_back(setupTime);
    samplingTimes.push_back(samplingTime);
  }

private:
  std::pair<double, double> meanAndStdDev(const std::vector<double> &data) const {
    double sum = 0.0;
    double sumSquaredDiff = 0.0;

    for (double val : data) {
      sum += val;
    }

    double mean = sum / data.size();

    for (double val : data) {
      double diff = val - mean;
      sumSquaredDiff += diff * diff;
    }

    double variance = sumSquaredDiff / data.size();
    double stdDev = std::sqrt(variance);

    return std::make_pair(mean, stdDev);
  }

  std::vector<double> setupTimes;
  std::vector<double> samplingTimes;
};

template <typename Engine>
PetscErrorCode testGibbsSampler(ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                                PetscScalar omega, GibbsSweepType sweepType,
                                TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getFineOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(fillVecRand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MulticolorGibbsSampler sampler(*problem.getFineOperator(), engine, omega, sweepType);

  auto setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  auto sampleTime = timer.elapsed();
  // Sampling done

  timingResult.add(setupTime, sampleTime);

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Engine>
PetscErrorCode testHogwildGibbsSampler(ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                                       TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getFineOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(fillVecRand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  HogwildGibbsSampler sampler(*problem.getFineOperator(), engine);

  auto setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  auto sampleTime = timer.elapsed();
  // Sampling done

  timingResult.add(setupTime, sampleTime);

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Engine>
PetscErrorCode testMGMCSampler(ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                               const MGMCParameters &params, TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getFineOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(fillVecRand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MultigridSampler sampler(problem.getFineOperator(), problem.getHierarchy(), engine, params);
  auto setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  auto sampleTime = timer.elapsed();
  // Sampling done

  timingResult.add(setupTime, sampleTime);

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResult(const std::string &name, const TimingResult &timing) {
  PetscFunctionBeginUser;

  auto [setupAvg, setupStd] = timing.getSetupTime();
  auto [sampleAvg, sampleStd] = timing.getSamplingTime();
  auto [totalAvg, totalStd] = timing.getTotalTime();

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\n+++-------------------------------------------------"
                                        "-----------+++\n\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Name: %s\n", name.c_str()));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Timing [s]:\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Setup time:    %.4f ± %.4f\n", setupAvg, setupStd));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Sampling time: %.4f ± %.4f\n", sampleAvg, sampleStd));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   ------------------------------\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Total:         %.4f ± %.4f\n", totalAvg, totalStd));
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

  PetscBool sizeIsFine = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-size-is-fine", &sizeIsFine, nullptr));

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

  ShiftedLaplaceFD problem(Dim{2}, size, nRefine, 1., true, sizeIsFine);
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
    TimingResult res;

    for (int i = 0; i < nRuns; ++i)
      PetscCall(testGibbsSampler(problem, nSamples, engine, 1., GibbsSweepType::Forward, res));

    PetscCall(printResult("Gibbs sampler, forward sweep, fixed rhs", res));
  }

  if (runMGMC) {
    TimingResult res;

    MGMCParameters params = MGMCParameters::defaultParams();
    params.coarseSamplerType = MGMCCoarseSamplerType::Standard;

    for (int i = 0; i < nRuns; ++i)
      PetscCall(testMGMCSampler(problem, nSamples, engine, params, res));

    PetscCall(printResult("MGMC sampler", res));
  }

  if (runHogwild) {
    TimingResult res;

    for (int i = 0; i < nRuns; ++i)
      PetscCall(testHogwildGibbsSampler(problem, nSamples, engine, res));

    PetscCall(printResult("Hogwild Gibbs sampler, forward sweep, fixed rhs", res));
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
