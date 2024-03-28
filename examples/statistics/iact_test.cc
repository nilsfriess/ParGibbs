#include "problems.hh"
#include "qoi.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <mpi.h>
#include <pcg_random.hpp>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include <petscviewer.h>
#include <random>

using namespace parmgmc;

[[nodiscard]] std::size_t integratedAutocorrTime(const std::vector<double> &samples,
                                                 std::size_t windowSize = 80) {
  const auto totalSamples = samples.size();
  if (windowSize > totalSamples)
    return totalSamples;

  double mean = 0;
  for (auto s : samples)
    mean += 1. / totalSamples * s;

  const auto rho = [&](std::size_t s) -> double {
    double sum = 0;
    for (std::size_t j = 0; j < totalSamples - s; ++j)
      sum += 1. / (totalSamples - s) * (samples[j] - mean) * (samples[j + s] - mean);
    return sum;
  };

  double sum = 0;
  const auto rhoZero = rho(0);
  for (std::size_t s = 1; s < windowSize; ++s)
    sum += rho(s) / rhoZero;
  const auto tau = static_cast<std::size_t>(std::ceil(1 + 2 * sum));
  return std::max(1UL, tau);
}

struct IACTResult {
  std::vector<std::size_t> iacts;
  std::vector<double> times;
};

template <class Sampler, class Engine>
PetscErrorCode computeIACT(Sampler &sampler, PetscInt nSamples, Vec rhs, const TestQOI &qoi,
                           Engine &engine, PetscInt nRuns, IACTResult &res) {
  PetscFunctionBeginUser;

  res.iacts.clear();
  res.times.clear();

  Vec sample;
  PetscCall(VecDuplicate(rhs, &sample));

  for (PetscInt run = 0; run < nRuns; ++run) {
    PetscCall(fillVecRand(sample, engine));

    const PetscInt nBurnin = 1000;
    for (PetscInt n = 0; n < nBurnin; ++n)
      PetscCall(sampler.sample(sample, rhs));

    std::vector<double> qoiSamples(nSamples);

    for (PetscInt n = 0; n < nSamples; ++n) {
      PetscCall(sampler.sample(sample, rhs));

      double q;
      PetscCall(qoi(sample, &q));

      qoiSamples[n] = q;
    }

    auto iact = integratedAutocorrTime(qoiSamples);
    res.iacts.push_back(iact);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResults(const std::string &name, const IACTResult &res) {
  double iactMean = 0;
  for (auto v : res.iacts)
    iactMean += 1. / res.iacts.size() * v;

  double iactStd = 0;
  for (auto v : res.iacts)
    iactStd += 1. / (res.iacts.size() - 1) * (v - iactMean) * (v - iactMean);
  iactStd = std::sqrt(iactStd);

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "%s IACT: %.2f ± %.2f\n", name.c_str(), iactMean, iactStd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscFunctionBeginUser;

  PetscInt size = 9;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-size", &size, nullptr));
  PetscInt nSamples = 10000;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-samples", &nSamples, nullptr));
  PetscInt nRuns = 5;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-runs", &nRuns, nullptr));
  PetscInt nRefine = 3;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-refine", &nRefine, nullptr));
  PetscReal kappainv = 1.;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-kappainv", &kappainv, nullptr));

  PetscBool runMGMCCoarseGibbs = PETSC_FALSE, runMGMCCoarseCholesky = PETSC_FALSE,
            runGibbs = PETSC_FALSE, runCholesky = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-gibbs", &runGibbs, nullptr));
  PetscCall(
      PetscOptionsGetBool(nullptr, nullptr, "-mgmc_coarse_gibbs", &runMGMCCoarseGibbs, nullptr));
  PetscCall(
      PetscOptionsGetBool(nullptr, nullptr, "-mgmc_coarse_chol", &runMGMCCoarseCholesky, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-cholesky", &runCholesky, nullptr));

  if (!(runGibbs || runMGMCCoarseGibbs || runMGMCCoarseCholesky || runCholesky)) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "No sampler selected, not running any tests.\n"
                          "Pass at least one of\n"
                          "      -gibbs       -mgmc_coarse_gibbs       -mgmc_coarse_chol\n"
                          "to run the test with the respective sampler.\n"));
    return 0;
  }

  PetscMPIInt mpisize, mpirank;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &mpisize));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpirank));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "####                 Running IACT test suite                ######\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));

  ShiftedLaplaceFD problem(size, nRefine, kappainv);

  DMDALocalInfo fineInfo, coarseInfo;
  PetscCall(DMDAGetLocalInfo(problem.getFineDM(), &fineInfo));
  PetscCall(DMDAGetLocalInfo(problem.getCoarseDM(), &coarseInfo));

  // Setup random number generator
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

  Vec direction;
  PetscCall(MatCreateVecs(problem.getOperator()->getMat(), &direction, nullptr));
  PetscCall(fillVecRand(direction, engine));
  PetscCall(VecNormalize(direction, nullptr));
  TestQOI qoi(direction);

  Vec rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->getMat(), &rhs, nullptr));
  PetscCall(fillVecRand(rhs, engine));
  PetscCall(VecScale(rhs, 100));

  Vec boundaryCond;
  PetscCall(VecDuplicate(rhs, &boundaryCond));
  PetscCall(MatZeroRowsColumns(problem.getOperator()->getMat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., boundaryCond, rhs));
  PetscCall(VecDestroy(&boundaryCond));

  IACTResult res;

  if (runGibbs) {
    MulticolorGibbsSampler sampler(problem.getOperator(), &engine);
    sampler.setFixedRhs(rhs);

    PetscCall(computeIACT(sampler, nSamples, rhs, qoi, engine, nRuns, res));
    PetscCall(printResults("Gibbs", res));
  }

  if (runMGMCCoarseCholesky) {
    MGMCParameters params;
    params.coarseSamplerType = MGMCCoarseSamplerType::Cholesky;
    params.cycleType = MGMCCycleType::V;
    params.smoothingType = MGMCSmoothingType::ForwardBackward;
    params.nSmooth = 2;

    MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), &engine, params);

    PetscCall(computeIACT(sampler, nSamples, rhs, qoi, engine, nRuns, res));
    PetscCall(printResults("MGMC (Cholesky coarse sampler)", res));
  }

  if (runMGMCCoarseGibbs) {
    MGMCParameters params;
    params.coarseSamplerType = MGMCCoarseSamplerType::Standard;
    params.cycleType = MGMCCycleType::V;
    params.smoothingType = MGMCSmoothingType::ForwardBackward;
    params.nSmooth = 2;

    MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), &engine, params);

    PetscCall(computeIACT(sampler, nSamples, rhs, qoi, engine, nRuns, res));
    PetscCall(printResults("MGMC (Gibbs coarse sampler)", res));
  }

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&direction));

  return 0;
}
