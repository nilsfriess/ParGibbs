#include "problems.hh"
#include "qoi.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <limits>
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

[[nodiscard]] std::size_t integratedAutocorrTime(const std::vector<std::vector<double>> &samples,
                                                 std::size_t windowSize = 999) {
  if (windowSize >= samples[0].size())
    return std::numeric_limits<std::size_t>::max();

  std::vector<double> perRunMeans(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    perRunMeans[i] = std::accumulate(samples[i].begin(), samples[i].end(), 0);
    perRunMeans[i] *= 1. / samples[i].size();
  }

  std::size_t totalNumSamples = 0;
  for (const auto &runSamples : samples)
    totalNumSamples += runSamples.size();

  double fullMean = 0;
  for (std::size_t i = 0; i < samples.size(); ++i)
    fullMean += samples[i].size() * perRunMeans[i];
  fullMean /= totalNumSamples;

  const auto rho = [&](std::size_t t) {
    double res = 0;
    for (const auto &runSamples : samples)
      for (std::size_t i = 0; i < runSamples.size() - t; ++i)
        res += (runSamples[i] - fullMean) * (runSamples[i + t] - fullMean);

    return res / (totalNumSamples - samples.size() * t);
  };

  double sum = rho(0);
  for (std::size_t s = 0; s < windowSize; ++s)
    sum += 2 * rho(s);
  sum /= (2 * rho(0));
  return std::max(1UL, (size_t)std::round(sum));
}

struct IACTResult {
  std::vector<std::size_t> iacts;
  std::vector<double> times;
};

template <class Sampler, class Engine>
PetscErrorCode computeIACT(Sampler &sampler, PetscInt nSamples, Vec rhs, const TestQOI &qoi,
                           Engine &engine, PetscInt nRuns, IACTResult &res) {
  (void)engine;

  PetscFunctionBeginUser;

  res.iacts.clear();
  res.times.clear();

  Vec sample;
  PetscCall(VecDuplicate(rhs, &sample));

  std::vector<std::vector<double>> qoiSamples(nRuns, std::vector<double>(nSamples));

  for (PetscInt run = 0; run < nRuns; ++run) {
    PetscCall(VecZeroEntries(sample));

    const PetscInt nBurnin = 10000;
    for (PetscInt n = 0; n < nBurnin; ++n)
      PetscCall(sampler.sample(sample, rhs));

    for (PetscInt n = 0; n < nSamples; ++n) {
      PetscCall(sampler.sample(sample, rhs));

      double q;
      PetscCall(qoi(sample, &q));
      qoiSamples[run][n] = q;

      std::cout << q << "\n";
    }
    std::cout << "\n\n";
  }

  auto iact = integratedAutocorrTime(qoiSamples);
  res.iacts.push_back(iact);

  PetscCall(VecDestroy(&sample));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResults(const std::string &name, const IACTResult &res) {
  // double iactMean = 0;
  // for (auto v : res.iacts)
  //   iactMean += 1. / res.iacts.size() * v;

  // double iactStd = 0;
  // for (auto v : res.iacts)
  //   iactStd += 1. / (res.iacts.size() - 1) * (v - iactMean) * (v - iactMean);
  // iactStd = std::sqrt(iactStd);

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "%s IACT: %zu\n", name.c_str(), res.iacts[0]));
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
  PetscInt dim = 2;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-dim", &dim, nullptr));
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

  PetscBool problemGMRF = PETSC_FALSE, problemDiagPrec = PETSC_FALSE,
            problemShiftedLaplace = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-problem_gmrf", &problemGMRF, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-problem_diag", &problemDiagPrec, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-problem_shifted_laplace",
                                &problemShiftedLaplace, nullptr));

  if (!(problemGMRF || problemDiagPrec || problemShiftedLaplace)) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "No problem selected, not running any tests.\n"
                          "Pass at least one of\n"
                          "   -problem_gmrf   -problem_diag   -problem_shifted_laplace\n"
                          "to run the test with the respective precision matrix.\n"));
    return 0;
  }

  std::string samplerName;
  if (runGibbs)
    samplerName = "Gibbs";
  else if (runMGMCCoarseGibbs)
    samplerName = "MGMC (coarse Gibbs)";
  else if (runMGMCCoarseCholesky)
    samplerName = "MGMC (coarse Cholesky)";

  PetscMPIInt mpisize, mpirank;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &mpisize));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpirank));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "####                 Running IACT test suite                ######\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));

  std::unique_ptr<Problem> problem;
  if (problemGMRF)
    problem = std::make_unique<SimpleGMRF>(Dim{dim}, size, nRefine);
  else if (problemDiagPrec)
    problem = std::make_unique<DiagonalPrecisionMatrix>(Dim{dim}, size, nRefine);
  else
    problem = std::make_unique<ShiftedLaplaceFD>(Dim{dim}, size, nRefine, kappainv);

  DMDALocalInfo fineInfo, coarseInfo;
  PetscCall(DMDAGetLocalInfo(problem->getFineDM(), &fineInfo));
  PetscCall(DMDAGetLocalInfo(problem->getCoarseDM(), &coarseInfo));

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

  PetscCall(PetscPrintf(
      MPI_COMM_WORLD,
      "Configuration: \n"
      "\tMPI rank(s):           %d\n"
      "\tSampler:               %s\n"
      "\tProblem:               %s\n"
      "\tProblem size (coarse): %dx%dx%d = %d\n"
      "\tProblem size (fine):   %dx%dx%d = %d\n"
      "\tLevels:                %d\n"
      "\tSamples:               %d\n"
      "\tRuns:                  %d\n"
      "\tRandom Seed:           %d\n",
      mpisize, samplerName.c_str(), problem->getName().c_str(), coarseInfo.mx, coarseInfo.my,
      coarseInfo.mz, (coarseInfo.mx * coarseInfo.my * coarseInfo.mz), fineInfo.mx, fineInfo.my,
      fineInfo.mz, (fineInfo.mx * fineInfo.my * fineInfo.mz), nRefine, nSamples, nRuns, seed));

  Vec direction;
  PetscCall(MatCreateVecs(problem->getOperator().getMat(), &direction, nullptr));
  PetscCall(VecSet(direction, 1.));
  TestQOI qoi(direction);

  Vec tgtMean, rhs, boundaryCond;
  PetscCall(MatCreateVecs(problem->getOperator().getMat(), &tgtMean, nullptr));
  PetscCall(VecDuplicate(tgtMean, &rhs));
  PetscCall(VecDuplicate(rhs, &boundaryCond));

  PetscCall(fillVecRand(tgtMean, engine));
  PetscCall(MatMult(problem->getOperator().getMat(), tgtMean, rhs));

  PetscCall(VecDestroy(&boundaryCond));
  PetscCall(VecDestroy(&tgtMean));

  IACTResult res;
  if (runGibbs) {
    MulticolorGibbsSampler sampler(problem->getOperator(), engine, 1.6);

    PetscCall(computeIACT(sampler, nSamples, rhs, qoi, engine, nRuns, res));
    PetscCall(printResults("Gibbs", res));
  } else if (runMGMCCoarseCholesky) {
    MGMCParameters params;
    params.coarseSamplerType = MGMCCoarseSamplerType::Cholesky;
    params.cycleType = MGMCCycleType::V;
    params.smoothingType = MGMCSmoothingType::ForwardBackward;
    params.nSmooth = 2;

    MultigridSampler sampler(problem->getOperator(), problem->getHierarchy(), engine, params);

    PetscCall(computeIACT(sampler, nSamples, rhs, qoi, engine, nRuns, res));
    PetscCall(printResults("MGMC (Cholesky coarse sampler)", res));
  } else if (runMGMCCoarseGibbs) {
    MGMCParameters params;
    params.coarseSamplerType = MGMCCoarseSamplerType::Standard;
    params.cycleType = MGMCCycleType::V;
    params.smoothingType = MGMCSmoothingType::ForwardBackward;
    params.nSmooth = 2;

    MultigridSampler sampler(problem->getOperator(), problem->getHierarchy(), engine, params);

    PetscCall(computeIACT(sampler, nSamples, rhs, qoi, engine, nRuns, res));
    PetscCall(printResults("MGMC (Gibbs coarse sampler)", res));
  }

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&direction));

  return 0;
}
