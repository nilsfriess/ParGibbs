#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/samplers/cholesky.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "problems.hh"

#include <cassert>
#include <pcg_random.hpp>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <random>

using namespace parmgmc;

PetscErrorCode denseInverse(Mat mat, Mat *inverse) {
  PetscFunctionBeginUser;

  PetscInt size;
  PetscCall(MatGetSize(mat, &size, nullptr));

  PetscCall(
      MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, size, size, nullptr, inverse));

  Mat identity;
  PetscCall(
      MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, size, size, nullptr, &identity));
  PetscCall(MatShift(identity, 1.));

  Mat factor;
  IS rowperm, colperm;
  PetscCall(MatGetFactor(mat, MATSOLVERPETSC, MAT_FACTOR_LU, &factor));
  PetscCall(MatGetOrdering(mat, MATORDERINGNATURAL, &rowperm, &colperm));
  PetscCall(MatLUFactorSymbolic(factor, mat, rowperm, colperm, nullptr));
  PetscCall(MatLUFactorNumeric(factor, mat, nullptr));

  PetscCall(MatMatSolve(factor, identity, *inverse));

  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));
  PetscCall(MatDestroy(&factor));
  PetscCall(MatDestroy(&identity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode outerProd(Vec a, Mat res, bool add = false) {
  PetscFunctionBeginUser;

  PetscScalar *arr;
  const PetscScalar *aArr;
  PetscCall(MatDenseGetArray(res, &arr));

  PetscCall(VecGetArrayRead(a, &aArr));

  PetscInt size;
  PetscCall(VecGetSize(a, &size));

  for (PetscInt j = 0; j < size; ++j) {
    for (PetscInt i = 0; i < size; ++i) {
      const auto idx = j * size + i;

      if (add)
        arr[idx] += aArr[i] * aArr[j];
      else
        arr[idx] = aArr[i] * aArr[j];
    }
  }

  PetscCall(VecRestoreArrayRead(a, &aArr));
  PetscCall(MatDenseRestoreArray(res, &arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeMean(const std::vector<Vec> &samples, Vec m) {
  PetscFunctionBeginUser;

  PetscCall(VecZeroEntries(m));

  const auto nSamples = samples.size();
  assert(nSamples > 0);

  for (auto &sample : samples)
    PetscCall(VecAXPY(m, 1., sample));
  PetscCall(VecScale(m, 1. / nSamples));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeCov(const std::vector<Vec> &samples, Vec mean, Mat c) {
  PetscFunctionBeginUser;

  PetscCall(MatZeroEntries(c));

  const auto nSamples = samples.size();
  assert(nSamples > 1);

  Vec tmp;
  PetscCall(VecDuplicate(mean, &tmp));

  Mat tmpMat;
  PetscCall(MatDuplicate(c, MAT_DO_NOT_COPY_VALUES, &tmpMat));

  for (auto &sample : samples) {
    PetscCall(VecWAXPY(tmp, -1., mean, sample));
    PetscCall(outerProd(tmp, tmpMat));
    PetscCall(MatAXPY(c, 1. / (nSamples - 1), tmpMat, SAME_NONZERO_PATTERN));
  }

  PetscCall(MatDestroy(&tmpMat));
  PetscCall(VecDestroy(&tmp));

  // PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_DENSE));
  // PetscCall(MatView(c, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Problem, typename SamplerConstructor>
PetscErrorCode run(const Problem &problem, Mat exactCov, SamplerConstructor &&sc) {
  PetscInt nSamplers = 10;
  PetscInt nRuns = 2;

  PetscFunctionBeginUser;

  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "--samplers", &nSamplers, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "--runs", &nRuns, nullptr));

  PetscReal exactCovNorm;
  PetscCall(MatNorm(exactCov, NORM_FROBENIUS, &exactCovNorm));

  std::vector<decltype(sc())> samplers;
  samplers.reserve(nSamplers);
  for (PetscInt i = 0; i < nSamplers; ++i)
    samplers.emplace_back(sc());

  std::vector<Vec> samples{static_cast<std::size_t>(nSamplers), nullptr};
  Vec rhs;
  for (auto &sample : samples)
    PetscCall(MatCreateVecs(problem.getFineOperator()->getMat(), &sample, nullptr));
  PetscCall(VecDuplicate(samples[0], &rhs));

  Vec mean;
  PetscCall(VecDuplicate(rhs, &mean));
  Mat cov;
  PetscCall(MatDuplicate(exactCov, MAT_DO_NOT_COPY_VALUES, &cov));

  for (PetscInt i = 0; i < nRuns; ++i) {
    for (PetscInt s = 0; s < nSamplers; ++s) {
      PetscCall(samplers[s].sample(samples[s], rhs));
    }

    PetscCall(computeMean(samples, mean));
    PetscCall(computeCov(samples, mean, cov));

    PetscScalar meanNorm, covNorm;
    PetscCall(VecNorm(mean, NORM_2, &meanNorm));

    PetscCall(MatAXPY(cov, -1., exactCov, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(cov, NORM_FROBENIUS, &covNorm));

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "%.4f, %.4f\n", meanNorm, covNorm / exactCovNorm));
  }

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&mean));
  PetscCall(MatDestroy(&cov));
  for (auto &sample : samples)
    PetscCall(VecDestroy(&sample));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscInt size = 3;
  PetscInt levels = 3;
  Dim dim{2};

  enum class SamplerType { Gibbs, MGMC, Cholesky };

  using Engine = pcg32;
  Engine engine{std::random_device{}()};

  SamplerType type = SamplerType::MGMC;

  SimpleGMRF problem(dim, size, levels);
  // ShiftedLaplaceFD problem(dim, size, levels, 100);
  // DiagonalPrecisionMatrix problem(dim, size, levels);
  auto mat = problem.getFineOperator()->getMat();

  Mat exactCov;
  PetscCall(denseInverse(mat, &exactCov));

  // PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_DENSE));
  // PetscCall(MatView(exactCov, PETSC_VIEWER_STDOUT_WORLD));

  switch (type) {
  case SamplerType::Gibbs:
    PetscCall(run(problem, exactCov, [&]() {
      return MulticolorGibbsSampler{*problem.getFineOperator(), engine, 1.9852};
    }));
    break;

  case SamplerType::MGMC: {
    MGMCParameters params;
    params.nSmooth = 2;
    params.cycleType = MGMCCycleType::V;
    params.smoothingType = MGMCSmoothingType::ForwardBackward;
    params.coarseSamplerType = MGMCCoarseSamplerType::Standard;

    PetscCall(run(problem, exactCov, [&]() {
      return MultigridSampler{problem.getFineOperator(), problem.getHierarchy(), engine, params};
    }));
    break;
  }

  case SamplerType::Cholesky:
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
    PetscCall(run(problem, exactCov, [&]() {
      return CholeskySampler{*problem.getFineOperator(), engine};
    }));
#else
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Not supported\n"));
#endif
    break;

  default:
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Not implemented\n"));
  }

  PetscCall(MatDestroy(&exactCov));

  // Vec sample, rhs;
  // PetscCall(MatCreateVecs(mat, &sample, &rhs));

  // Vec tgtMean;
  // PetscCall(VecDuplicate(rhs, &tgtMean));
  // PetscCall(VecSet(tgtMean, 2));

  // double tgtMeanNorm;
  // PetscCall(VecNorm(tgtMean, NORM_2, &tgtMeanNorm));

  // PetscCall(MatMult(problem.getOperator().getMat(), tgtMean, rhs));

  // Statistics stat{exactCov, sample};

  // std::mt19937 engine{std::random_device{}()};

  // // MulticolorGibbsSampler sampler(problem.getOperator(), engine);
  // MGMCParameters params;
  // params.nSmooth = 2;
  // params.cycleType = MGMCCycleType::V;
  // params.smoothingType = MGMCSmoothingType::ForwardOnly;
  // params.coarseSamplerType = MGMCCoarseSamplerType::Cholesky;
  // MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), engine, params);
  // // CholeskySampler sampler(problem.getOperator(), engine);

  // Mat err;
  // PetscCall(MatDuplicate(exactCov, MAT_DO_NOT_COPY_VALUES, &err));

  // PetscInt nSamples = 1000;
  // for (PetscInt i = 0; i < nSamples; ++i) {
  //   PetscCall(sampler.sample(sample, rhs));
  //   PetscCall(stat.addSample(sample));

  //   PetscReal meanNorm;
  //   PetscCall(VecNorm(stat.getMean(), NORM_2, &meanNorm));

  //   PetscCall(MatCopy(exactCov, err, SAME_NONZERO_PATTERN));
  //   PetscCall(MatAXPY(err, -1., stat.getCov(), SAME_NONZERO_PATTERN));

  //   PetscReal covNorm;
  //   PetscCall(MatNorm(err, NORM_FROBENIUS, &covNorm));

  //   PetscCall(PetscPrintf(MPI_COMM_WORLD, "%.4f, %.4f\n", std::abs(meanNorm - tgtMeanNorm),
  //                         (covNorm / exactCovNorm)));
  // }

  // PetscCall(VecDestroy(&rhs));
  // PetscCall(VecDestroy(&tgtMean));
  // PetscCall(VecDestroy(&sample));
  PetscCall(MatDestroy(&exactCov));
}
