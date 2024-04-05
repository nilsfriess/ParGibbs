#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/samplers/cholesky.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "problems.hh"

#include <petscmat.h>
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

PetscErrorCode outerProd(Vec a, Vec b, Mat res) {
  PetscFunctionBeginUser;

  PetscScalar *arr;
  const PetscScalar *aArr, *bArr;
  PetscCall(MatDenseGetArray(res, &arr));

  PetscCall(VecGetArrayRead(a, &aArr));
  PetscCall(VecGetArrayRead(b, &bArr));

  PetscInt size;
  PetscCall(VecGetSize(a, &size));

  for (PetscInt j = 0; j < size; ++j) {
    for (PetscInt i = 0; i < size; ++i) {
      const auto idx = j * size + i;

      arr[idx] = aArr[i] * bArr[j];
    }
  }

  PetscCall(VecRestoreArrayRead(a, &aArr));
  PetscCall(VecRestoreArrayRead(b, &bArr));

  PetscCall(MatDenseRestoreArray(res, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

class Statistics {
public:
  Statistics(Mat covReference, Vec meanReference) {
    PetscFunctionBeginUser;

    PetscCallVoid(MatDuplicate(covReference, MAT_DO_NOT_COPY_VALUES, &cov));
    PetscCallVoid(MatDuplicate(covReference, MAT_DO_NOT_COPY_VALUES, &outerProdTmp));

    PetscCallVoid(VecDuplicate(meanReference, &mean));
    PetscCallVoid(VecDuplicate(meanReference, &tmp));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode addSample(Vec newSample) {
    PetscFunctionBeginUser;

    // If sampleIdx == 0, we can't compute the covariance yet
    if (sampleIdx == 0) {
      PetscCall(VecCopy(newSample, mean));
    } else if (sampleIdx == 1) {
      PetscCall(VecCopy(mean, tmp)); // mean == sample 0

      // Compute mean
      PetscCall(VecAXPY(mean, 1., newSample));
      PetscCall(VecScale(mean, 0.5));

      // Compute sample covariance
      PetscCall(VecAXPY(tmp, -1, mean)); // == sample0 - mean1
      PetscCall(outerProd(tmp, tmp, cov));

      PetscCall(VecCopy(newSample, tmp));
      PetscCall(VecAXPY(tmp, -1, mean)); // == sample1 - mean1
      PetscCall(outerProd(tmp, tmp, outerProdTmp));

      PetscCall(MatAXPY(cov, 1., outerProdTmp, SAME_NONZERO_PATTERN));
    } else {
      // Mean update
      PetscCall(VecCopy(newSample, tmp));

      PetscCall(VecAXPY(tmp, -1., mean));
      PetscCall(VecScale(tmp, 1. / (1 + sampleIdx)));
      PetscCall(VecAXPY(mean, 1., tmp));

      // Cov update
      PetscCall(MatScale(cov, (1. * sampleIdx) / (1 + sampleIdx)));

      PetscCall(VecCopy(newSample, tmp));
      PetscCall(VecAXPY(tmp, -1., mean));
      PetscCall(outerProd(tmp, tmp, outerProdTmp));
      PetscCall(MatAXPY(cov, (1. * sampleIdx) / ((1 + sampleIdx) * (1 + sampleIdx)), outerProdTmp,
                        SAME_NONZERO_PATTERN));

      // PetscCall(MatScale(cov, (sampleIdx - 2.) / (sampleIdx - 1.)));
      // PetscCall(outerProd(newSample, newSample, outerProdTmp));
      // PetscCall(MatAXPY(cov, 1. / sampleIdx, outerProdTmp, SAME_NONZERO_PATTERN));
    }

    sampleIdx++;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~Statistics() {
    PetscFunctionBeginUser;

    PetscCallVoid(MatDestroy(&cov));
    PetscCallVoid(MatDestroy(&outerProdTmp));

    PetscCallVoid(VecDestroy(&mean));
    PetscCallVoid(VecDestroy(&tmp));

    PetscFunctionReturnVoid();
  }

  Vec getMean() const { return mean; }
  Mat getCov() const { return cov; }

private:
  PetscInt sampleIdx = 0;

  Mat cov;
  Vec mean;

  Vec tmp;
  Mat outerProdTmp;
};

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscInt size = 5;
  PetscInt levels = 3;
  Dim dim{2};

  ShiftedLaplaceFD problem(dim, size, levels, 10);
  auto mat = problem.getOperator().getMat();

  Mat exactCov;
  PetscCall(denseInverse(mat, &exactCov));

  PetscReal exactCovNorm;
  PetscCall(MatNorm(exactCov, NORM_FROBENIUS, &exactCovNorm));

  Vec sample, rhs;
  PetscCall(MatCreateVecs(mat, &sample, &rhs));

  Vec tgtMean;
  PetscCall(VecDuplicate(rhs, &tgtMean));
  PetscCall(VecSet(tgtMean, 1));

  double tgtMeanNorm;
  PetscCall(VecNorm(tgtMean, NORM_2, &tgtMeanNorm));

  PetscCall(MatMult(problem.getOperator().getMat(), tgtMean, rhs));

  Statistics stat{exactCov, sample};

  std::mt19937 engine{std::random_device{}()};

  // MulticolorGibbsSampler sampler(problem.getOperator(), engine);
  MGMCParameters params;
  params.coarseSamplerType = MGMCCoarseSamplerType::Standard;
  MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), engine, params);
  // CholeskySampler sampler(problem.getOperator(), engine);

  Mat err;
  PetscCall(MatDuplicate(exactCov, MAT_DO_NOT_COPY_VALUES, &err));

  PetscInt nSamples = 1000;
  for (PetscInt i = 0; i < nSamples; ++i) {
    PetscCall(sampler.sample(sample, rhs));
    PetscCall(stat.addSample(sample));

    PetscReal meanNorm;
    PetscCall(VecNorm(stat.getMean(), NORM_2, &meanNorm));

    PetscCall(MatCopy(exactCov, err, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(err, -1., stat.getCov(), SAME_NONZERO_PATTERN));

    PetscReal covNorm;
    PetscCall(MatNorm(err, NORM_FROBENIUS, &covNorm));

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "%.4f, %.4f\n", std::abs(meanNorm - tgtMeanNorm),
                          std::abs(covNorm / exactCovNorm)));
  }

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&tgtMean));
  PetscCall(VecDestroy(&sample));
  PetscCall(MatDestroy(&exactCov));
}
