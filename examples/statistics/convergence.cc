#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "problems.hh"

#include <iostream>
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

PetscErrorCode updateSampleMeanAndCov(Vec newSample, PetscInt sampleIdx, Vec mean, Mat cov) {
  PetscFunctionBeginUser;

  // Mean update
  Vec tmp;
  PetscCall(VecDuplicate(newSample, &tmp));
  PetscCall(VecCopy(newSample, tmp));

  PetscCall(VecAXPY(tmp, -1., mean));
  PetscCall(VecScale(tmp, 1. / (1 + sampleIdx)));
  PetscCall(VecAXPY(mean, 1., tmp));

  // Cov update
  PetscCall(MatScale(cov, (1. * sampleIdx) / (1 + sampleIdx)));

  Mat outerProdMat;
  PetscCall(MatDuplicate(cov, MAT_DO_NOT_COPY_VALUES, &outerProdMat));

  PetscCall(VecCopy(newSample, tmp));
  PetscCall(VecAXPY(tmp, -1., mean));
  PetscCall(outerProd(tmp, tmp, outerProdMat));
  PetscCall(MatAXPY(cov, (1. * sampleIdx) / ((1 + sampleIdx) * (1 + sampleIdx)), outerProdMat,
                    SAME_NONZERO_PATTERN));

  PetscCall(VecDestroy(&tmp));
  PetscCall(MatDestroy(&outerProdMat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscInt size = 5;

  ShiftedLaplaceFD problem(Dim{2}, size, 1);

  auto mat = problem.getOperator().getMat();

  Mat exactCov;
  PetscCall(denseInverse(mat, &exactCov));

  PetscReal exactCovNorm;
  PetscCall(MatNorm(exactCov, NORM_FROBENIUS, &exactCovNorm));

  Vec sample, rhs;
  PetscCall(MatCreateVecs(mat, &sample, &rhs));

  // PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_DENSE));
  // PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatZeroRowsColumns(problem.getOperator().getMat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., sample, rhs));

  std::mt19937 engine{std::random_device{}()};
  // MulticolorGibbsSampler sampler(problem.getOperator(), &engine);
  MGMCParameters params;
  MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), engine);

  PetscInt nSamples = 100;

  Mat estCov;
  PetscCall(MatDuplicate(exactCov, MAT_DO_NOT_COPY_VALUES, &estCov));
  Vec estMean;

  PetscCall(VecDuplicate(sample, &estMean));

  Vec sample0;
  PetscCall(VecDuplicate(sample, &sample0));

  for (PetscInt i = 0; i < nSamples; ++i) {
    PetscCall(sampler.sample(sample, rhs));

    if (i == 0)
      PetscCall(VecCopy(sample, sample0));
    if (i == 1) {
      Mat tmp;
      PetscCall(MatDuplicate(estCov, MAT_DO_NOT_COPY_VALUES, &tmp));
      PetscCall(outerProd(sample0, sample0, estCov));
      PetscCall(outerProd(sample, sample, tmp));
      PetscCall(MatAXPY(estCov, 1., tmp, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&tmp));
    }

    PetscCall(updateSampleMeanAndCov(sample, i, estMean, estCov));

    PetscReal meanNorm;
    PetscCall(VecNorm(estMean, NORM_2, &meanNorm));

    PetscReal covNorm;
    PetscCall(MatNorm(estCov, NORM_FROBENIUS, &covNorm));

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "%.4f, %.4f\n", meanNorm,
                          std::abs((covNorm - exactCovNorm) / exactCovNorm)));
  }

  // PetscCall(MatView(estCov, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&sample0));
  PetscCall(MatDestroy(&estCov));
  PetscCall(VecDestroy(&estMean));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&sample));
  PetscCall(MatDestroy(&exactCov));
}
