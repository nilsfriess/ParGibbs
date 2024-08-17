#include "parmgmc/stats.h"
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

static PetscErrorCode DenseInverse(Mat A, Mat *Q)
{
  PetscInt n;
  Mat      F;
  IS       rowperm, colperm;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A), PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, Q));
  PetscCall(MatShift(*Q, 1));

  PetscCall(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU, &F));
  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &rowperm, &colperm));
  PetscCall(MatLUFactorSymbolic(F, A, rowperm, colperm, NULL));
  PetscCall(MatLUFactorNumeric(F, A, NULL));
  PetscCall(MatMatSolve(F, *Q, *Q));
  PetscCall(MatDestroy(&F));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OuterProd(Vec a, Vec b, Mat res)
{
  PetscScalar       *arr;
  const PetscScalar *aArr, *bArr;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArray(res, &arr));
  PetscCall(VecGetArrayRead(a, &aArr));
  PetscCall(VecGetArrayRead(b, &bArr));
  PetscCall(VecGetSize(a, &n));

  for (PetscInt j = 0; j < n; ++j) {
    for (PetscInt i = 0; i < n; ++i) {
      PetscInt idx = j * n + i;
      arr[idx]     = aArr[i] * bArr[j];
    }
  }

  PetscCall(VecRestoreArrayRead(a, &aArr));
  PetscCall(VecRestoreArrayRead(b, &bArr));
  PetscCall(MatDenseRestoreArray(res, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Estimate the relative error between the sample covariance matrix and the true covariance matrix using samples from multiple Markov chains.

    The samples are assumed to be given as follows: First all samples with index 0, then all samples with index 1 etc., i.e,
    { sample 0 from chain 0, sample 0 from chain 1, ..., sample 1 from chain 0, sample 1 from chain 1, ... }

    The returned error array has length `samples_per_chain`. Therefore, a sufficiently high number of chains is necessary for this
    estimate to be accurate.
*/
PetscErrorCode EstimateCovarianceMatErrors(Mat A, PetscInt chains, PetscInt samples_per_chain, const Vec *samples, PetscScalar *errs)
{
  Vec       mean;
  Vec       tmp;
  Mat       outer_prod_tmp;
  PetscInt  n;
  Mat       C;
  Mat       Q;
  PetscReal Qnorm;

  PetscFunctionBeginUser;
  PetscCall(MatNorm(A, NORM_FROBENIUS, &Qnorm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "A Norm: %.5f\n", Qnorm));
  PetscCall(DenseInverse(A, &Q));
  PetscCall(MatNorm(Q, NORM_FROBENIUS, &Qnorm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Q Norm: %.5f\n", Qnorm));
  PetscCall(VecDuplicate(samples[0], &mean));
  PetscCall(VecDuplicate(samples[0], &tmp));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatDuplicate(Q, MAT_DO_NOT_COPY_VALUES, &C));
  PetscCall(MatDuplicate(C, MAT_DO_NOT_COPY_VALUES, &outer_prod_tmp));
  for (PetscInt i = 0; i < samples_per_chain; ++i) {
    // Compute between chain means
    PetscCall(VecZeroEntries(mean));
    PetscCall(MatZeroEntries(C));
    for (PetscInt j = 0; j < chains; ++j) { PetscCall(VecAXPY(mean, 1. / chains, samples[i * chains + j])); }
    for (PetscInt j = 0; j < chains; ++j) {
      PetscCall(VecCopy(samples[i * chains + j], tmp));
      PetscCall(VecAXPY(tmp, -1., mean));
      PetscCall(OuterProd(tmp, tmp, outer_prod_tmp));
      PetscCall(MatAXPY(C, 1. / (chains - 1), outer_prod_tmp, SAME_NONZERO_PATTERN));
    }
    PetscCall(MatAXPY(C, -1, Q, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(C, NORM_FROBENIUS, &errs[i]));
    errs[i] /= Qnorm;
  }
  PetscCall(MatDestroy(&outer_prod_tmp));
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&tmp));
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}
