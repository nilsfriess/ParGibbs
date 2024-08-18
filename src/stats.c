#include "parmgmc/stats.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
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
      arr[idx]     = aArr[j] * bArr[i];
    }
  }

  PetscCall(VecRestoreArrayRead(a, &aArr));
  PetscCall(VecRestoreArrayRead(b, &bArr));
  PetscCall(MatDenseRestoreArray(res, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SampleMean(PetscInt n, Vec *samples, Vec mean)
{
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(mean));
  for (PetscInt i = 0; i < n; ++i) PetscCall(VecAXPY(mean, 1. / n, samples[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SampleCovariance(PetscInt n, Vec *samples, Mat C)
{
  Vec m, w;
  Mat T;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(C));
  PetscCall(VecDuplicate(samples[0], &m));
  PetscCall(VecDuplicate(samples[0], &w));
  PetscCall(MatDuplicate(C, MAT_DO_NOT_COPY_VALUES, &T));
  PetscCall(SampleMean(n, samples, m));
  for (PetscInt i = 0; i < n; ++i) {
    PetscCall(VecCopy(samples[i], w));
    PetscCall(VecAXPY(w, -1., m));
    PetscCall(OuterProd(w, w, T));
    PetscCall(MatAXPY(C, 1. / (n - 1), T, SAME_NONZERO_PATTERN));
  }
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&m));
  PetscCall(MatDestroy(&T));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Estimate the relative error between the sample covariance matrix and the true covariance matrix using samples from multiple Markov chains.

    The samples are assumed to be given as follows: First all samples with index 0, then all samples with index 1 etc., i.e,
    { sample 0 from chain 0, sample 0 from chain 1, ..., sample 1 from chain 0, sample 1 from chain 1, ... }

    The returned error array has length `samples_per_chain`. Therefore, a sufficiently high number of chains is necessary for this
    estimate to be accurate.
*/
PetscErrorCode EstimateCovarianceMatErrors(Mat A, PetscInt chains, PetscInt samples_per_chain, Vec *samples, PetscScalar *errs)
{
  Mat         C;
  Mat         Q;
  PetscReal   Qnorm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP_SYS, "Estimating covariance matrix only supported in sequential execution");
  PetscCall(DenseInverse(A, &Q));
  PetscCall(MatNorm(Q, NORM_FROBENIUS, &Qnorm));

  PetscCall(MatDuplicate(Q, MAT_DO_NOT_COPY_VALUES, &C));
  for (PetscInt i = 0; i < samples_per_chain; ++i) {
    PetscCall(SampleCovariance(chains, samples + i * chains, C));
    PetscCall(MatAXPY(C, -1, Q, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(C, NORM_FROBENIUS, &errs[i]));
    errs[i] /= Qnorm;
  }
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}
