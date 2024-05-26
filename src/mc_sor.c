#include "parmgmc/mc_sor.h"

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

PetscErrorCode MatMultiColorSOR(Mat mat, const PetscInt *diagptrs, Vec idiag, Vec b, PetscReal omega, PetscInt ncolors, const IS *ind, const VecScatter *sct, const Vec *ghostvec, Vec y)
{
  Mat              ad, ao; // Local and off-processor parts of mat
  PetscInt         nind, gcnt;
  const PetscInt  *rowptr, *colptr, *bRowptr, *bColptr, *rowind;
  const PetscReal *idiagarr, *barr, *ghostarr;
  PetscReal       *matvals, *bMatvals, *yarr;

  PetscFunctionBeginUser;
  if (sct) {
    PetscCall(MatMPIAIJGetSeqAIJ(mat, &ad, &ao, NULL));
    PetscCall(MatSeqAIJGetCSRAndMemType(ao, &bRowptr, &bColptr, &bMatvals, NULL));
  } else {
    ad = mat;
  }
  PetscCall(MatSeqAIJGetCSRAndMemType(ad, &rowptr, &colptr, &matvals, NULL));
  PetscCall(VecGetArrayRead(idiag, &idiagarr));
  PetscCall(VecGetArrayRead(b, &barr));

  for (PetscInt color = 0; color < ncolors; ++color) {
    if (sct) {
      PetscCall(VecScatterBegin(sct[color], y, ghostvec[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sct[color], y, ghostvec[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(ghostvec[color], &ghostarr));
    }

    PetscCall(ISGetLocalSize(ind[color], &nind));
    PetscCall(ISGetIndices(ind[color], &rowind));
    PetscCall(VecGetArray(y, &yarr));

    gcnt = 0;
    for (PetscInt i = 0; i < nind; ++i) {
      PetscReal sum = 0;

      for (PetscInt k = rowptr[rowind[i]]; k < diagptrs[rowind[i]]; ++k) sum -= matvals[k] * yarr[colptr[k]];
      for (PetscInt k = diagptrs[rowind[i]] + 1; k < rowptr[rowind[i] + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];
      if (sct) {
        for (PetscInt k = bRowptr[rowind[i]]; k < bRowptr[rowind[i] + 1]; ++k) { //
          sum -= bMatvals[k] * ghostarr[gcnt++];
        }
      }

      yarr[rowind[i]] = (1 - omega) * yarr[rowind[i]] + omega * idiagarr[rowind[i]] * (sum + barr[rowind[i]]);
    }

    PetscCall(VecRestoreArray(y, &yarr));
    if (sct) PetscCall(VecRestoreArrayRead(ghostvec[color], &ghostarr));
    PetscCall(ISRestoreIndices(ind[color], &rowind));
  }

  PetscCall(VecRestoreArrayRead(b, &barr));
  PetscCall(VecRestoreArrayRead(idiag, &idiagarr));

  PetscFunctionReturn(PETSC_SUCCESS);
}
