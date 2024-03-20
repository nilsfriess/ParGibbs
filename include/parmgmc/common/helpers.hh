#pragma once

#include "parmgmc/common/petsc_helper.hh"
#include "petscao.h"
#include "petscdmda.h"
#include "petscdmtypes.h"

#include <algorithm>
#include <memory>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscviewer.h>
#include <random>

#include <mpi.h>
#include <petscerror.h>
#include <petsclog.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <utility>

namespace parmgmc {
template <class Engine>
inline PetscErrorCode fillVecRand(Vec vec, PetscInt size, Engine &engine) {
  static std::normal_distribution<PetscReal> dist;

  PetscFunctionBeginUser;

  PetscCall(PetscHelper::beginRngEvent());

  PetscScalar *rArr;
  PetscCall(VecGetArrayWrite(vec, &rArr));
  std::generate_n(rArr, size, [&]() { return dist(engine); });
  PetscCall(VecRestoreArrayWrite(vec, &rArr));

  // Estimated using perf
  PetscCall(PetscLogFlops(size * 27));

  PetscCall(PetscHelper::endRngEvent());

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine>
inline PetscErrorCode fillVecRand(Vec vec, Engine &engine) {
  PetscFunctionBeginUser;

  PetscInt size;
  PetscCall(VecGetLocalSize(vec, &size));

  PetscCall(fillVecRand(vec, size, engine));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode vecScatterForMat(Mat m, VecScatter *scatter,
                                         Vec sctVec = nullptr) {
  PetscFunctionBeginUser;

  PetscInt firstRow;
  PetscCall(MatGetOwnershipRange(m, &firstRow, nullptr));

  Mat b;
  const PetscInt *colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(m, nullptr, &b, &colmap));

  const PetscInt *bi, *bj;
  PetscCall(MatSeqAIJGetCSRAndMemType(b, &bi, &bj, nullptr, nullptr));

  PetscInt localRows, globalRows;
  PetscCall(MatGetSize(m, &globalRows, nullptr));
  PetscCall(MatGetLocalSize(m, &localRows, nullptr));

  /* This array will have non-zero values at indices corresponding to ghost
     vertices. These are identified by looping over the rows of the
     off-diagonal portion B of the given matrix in MPIAIJ format and marking
     columns that contain non-zero entries.

     For example a 5x5 DMDA grid on four processors would lead to an indices
     array (on rank 0) that could look like this:
     indices = {0 0 0 0 0 0 0 0 0 2 0 5 0 8 0 6 7 8 0 0 0 0 0 0 0}.

     The values of the non-zero entries are the rows where B contains non-zero
     columns, using global indexing. These are used below to determine which
     ghost values have to be communicated during red Gibbs sweeps and which
     during black sweeps.
   */
  std::vector<PetscInt> indices(globalRows, 0);
  std::size_t nzCols = 0;
  for (PetscInt row = 0; row < localRows; ++row) {
    for (PetscInt k = bi[row]; k < bi[row + 1]; ++k) {
      /* We have to use colmap here since B is compactified, i.e., its
       * non-zero columns are {0, ..., nz_cols} (see MatSetUpMultiply_MPIAIJ).
       */
      if (!indices[colmap[bj[k]]])
        nzCols++;
      indices[colmap[bj[k]]] = 1;
    }
  }

  // Form array of needed columns.
  std::vector<PetscInt> ghostArr(nzCols);
  PetscInt cnt = 0;
  for (PetscInt i = 0; i < globalRows; ++i)
    if (indices[i])
      ghostArr[cnt++] = i;

  IS from;
  PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
                            ghostArr.size(),
                            ghostArr.data(),
                            PETSC_COPY_VALUES,
                            &from));

  bool returnSctVec = true;
  if (sctVec == nullptr)
    returnSctVec = false;

  Vec gvec, lvec;

  PetscCall(MatCreateVecs(b, &lvec, nullptr));
  // Create global vec without allocating actualy memory
  PetscCall(VecCreateMPIWithArray(
      MPI_COMM_WORLD, 1, localRows, globalRows, nullptr, &gvec));

  PetscCall(VecScatterCreate(gvec, from, lvec, nullptr, scatter));

  if (returnSctVec)
    sctVec = lvec;
  else
    PetscCall(VecDestroy(&lvec));

  PetscCall(ISDestroy(&from));
  PetscCall(VecDestroy(&gvec));

  PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace parmgmc
