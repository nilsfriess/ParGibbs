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
inline PetscErrorCode fill_vec_rand(Vec vec, PetscInt size, Engine &engine) {
  static std::normal_distribution<PetscReal> dist;

  PetscFunctionBeginUser;

  PetscCall(PetscHelper::begin_rng_event());

  PetscScalar *r_arr;
  PetscCall(VecGetArrayWrite(vec, &r_arr));
  std::generate_n(r_arr, size, [&]() { return dist(engine); });
  PetscCall(VecRestoreArrayWrite(vec, &r_arr));

  // Estimated using perf
  PetscCall(PetscLogFlops(size * 27));

  PetscCall(PetscHelper::end_rng_event());

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine>
inline PetscErrorCode fill_vec_rand(Vec vec, Engine &engine) {
  PetscFunctionBeginUser;

  PetscInt size;
  PetscCall(VecGetLocalSize(vec, &size));

  PetscCall(fill_vec_rand(vec, size, engine));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode VecScatter_for_Mat(Mat m, VecScatter *scatter,
                                         Vec sct_vec = nullptr) {
  PetscFunctionBeginUser;

  PetscInt first_row;
  PetscCall(MatGetOwnershipRange(m, &first_row, NULL));

  Mat B;
  const PetscInt *colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(m, NULL, &B, &colmap));

  const PetscInt *Bi, *Bj;
  PetscCall(MatSeqAIJGetCSRAndMemType(B, &Bi, &Bj, NULL, NULL));

  PetscInt local_rows, global_rows;
  PetscCall(MatGetSize(m, &global_rows, NULL));
  PetscCall(MatGetLocalSize(m, &local_rows, NULL));

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
  std::vector<PetscInt> indices(global_rows, 0);
  std::size_t nz_cols = 0;
  for (PetscInt row = 0; row < local_rows; ++row) {
    for (PetscInt k = Bi[row]; k < Bi[row + 1]; ++k) {
      /* We have to use colmap here since B is compactified, i.e., its
       * non-zero columns are {0, ..., nz_cols} (see MatSetUpMultiply_MPIAIJ).
       */
      if (!indices[colmap[Bj[k]]])
        nz_cols++;
      indices[colmap[Bj[k]]] = 1;
    }
  }

  // Form array of needed columns.
  std::vector<PetscInt> ghost_arr(nz_cols);
  PetscInt cnt = 0;
  for (PetscInt i = 0; i < global_rows; ++i)
    if (indices[i])
      ghost_arr[cnt++] = i;

  IS from;
  PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
                            ghost_arr.size(),
                            ghost_arr.data(),
                            PETSC_COPY_VALUES,
                            &from));

  bool return_sct_vec = true;
  if (sct_vec == nullptr)
    return_sct_vec = false;

  Vec gvec, lvec;

  PetscCall(MatCreateVecs(B, &lvec, NULL));
  // Create global vec without allocating actualy memory
  PetscCall(VecCreateMPIWithArray(
      MPI_COMM_WORLD, 1, local_rows, global_rows, nullptr, &gvec));

  PetscCall(VecScatterCreate(gvec, from, lvec, nullptr, scatter));

  if (return_sct_vec)
    sct_vec = lvec;
  else
    PetscCall(VecDestroy(&lvec));

  PetscCall(ISDestroy(&from));
  PetscCall(VecDestroy(&gvec));

  PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace parmgmc
