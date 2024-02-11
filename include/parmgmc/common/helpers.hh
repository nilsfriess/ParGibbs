#pragma once

#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/types.hh"
#include "petscao.h"
#include "petscdmda.h"
#include "petscdmtypes.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsf.h>
#include <petscviewer.h>
#include <random>

#include <mpi.h>
#include <petscerror.h>
#include <petsclog.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <set>
#include <utility>

namespace parmgmc {
template <class Engine>
inline PetscErrorCode fill_vec_rand(Vec vec, PetscInt size, Engine &engine) {
  static std::normal_distribution<PetscReal> dist;

  PetscFunctionBeginUser;

  PetscLogEvent rng_event;
  PetscCall(PetscHelper::get_rng_event(&rng_event));
  PetscCall(PetscLogEventBegin(rng_event, NULL, NULL, NULL, NULL));

  PetscScalar *r_arr;
  PetscCall(VecGetArrayWrite(vec, &r_arr));
  std::generate_n(r_arr, size, [&]() { return dist(engine); });
  PetscCall(VecRestoreArrayWrite(vec, &r_arr));

  // Estimated using perf
  PetscCall(PetscLogFlops(size * 27));

  PetscCall(PetscLogEventEnd(rng_event, NULL, NULL, NULL, NULL));

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

inline PetscErrorCode ISColoring_for_Mat(Mat m, ISColoring *coloring) {
  PetscFunctionBeginUser;

  MatColoring mc;
  PetscCall(MatColoringCreate(m, &mc));
  PetscCall(MatColoringSetDistance(mc, 1));
  PetscCall(MatColoringSetType(mc, MATCOLORINGJP));
  PetscCall(MatColoringApply(mc, coloring));
  PetscCall(MatColoringDestroy(&mc));

  PetscCall(ISColoringSetType(*coloring, IS_COLORING_LOCAL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode ISColoring_for_Mat(Mat mat, DM dm, ISColoring *coloring) {
  PetscFunctionBeginUser;

  const PetscInt ncolors = 2;

  PetscInt start, end;
  PetscCall(MatGetOwnershipRange(mat, &start, &end));

  // Global indices owned by current MPI rank
  std::vector<PetscInt> indices(end - start);
  std::iota(indices.begin(), indices.end(), start);

  // Convert to natural indices
  AO ao;
  PetscCall(DMDAGetAO(dm, &ao));
  PetscCall(AOPetscToApplication(ao, indices.size(), indices.data()));

  std::vector<ISColoringValue> colors(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] % 2 == 0)
      colors[i] = 0; // red
    else
      colors[i] = 1; // black
  }

  PetscCall(ISColoringCreate(MPI_COMM_WORLD,
                             ncolors,
                             colors.size(),
                             colors.data(),
                             PETSC_COPY_VALUES,
                             coloring));
  PetscCall(ISColoringSetType(*coloring, IS_COLORING_LOCAL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode VecScatter_for_Mat(Mat m, VecScatter *scatter,
                                         Vec sct_vec = nullptr) {
  PetscFunctionBeginUser;

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

inline PetscErrorCode make_botmidtop_partition(Mat mat,
                                               BotMidTopPartition &partition) {
  PetscFunctionBeginUser;

  // Get the column layout of the matrix, i.e., the mapping from column ->
  // owning MPI rank
  PetscLayout layout;
  PetscCall(MatGetLayouts(mat, nullptr, &layout));

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  Mat A, B;
  const PetscInt *colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &A, &B, &colmap));

  const PetscInt *Bi, *Bj;
  PetscCall(MatSeqAIJGetCSRAndMemType(B, &Bi, &Bj, nullptr, nullptr));

  PetscInt local_rows, global_rows;
  PetscCall(MatGetSize(mat, &global_rows, nullptr));
  PetscCall(MatGetLocalSize(mat, &local_rows, nullptr));

  ////////////////////////////////////////////////////////////////////////
  //// STEP 1. Create top, bot and interior sets and scatter contexts ////
  ////////////////////////////////////////////////////////////////////////
  std::vector<PetscInt> interior;
  interior.reserve(local_rows); // TODO: this is more than is needed

  // `top_from` are (global) indices of remote vertices from lower processes
  // that are needed by the respective local index in `top_to`.
  std::vector<PetscInt> top_from;
  std::vector<PetscInt> top_to;

  // `bot_from` are (global) indices of remote vertices from higher processes
  // that are needed by the respective local index in `bot_to`.
  std::vector<PetscInt> bot_from;
  std::vector<PetscInt> bot_to;

  std::set<PetscInt> top_vertices, bot_vertices;

  std::vector<PetscInt> curr_from;
  curr_from.reserve(4);

  for (PetscInt i = 0; i < local_rows; ++i) {
    bool top = false;
    bool bot = false;

    curr_from.clear();

    /* Loop over the off-processor entries in row i and find out which MPI
     * process owns the respective vertex to decide if the current row
     * corresponds to a top, bot or mid vertex. */
    for (PetscInt j = Bi[i]; j < Bi[i + 1]; ++j) {
      auto col = colmap[Bj[j]];

      PetscMPIInt owner;
      PetscCall(PetscLayoutFindOwner(layout, col, &owner));

      if (owner > rank)
        bot = true;
      if (owner < rank)
        top = true;

      curr_from.push_back(col);
    }

    // An interior node is a node without neighbours on other processes. Thus we
    // can simply check if the off-processor part B contains any entries in the
    // current row.
    if (Bi[i] == Bi[i + 1]) {
      interior.push_back(i);
    } else {
      // Node is some type of border node, check if it's top, bot or mid
      if (top and not bot) {
        top_vertices.insert(i);

        for (auto from : curr_from) {
          top_from.push_back(from);
          top_to.push_back(i);
        }
      } else if (bot and not top) {
        bot_vertices.insert(i);

        for (auto from : curr_from) {
          bot_from.push_back(from);
          bot_to.push_back(i);
        }
      }
    }
  }

  Vec lvec, gvec;
  PetscCall(VecCreateSeq(MPI_COMM_SELF, local_rows, &lvec));
  PetscCall(VecCreateMPIWithArray(
      MPI_COMM_WORLD, 1, local_rows, global_rows, nullptr, &gvec));

  IS from;
  PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
                            top_from.size(),
                            top_from.data(),
                            PETSC_COPY_VALUES,
                            &from));
  IS to;
  PetscCall(ISCreateGeneral(
      MPI_COMM_WORLD, top_to.size(), top_to.data(), PETSC_COPY_VALUES, &to));

  PetscCall(VecScatterCreate(gvec, from, lvec, to, &partition.topscatter));
  PetscCall(VecScatterSetUp(partition.topscatter));

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));

  PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
                            bot_from.size(),
                            bot_from.data(),
                            PETSC_COPY_VALUES,
                            &from));
  PetscCall(ISCreateGeneral(
      MPI_COMM_WORLD, bot_to.size(), bot_to.data(), PETSC_COPY_VALUES, &to));

  PetscCall(VecScatterCreate(gvec, from, lvec, to, &partition.botscatter));
  PetscCall(VecScatterSetUp(partition.botscatter));

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));

  PetscCall(VecDestroy(&lvec));
  PetscCall(VecDestroy(&gvec));

  partition.bot.clear();
  partition.top.clear();

  std::copy(top_vertices.begin(),
            top_vertices.end(),
            std::back_inserter(partition.top));
  std::copy(bot_vertices.begin(),
            bot_vertices.end(),
            std::back_inserter(partition.bot));

  //////////////////////////////////////////////////////////////////
  //// STEP 2. Partition interior nodes into 1&2 parts          ////
  //////////////////////////////////////////////////////////////////
  const PetscInt *ii, *jj;
  PetscBool done;
  PetscCall(
      MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &ii, &jj, &done));
  assert(done); // TODO: Handle errors

  // Estimate cost of set of vertices by summing number of neighbors of each
  // index
  const auto cost = [&](const std::vector<PetscInt> &indices) {
    PetscInt c = 0;
    for (auto i : indices)
      c += ii[i + 1] - ii[i];
    return c;
  };

  auto top_cost = cost(partition.top);
  auto bot_cost = cost(partition.bot);
  auto int_cost = cost(interior);
  
  // We want to achieve (approximately) |Int1| + |Top| = |Int2| + |Bot|. Let
  // |Int| = |Int1| + |Int2|. Then we see that 2|Int1| + |Top| = |Int| + |Bot|
  // or |Int1| = (|Int| + |Bot| - |Top|) / 2.
  auto tgt_cost = std::max((int_cost + bot_cost - top_cost) / 2., 0.);

  double curr_cost = 0;
  std::size_t split_point;

  for (split_point = 0; split_point < interior.size(); ++split_point) {
    curr_cost += ii[split_point + 1] - ii[split_point];

    if (curr_cost >= tgt_cost)
      break;
  }

  partition.interior1 =
      std::vector(interior.begin(), interior.begin() + split_point);
  partition.interior2 =
      std::vector(interior.begin() + split_point, interior.end());

  PetscFunctionReturn(PETSC_SUCCESS);
}
} // namespace parmgmc
