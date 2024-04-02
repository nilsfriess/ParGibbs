#pragma once

#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"
#include "petscao.h"
#include "petscdmda.h"
#include "petscdmtypes.h"
#include "petscerror.h"

#include <cstring>
#include <numeric>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsftypes.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <cstddef>
#include <vector>

namespace parmgmc {
class Coloring {
public:
  Coloring(Mat mat, MatColoringType coloringType = MATCOLORINGJP) {
    PetscFunctionBeginUser;

    // Don't create coloring if running sequentially
    PetscMPIInt size;
    PetscCallVoid(MPI_Comm_size(MPI_COMM_WORLD, &size));
    if (size == 1) {
      PetscInt start, end;
      PetscCallVoid(MatGetOwnershipRange(mat, &start, &end));

      colorIndices.resize(1);
      colorIndices[0].resize(end - start);

      std::iota(colorIndices[0].begin(), colorIndices[0].end(), start);

      PetscFunctionReturnVoid();
    }

    Timer timer;

    MatColoring mc;
    PetscCallVoid(MatColoringCreate(mat, &mc));
    PetscCallVoid(MatColoringSetDistance(mc, 1));
    PetscCallVoid(MatColoringSetType(mc, coloringType));

    ISColoring ic;
    PetscCallVoid(MatColoringApply(mc, &ic));
    PetscCallVoid(ISColoringSetType(ic, IS_COLORING_LOCAL));

    // Copy ISColoring to std::vector<std::vector<PetscInt>>
    PetscInt ncolors;
    IS *isc;
    PetscCallVoid(ISColoringGetIS(ic, PETSC_USE_POINTER, &ncolors, &isc));
    colorIndices.resize(ncolors);
    for (PetscInt i = 0; i < ncolors; ++i) {
      PetscInt nidx;
      const PetscInt *indices;
      PetscCallVoid(ISGetLocalSize(isc[i], &nidx));
      PetscCallVoid(ISGetIndices(isc[i], &indices));
      colorIndices[i].resize(nidx);
      std::copy(indices, indices + nidx, colorIndices[i].begin());
      PetscCallVoid(ISRestoreIndices(isc[i], &indices));
    }

    PetscCallVoid(ISColoringDestroy(&ic));
    PetscCallVoid(MatColoringDestroy(&mc));

    PetscCallVoid(createScatters(mat));

    PARMGMC_INFO << "Created matrix coloring with " << colorIndices.size()
                 << " colors, took " << timer.elapsed() << " seconds.\n";

    PetscFunctionReturnVoid();
  }

  /** Creates a red/black coloring
   */
  Coloring(Mat mat, DM dm) {
    PetscFunctionBeginUser;

    PetscInt start, end;
    PetscCallVoid(MatGetOwnershipRange(mat, &start, &end));

    // Global indices owned by current MPI rank
    std::vector<PetscInt> indices(end - start);
    std::iota(indices.begin(), indices.end(), start);

    // Convert to natural indices
    AO ao;
    PetscCallVoid(DMDAGetAO(dm, &ao));
    PetscCallVoid(AOPetscToApplication(ao, indices.size(), indices.data()));

    colorIndices.resize(2);
    colorIndices[0].reserve((indices.size() + 1) / 2); // red vertices
    colorIndices[1].reserve((indices.size() + 1) / 2); // black vertices

    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] % 2 == 0)
        colorIndices[0].push_back(i);
      else
        colorIndices[1].push_back(i);
    }

    PetscCallVoid(createScatters(mat));

    PetscFunctionReturnVoid();
  }

  template <typename Handler>
  auto forEachIdxOfColor(std::size_t colorIdx, Handler &&h) const {
    PetscFunctionBeginUser;
    for (auto idx : colorIndices.at(colorIdx)) {
      if constexpr (std::is_same_v<decltype(h(1)), void>)
        h(idx);
      else
        PetscCall(h(idx));
    }
    if constexpr (std::is_same_v<decltype(h(1)), void>)
      PetscFunctionReturnVoid();
    else
      PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename Handler> auto forEachColor(Handler &&h) const {
    PetscFunctionBeginUser;
    for (std::size_t i = 0; i < colorIndices.size(); ++i) {
      if constexpr (std::is_same_v<decltype(h(0, colorIndices[0])), void>)
        h(i, colorIndices[i]);
      else
        PetscCall(h(i, colorIndices[i]));
    }
    if constexpr (std::is_same_v<decltype(h(0, colorIndices[0])), void>)
      PetscFunctionReturnVoid();
    else
      PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename Handler> auto forEachColorReverse(Handler &&h) const {
    PetscFunctionBeginUser;
    for (int i = colorIndices.size() - 1; i >= 0; --i) {
      if constexpr (std::is_same_v<decltype(h(0, colorIndices[0])), void>)
        h(i, colorIndices[i]);
      else
        PetscCall(h(i, colorIndices[i]));
    }
    if constexpr (std::is_same_v<decltype(h(0, colorIndices[0])), void>)
      PetscFunctionReturnVoid();
    else
      PetscFunctionReturn(PETSC_SUCCESS);
  }

  // template <typename Handler>
  // PetscErrorCode for_each_idx_of_each_color(Handler &h) const {
  //   PetscFunctionBeginUser;
  //   for (const auto &indices : color_indices)
  //     for (auto idx : indices)
  //       PetscCall(h(idx));
  //   PetscFunctionReturn(PETSC_SUCCESS);
  // }

  [[nodiscard]] VecScatter getScatter(std::size_t colorIdx) const {
    return vecscatters[colorIdx];
  }

  [[nodiscard]] Vec getGhostVec(std::size_t colorIdx) const {
    return ghostVecs[colorIdx];
  }

  ~Coloring() {
    PetscFunctionBeginUser;

    for (auto &vec : ghostVecs)
      PetscCallVoid(VecDestroy(&vec));

    for (auto &scatter : vecscatters)
      PetscCallVoid(VecScatterDestroy(&scatter));

    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode createScatters(Mat mat) {
    PetscFunctionBeginUser;

    MatType type;
    PetscCall(MatGetType(mat, &type));

    // Scatters are only necessary if we're dealing with a parallel matrix
    // TOOD: Check for other formats and throw an error if unsupported
    if (std::strcmp(type, MATMPIAIJ) == 0) {
      /* For each index in each color (which corresponds to a row in the
       * matrix), we get all off-processor neighbors and create a VecScatter for
       * them. */

      Mat ao;
      const PetscInt *colmap;
      PetscCall(MatMPIAIJGetSeqAIJ(mat, nullptr, &ao, &colmap));

      const PetscInt *rowptr, *colptr;
      PetscCall(
          MatSeqAIJGetCSRAndMemType(ao, &rowptr, &colptr, nullptr, nullptr));

      // Create a vector comptible with the matrix (= same shape as the samples
      // used later)
      Vec gvec;
      PetscInt localRows, globalRows;
      PetscCall(MatGetSize(mat, &globalRows, nullptr));
      PetscCall(MatGetLocalSize(mat, &localRows, nullptr));
      PetscCall(VecCreateMPIWithArray(
          MPI_COMM_WORLD, 1, localRows, globalRows, nullptr, &gvec));

      for (std::size_t i = 0; i < colorIndices.size(); ++i) {
        std::vector<PetscInt> offProcIdx;
        for (auto row : colorIndices[i]) {
          for (PetscInt k = rowptr[row]; k < rowptr[row + 1]; ++k) {
            offProcIdx.push_back(colmap[colptr[k]]);
          }
        }

        // Now create a scatter that scatters the values from off_proc_idx into
        // a sequential vector
        IS is;
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,
                                  offProcIdx.size(),
                                  offProcIdx.data(),
                                  PETSC_COPY_VALUES,
                                  &is));
        PetscCall(VecCreateSeq(
            MPI_COMM_SELF, offProcIdx.size(), &ghostVecs.emplace_back()));
        PetscCall(VecScatterCreate(
            gvec, is, ghostVecs[i], nullptr, &vecscatters.emplace_back()));
        PetscCall(ISDestroy(&is));
      }

      PetscCall(VecDestroy(&gvec));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::vector<std::vector<PetscInt>> colorIndices;

  /** Array of VecScatters to send/receive the values required for updating a
   * certain color. That is, on order to update the indices in color_indices[i],
   * one has to perform a VecScatterBegin/End using the VecScatter in
   * vescscatters[i]. */
  std::vector<VecScatter> vecscatters;
  std::vector<Vec> ghostVecs;
};
} // namespace parmgmc