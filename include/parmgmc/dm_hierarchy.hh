#pragma once

#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"

#include <iostream>
#include <vector>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscmat.h>

namespace parmgmc {
enum class DMInitialType { Coarsest, Finest };

class DMHierarchy {
public:
  DMHierarchy() = default;

  // TODO: Add error handling when coarse DM does not have (2^n)-1 vertices per
  // dimension.
  DMHierarchy(DM initialSpace, PetscInt nLevels, bool transferOwnership = true,
              DMInitialType dmtype = DMInitialType::Coarsest)
      : nLevels{nLevels}, transferOwnership{transferOwnership} {
    PetscFunctionBeginUser;

    PARMGMC_INFO << "Start setting up DMHierarchy with " << nLevels << " levels.\n";
    Timer timer;

    interpolations.resize(nLevels - 1);
    dms.resize(nLevels);
    if (dmtype == DMInitialType::Coarsest) {
      dms[0] = initialSpace;

      for (PetscInt l = 0; l < nLevels - 1; ++l) {
        PetscCallVoid(DMRefine(dms[l], MPI_COMM_NULL, &dms[l + 1]));
        PetscCallVoid(DMCreateInterpolation(dms[l], dms[l + 1], &interpolations[l], nullptr));
      }

    } else {
      dms[nLevels - 1] = initialSpace;

      for (PetscInt l = nLevels - 1; l > 0; --l) {
        PetscCallVoid(DMCoarsen(dms[l], MPI_COMM_NULL, &dms[l - 1]));
        PetscCallVoid(DMCreateInterpolation(dms[l - 1], dms[l], &interpolations[l], nullptr));
      }
    }

    auto elapsed = timer.elapsed();

    PetscInt m, n, p;

    PetscCallVoid(DMDAGetInfo(getCoarse(), nullptr, &m, &n, &p, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr));
    auto coarseVertices = m * n * p;
    PetscCallVoid(DMDAGetInfo(getFine(), nullptr, &m, &n, &p, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr));
    auto fineVertices = m * n * p;

    PARMGMC_INFO << "Done setting up DMHierarchy (took " << elapsed << " seconds, finest level has "
                 << fineVertices << " vertices, coarsest has " << coarseVertices << ").\n";

    PetscFunctionReturnVoid();
  }

  DMHierarchy(const DMHierarchy &) = delete;
  DMHierarchy(DMHierarchy &&) = default;

  DMHierarchy &operator=(const DMHierarchy &) = delete;
  DMHierarchy &operator=(DMHierarchy &&) = default;

  /* Get matrix that represents interpolation operator from level `level` to
   * level `level + 1`.*/
  [[nodiscard]] Mat getInterpolation(PetscInt level) const {
    if (level >= nLevels - 1)
      return nullptr;
    return interpolations[level];
  }

  /* Get DM on level `level`. The coarsest level is 0, the finest is level
   * `num_levels() - 1`. */
  [[nodiscard]] DM getDm(PetscInt level) const {
    if (level > nLevels - 1)
      return nullptr;
    return dms[level];
  }

  [[nodiscard]] DM getCoarse() const { return dms[0]; }
  [[nodiscard]] DM getFine() const { return dms[nLevels - 1]; }

  [[nodiscard]] PetscInt numLevels() const { return nLevels; }

  [[nodiscard]] PetscErrorCode printInfo() const {
    PetscFunctionBeginUser;

    for (auto dm : dms)
      PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~DMHierarchy() {
    PetscFunctionBeginUser;

    for (std::size_t i = 0; i < dms.size(); ++i) {
      if (i == 0 && !transferOwnership)
        continue;
      PetscCallVoid(DMDestroy(&dms[i]));
    }

    for (auto i : interpolations)
      PetscCallVoid(MatDestroy(&i));

    PetscFunctionReturnVoid();
  }

private:
  std::vector<DM> dms;
  std::vector<Mat> interpolations;

  PetscInt nLevels;

  bool transferOwnership;
};
} // namespace parmgmc
