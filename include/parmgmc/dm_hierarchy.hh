#pragma once

#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"

#include <memory>
#include <vector>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscmat.h>

namespace parmgmc {
class DMHierarchy {
public:
  // TODO: Add error handling when coarse DM does not have (2^n)-1 vertices per
  // dimension.
  DMHierarchy(DM coarseSpace, std::size_t nLevels,
              bool transferOwnership = true)
      : nLevels{nLevels}, transferOwnership{transferOwnership} {
    PetscFunctionBeginUser;

    PARMGMC_INFO << "Start setting up DMHierarchy with " << nLevels
                 << " levels.\n";
    Timer timer;

    interpolations.resize(nLevels - 1);

    DM coarseDm = coarseSpace;
    dms.push_back(coarseDm);

    for (std::size_t level = 0; level < nLevels - 1; level++) {
      DM fineDm;
      PetscCallVoid(DMRefine(coarseDm, MPI_COMM_WORLD, &fineDm));

      PetscCallVoid(DMCreateInterpolation(
          coarseDm, fineDm, &interpolations[level], nullptr));

      dms.push_back(fineDm);
      coarseDm = fineDm;
    }

    auto elapsed = timer.elapsed();

    const auto numVertices =
        [](PetscInt dim, PetscInt m, PetscInt n, PetscInt p) {
          PetscInt res = m;
          if (dim > 1)
            res *= n;
          if (dim > 2)
            res *= p;
          return res;
        };

    PetscInt dim, m, n, p;

    PetscCallVoid(DMDAGetInfo(getCoarse(),
                              &dim,
                              &m,
                              &n,
                              &p,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr));
    auto coarseVertices = numVertices(dim, m, n, p);
    PetscCallVoid(DMDAGetInfo(getFine(),
                              &dim,
                              &m,
                              &n,
                              &p,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr));
    auto fineVertices = numVertices(dim, m, n, p);

    PARMGMC_INFO << "Done setting up DMHierarchy (took " << elapsed
                 << " seconds, finest level has " << fineVertices
                 << " vertices, coarsest has " << coarseVertices << ").\n";

    PetscFunctionReturnVoid();
  }

  /* Get matrix that represents interpolation operator from level `level` to
   * level `level + 1`.*/
  [[nodiscard]] Mat getInterpolation(std::size_t level) const {
    if (level >= nLevels - 1)
      return nullptr;
    return interpolations[level];
  }

  /* Get DM on level `level`. The coarsest level is 0, the finest is level
   * `num_levels() - 1`. */
  [[nodiscard]] DM getDm(std::size_t level) const {
    if (level > nLevels - 1)
      return nullptr;
    return dms[level];
  }

  [[nodiscard]] DM getCoarse() const { return dms[0]; }
  [[nodiscard]] DM getFine() const { return dms[nLevels - 1]; }

  [[nodiscard]] std::size_t numLevels() const { return nLevels; }

  [[nodiscard]] PetscErrorCode printInfo() const {
    PetscFunctionBeginUser;

    for (auto dm : dms)
      PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~DMHierarchy() {
    PetscFunctionBeginUser;

    if (transferOwnership)
      PetscCallVoid(DMDestroy(&dms[0]));

    for (std::size_t level = 1; level < nLevels; ++level)
      PetscCallVoid(DMDestroy(&dms[level]));

    for (auto i : interpolations)
      PetscCallVoid(MatDestroy(&i));

    PetscFunctionReturnVoid();
  }

private:
  std::vector<DM> dms;
  std::vector<Mat> interpolations;

  std::size_t nLevels;

  bool transferOwnership;
};
} // namespace parmgmc
