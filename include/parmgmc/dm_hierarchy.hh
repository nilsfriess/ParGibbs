#pragma once

#include <memory>
#include <vector>

#include <petscdmda.h>
#include <petscerror.h>
#include <petscmat.h>

namespace parmgmc {
class DMHierarchy {
public:
  DMHierarchy(DM coarse_space, std::size_t n_levels, bool transfer_ownership = true)
      : n_levels{n_levels}, transfer_ownership{transfer_ownership} {
    PetscFunctionBeginUser;

    interpolations.resize(n_levels - 1);

    DM coarse_dm = coarse_space;
    dms.push_back(coarse_dm);

    for (std::size_t level = 0; level < n_levels - 1; level++) {
      DM fine_dm;
      PetscCallVoid(DMRefine(coarse_dm, MPI_COMM_WORLD, &fine_dm));

      PetscCallVoid(DMCreateInterpolation(
          coarse_dm, fine_dm, &interpolations[level], NULL));

      // if (level != 0)
      //   PetscCallVoid(DMDestroy(&coarse_dm));

      dms.push_back(fine_dm);
      coarse_dm = fine_dm;
    }

    PetscFunctionReturnVoid();
  }

  /* Get matrix that represents interpolation operator from level `level` to
   * level `level + 1`.*/
  Mat get_interpolation(std::size_t level) const {
    if (level >= n_levels - 1)
      return nullptr;
    return interpolations[level];
  }

  /* Get DM on level `level`. The coarsest level is 0, the finest is level
   * `num_levels() - 1`. */
  DM get_dm(std::size_t level) const {
    if (level > n_levels - 1)
      return nullptr;
    return dms[level];
  }

  DM get_coarse() const { return dms[0]; }
  DM get_fine() const { return dms[n_levels - 1]; }

  std::size_t num_levels() const { return n_levels; }

  PetscErrorCode print_info() const {
    PetscFunctionBeginUser;

    for (auto dm : dms)
      PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~DMHierarchy() {
    PetscFunctionBeginUser;

    if (transfer_ownership)
      PetscCallVoid(DMDestroy(&dms[0]));

    for (std::size_t level = 1; level < n_levels; ++level)
      PetscCallVoid(DMDestroy(&dms[level]));

    for (auto I : interpolations)
      PetscCallVoid(MatDestroy(&I));

    PetscFunctionReturnVoid();
  }

private:
  std::vector<DM> dms;
  std::vector<Mat> interpolations;

  std::size_t n_levels;

  bool transfer_ownership;
};
} // namespace parmgmc
