#pragma once

#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/gibbs.hh"

#include <memory>

#include <petscdm.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class MultigridSampler {
public:
  MultigridSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
                   std::size_t n_levels, std::size_t n_smooth = 2)
      : n_levels{n_levels}, n_smooth{n_smooth} {
    PetscFunctionBeginUser;

    ops.resize(n_levels);
    ops[n_levels - 1] = grid_operator;

    interpolations.resize(n_levels - 1);

    for (std::size_t level = n_levels - 1; level > 0; --level) {
      ops[level - 1] = std::make_shared<GridOperator>();
      PetscCallVoid(
          DMCoarsen(ops[level]->dm, MPI_COMM_WORLD, &ops[level - 1]->dm));
      PetscCallVoid(DMCreateInterpolation(ops[level - 1]->dm,
                                          ops[level]->dm,
                                          &interpolations[level - 1],
                                          NULL));

      // Create coarser matrix using Galerkin projection
      PetscCallVoid(MatPtAP(ops[level]->mat,
                            interpolations[level - 1],
                            MAT_INITIAL_MATRIX,
                            PETSC_DEFAULT,
                            &ops[level - 1]->mat));

      PetscCallVoid(ops[level - 1]->color_general());
      MatType type;
      PetscCallVoid(MatGetType(ops[level - 1]->mat, &type));
      if (std::strcmp(type, MATMPIAIJ) == 0) {
        PetscCallVoid(ops[level - 1]->create_rb_scatter());
      }
    }

    // Create work vectors and smoothers
    vs.resize(n_levels);
    xs.resize(n_levels);
    rs.resize(n_levels);

    for (std::size_t level = 0; level < n_levels; ++level) {
      PetscCallVoid(DMCreateGlobalVector(ops[level]->dm, &vs[level]));
      PetscCallVoid(VecDuplicate(vs[level], &xs[level]));
      PetscCallVoid(VecDuplicate(vs[level], &rs[level]));

      smoothers.push_back(
          std::make_shared<GibbsSampler<Engine>>(ops[level], engine));
    }

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, const Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(sample_impl(n_levels - 1, sample, rhs));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~MultigridSampler() {
    PetscFunctionBeginUser;

    for (auto &M : interpolations)
      PetscCallVoid(MatDestroy(&M));

    for (auto &V : vs)
      PetscCallVoid(VecDestroy(&V));

    for (auto &V : rs)
      PetscCallVoid(VecDestroy(&V));

    for (auto &V : xs)
      PetscCallVoid(VecDestroy(&V));

    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode sample_impl(std::size_t level, Vec sample, const Vec rhs) {
    PetscFunctionBeginUser;

    if (level > 0) {
      // Pre smooth
      smoothers[level]->setSweepType(GibbsSweepType::FORWARD);
      PetscCall(smoothers[level]->sample(sample, rhs, n_smooth));

      // Restrict residual
      PetscCall(MatResidual(ops[level]->mat, rhs, sample, rs[level]));
      PetscCall(
          MatRestrict(interpolations[level - 1], rs[level], xs[level - 1]));

      // Recursive call to multigrid sampler
      PetscCall(VecZeroEntries(vs[level - 1]));
      PetscCall(sample_impl(level - 1, vs[level - 1], xs[level - 1]));

      // Prolongate add result
      PetscCall(MatInterpolateAdd(
          interpolations[level - 1], xs[level - 1], sample, sample));

      // Post smooth
      smoothers[level]->setSweepType(GibbsSweepType::BACKWARD);
      PetscCall(smoothers[level]->sample(sample, rhs, n_smooth));
    } else {
      // Coarse level
      smoothers[0]->setSweepType(GibbsSweepType::SYMMETRIC);
      PetscCall(smoothers[0]->sample(sample, rhs, 2 * n_smooth));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::vector<std::shared_ptr<GridOperator>> ops;
  std::vector<std::shared_ptr<GibbsSampler<Engine>>> smoothers;

  std::vector<Mat> interpolations;
  std::vector<Vec> vs; // sample vector for each level
  std::vector<Vec> rs; // residual vector for each level
  std::vector<Vec> xs; // work vector for each level

  std::size_t n_levels;
  std::size_t n_smooth;
};
} // namespace parmgmc
