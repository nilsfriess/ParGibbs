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
enum class MGMCSmoothingType { ForwardBackward, Symmetric };
enum class MGMCCycleType : unsigned int { V = 1, W = 2 };

template <class Engine> class MultigridSampler {
public:
  MultigridSampler(
      std::shared_ptr<GridOperator> grid_operator, Engine *engine,
      std::size_t n_levels_, std::size_t n_smooth = 2,
      MGMCCycleType cycle_type = MGMCCycleType::V,
      MGMCSmoothingType smoothing_type = MGMCSmoothingType::ForwardBackward)
      : n_levels{n_levels_}, n_smooth{n_smooth}, smoothing_type{smoothing_type},
        cycles{static_cast<unsigned int>(cycle_type)} {
    PetscFunctionBeginUser;

    PetscInt npoints;
    PetscCallVoid(DMDAGetCorners(
        grid_operator->get_dm(), NULL, NULL, NULL, &npoints, NULL, NULL));
    if ((npoints - 1) / (1 << (n_levels - 1)) == 0) {
      n_levels = std::floor(std::log2(npoints)) + 1;

      PetscCallVoid(PetscPrintf(MPI_COMM_WORLD,
                                "[ParMGMC] In initilisation of MGMC: too many "
                                "levels. Using %zu instead.\n",
                                n_levels));
    }

    ops.resize(n_levels);
    ops[n_levels - 1] = grid_operator;

    interpolations.resize(n_levels - 1);

    for (std::size_t level = n_levels - 1; level > 0; --level) {
      DM coarse_dm;
      PetscCallVoid(
          DMCoarsen(ops[level]->get_dm(), MPI_COMM_WORLD, &coarse_dm));

      PetscCallVoid(DMCreateInterpolation(
          coarse_dm, ops[level]->get_dm(), &interpolations[level - 1], NULL));

      // Create coarser matrix using Galerkin projection
      Mat coarse_mat;
      PetscCallVoid(MatPtAP(ops[level]->get_mat(),
                            interpolations[level - 1],
                            MAT_INITIAL_MATRIX,
                            PETSC_DEFAULT,
                            &coarse_mat));

      ops[level - 1] = std::make_shared<GridOperator>(
          coarse_dm,
          coarse_mat,
          ops[level]->corners().first,
          ops[level]->corners().second,
          ops[n_levels - 1]->get_coloring_type());
    }

    // Create work vectors and smoothers
    bs.resize(n_levels);
    xs.resize(n_levels);
    rs.resize(n_levels);

    for (std::size_t level = 0; level < n_levels; ++level) {
      PetscCallVoid(DMCreateGlobalVector(ops[level]->get_dm(), &bs[level]));
      PetscCallVoid(VecDuplicate(bs[level], &xs[level]));
      PetscCallVoid(VecDuplicate(bs[level], &rs[level]));

      smoothers.push_back(
          std::make_shared<GibbsSampler<Engine>>(ops[level], engine));
    }

    if (smoothing_type == MGMCSmoothingType::Symmetric)
      for (auto &smoother : smoothers)
        smoother->setSweepType(GibbsSweepType::Symmetric);

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, const Vec rhs, std::size_t n_samples) {
    PetscFunctionBeginUser;

    PetscCall(VecCopy(rhs, bs[n_levels - 1]));
    PetscCall(VecCopy(sample, xs[n_levels - 1]));

    for (std::size_t n = 0; n < n_samples; ++n)
      PetscCall(sample_impl(n_levels - 1));

    PetscCall(VecCopy(xs[n_levels - 1], sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~MultigridSampler() {
    PetscFunctionBeginUser;

    for (auto &M : interpolations)
      PetscCallVoid(MatDestroy(&M));

    for (auto &V : bs)
      PetscCallVoid(VecDestroy(&V));

    for (auto &V : rs)
      PetscCallVoid(VecDestroy(&V));

    for (auto &V : xs)
      PetscCallVoid(VecDestroy(&V));

    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode sample_impl(std::size_t level) {
    PetscFunctionBeginUser;

    if (level > 0) {
      // Pre smooth
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        smoothers[level]->setSweepType(GibbsSweepType::Forward);
      PetscCall(smoothers[level]->sample(xs[level], bs[level], n_smooth));

      // Restrict residual
      PetscCall(
          MatResidual(ops[level]->get_mat(), bs[level], xs[level], rs[level]));
      PetscCall(
          MatRestrict(interpolations[level - 1], rs[level], bs[level - 1]));

      // Recursive call to multigrid sampler
      PetscCall(VecZeroEntries(xs[level - 1]));

      for (std::size_t c = 0; c < cycles; ++c)
        PetscCall(sample_impl(level - 1));

      // Prolongate add result
      PetscCall(MatInterpolateAdd(
          interpolations[level - 1], xs[level - 1], xs[level], xs[level]));

      // Post smooth
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        smoothers[level]->setSweepType(GibbsSweepType::Backward);
      PetscCall(smoothers[level]->sample(xs[level], bs[level], n_smooth));
    } else {
      // Coarse level
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        smoothers[0]->setSweepType(GibbsSweepType::Symmetric);
      PetscCall(smoothers[0]->sample(xs[level], bs[level], 2 * n_smooth));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::vector<std::shared_ptr<GridOperator>> ops;
  std::vector<std::shared_ptr<GibbsSampler<Engine>>> smoothers;

  std::vector<Mat> interpolations;
  std::vector<Vec> xs; // sample vector for each level
  std::vector<Vec> rs; // residual vector for each level
  std::vector<Vec> bs; // rhs vector for each level

  std::size_t n_levels;
  std::size_t n_smooth;

  MGMCSmoothingType smoothing_type;
  unsigned int cycles;
};
} // namespace parmgmc
