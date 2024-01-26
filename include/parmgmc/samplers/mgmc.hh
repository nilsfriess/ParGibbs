#pragma once

#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/gibbs.hh"

#include <iostream>
#include <memory>

#include <petscdm.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

namespace parmgmc {
enum class MGMCSmoothingType { ForwardBackward, Symmetric };
enum class MGMCCycleType : unsigned int { V = 1, W = 2 };

struct MGMCParameters {
  std::size_t n_smooth;
  MGMCSmoothingType smoothing_type;
  MGMCCycleType cycle_type;
};

template <class Engine> class MultigridSampler {
public:
  /* Construct a Multigrid sampler using a given hierarchy of DMs and a matrix
   * assembly routine. For each level (the number of levels is determined by the
   * number of DMs in the `dm_hierarchy`) the assembly routine is called with
   * the corresponding DM as its argument. The signature of the assembly
   * function should be `PetscErrorCode assembler(DM, Mat*)`. The Mat must
   * be created (e.g. using DMCreateMatrix) by the assembly function. */
  template <class Assembler>
  MultigridSampler(std::shared_ptr<DMHierarchy> dm_hierarchy,
                   Assembler &&assembler, Engine *engine,
                   const MGMCParameters &params)
      : dm_hierarchy{dm_hierarchy}, n_levels{dm_hierarchy->num_levels()},
        n_smooth{params.n_smooth}, smoothing_type{params.smoothing_type},
        cycles{static_cast<unsigned int>(params.cycle_type)} {
    PetscFunctionBeginUser;

    ops.resize(n_levels);
    for (std::size_t level = 0; level < n_levels; level++) {
      Mat mat;
      PetscCallVoid(assembler(dm_hierarchy->get_dm(level), &mat));
      ops[level] = std::make_shared<LinearOperator>(mat);
    }

    PetscCallVoid(init_vecs_and_smoothers(engine));

    PetscFunctionReturnVoid();
  }

  /* Construct a Multigrid sampler using a given linear operator and a hierarchy
   * of DMs. The operators must be an operator on the finest DM in the
   * hierarchy, the remaining operators are created by Galerkin projection
   * A_coarse = P^T A_fine P. */
  MultigridSampler(std::shared_ptr<LinearOperator> fine_operator,
                   std::shared_ptr<DMHierarchy> dm_hierarchy, Engine *engine,
                   const MGMCParameters &params)
      : dm_hierarchy{dm_hierarchy}, n_levels{dm_hierarchy->num_levels()},
        n_smooth{params.n_smooth}, smoothing_type{params.smoothing_type},
        cycles{static_cast<unsigned int>(params.cycle_type)} {
    PetscFunctionBeginUser;

    ops.resize(n_levels);
    ops[n_levels - 1] = fine_operator;

    for (int level = n_levels - 1; level > 0; --level) {
      // Create fine matrix using Galerkin projection
      Mat coarse_mat;
      PetscCallVoid(MatPtAP(ops[level]->get_mat(),
                            dm_hierarchy->get_interpolation(level-1),
                            MAT_INITIAL_MATRIX,
                            PETSC_DEFAULT,
                            &coarse_mat));
      ops[level - 1] = std::make_shared<LinearOperator>(coarse_mat);
    }

    PetscCallVoid(init_vecs_and_smoothers(engine));

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

    for (auto &V : bs)
      PetscCallVoid(VecDestroy(&V));

    for (auto &V : rs)
      PetscCallVoid(VecDestroy(&V));

    for (auto &V : xs)
      PetscCallVoid(VecDestroy(&V));

    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode init_vecs_and_smoothers(Engine *engine) {
    PetscFunctionBeginUser;

    bs.resize(n_levels);
    xs.resize(n_levels);
    rs.resize(n_levels);

    for (std::size_t level = 0; level < n_levels; ++level) {
      PetscCall(MatCreateVecs(ops[level]->get_mat(), &bs[level], NULL));
      PetscCall(VecDuplicate(bs[level], &xs[level]));
      PetscCall(VecDuplicate(bs[level], &rs[level]));

      PetscCall(VecZeroEntries(bs[level]));
      PetscCall(VecZeroEntries(xs[level]));
      PetscCall(VecZeroEntries(rs[level]));

      smoothers.push_back(
          std::make_shared<GibbsSampler<Engine>>(ops[level], engine));
    }

    if (smoothing_type == MGMCSmoothingType::Symmetric)
      for (auto &smoother : smoothers)
        smoother->setSweepType(GibbsSweepType::Symmetric);

    PetscFunctionReturn(PETSC_SUCCESS);
  }

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

      PetscCall(MatRestrict(dm_hierarchy->get_interpolation(level - 1),
                            rs[level],
                            bs[level - 1]));

      // Recursive call to multigrid sampler
      PetscCall(VecZeroEntries(xs[level - 1]));

      for (std::size_t c = 0; c < cycles; ++c)
        PetscCall(sample_impl(level - 1));

      // Prolongate add result
      PetscCall(MatInterpolateAdd(dm_hierarchy->get_interpolation(level - 1),
                                  xs[level - 1],
                                  xs[level],
                                  xs[level]));

      // Post smooth
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        smoothers[level]->setSweepType(GibbsSweepType::Backward);
      PetscCall(smoothers[level]->sample(xs[level], bs[level], n_smooth));
    } else {
      // Coarse level
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        smoothers[0]->setSweepType(GibbsSweepType::Symmetric);
      PetscCall(smoothers[0]->sample(xs[0], bs[0], 2 * n_smooth));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::vector<std::shared_ptr<LinearOperator>> ops;
  std::shared_ptr<DMHierarchy> dm_hierarchy;
  std::vector<std::shared_ptr<GibbsSampler<Engine>>> smoothers;

  std::vector<Vec> xs; // sample vector for each level
  std::vector<Vec> rs; // residual vector for each level
  std::vector<Vec> bs; // rhs vector for each level

  std::size_t n_levels;
  std::size_t n_smooth;

  MGMCSmoothingType smoothing_type;
  unsigned int cycles;

  // #ifdef PARMGMC_HAS_MFEM
  //   std::vector<mfem::Operator *> mfem_interpolations;
  // #endif
};
} // namespace parmgmc
