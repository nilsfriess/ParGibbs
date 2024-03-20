#pragma once

#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <iostream>
#include <memory>

#include <petscdm.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
#include "parmgmc/samplers/cholesky.hh"
#endif

namespace parmgmc {
enum class MGMCSmoothingType { ForwardBackward, Symmetric };
enum class MGMCCycleType : unsigned int { V = 1, W = 2 };
enum class MGMCCoarseSamplerType {
  Standard
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
  ,
  Cholesky
#endif
};

struct MGMCParameters {
  std::size_t n_smooth;
  MGMCSmoothingType smoothing_type;
  MGMCCycleType cycle_type;
  MGMCCoarseSamplerType coarse_sampler_type;

  static MGMCParameters Default() {
    MGMCParameters params;
    params.n_smooth = 2;
    params.smoothing_type = MGMCSmoothingType::ForwardBackward;
    params.cycle_type = MGMCCycleType::V;
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
    params.coarse_sampler_type = MGMCCoarseSamplerType::Cholesky;
#else
    params.coarse_sampler_type = MGMCCoarseSamplerType::Standard;
#endif
    return params;
  }
};

template <class Engine, class Smoother = MulticolorGibbsSampler<Engine>> class MultigridSampler {
public:
  /* Construct a Multigrid sampler using a given linear operator and a hierarchy
   * of DMs. The operator must be an operator on the finest DM in the
   * hierarchy, the remaining operators are created by Galerkin projection
   * A_coarse = P^T A_fine P. */
  MultigridSampler(const std::shared_ptr<LinearOperator> &fine_operator,
                   const std::shared_ptr<DMHierarchy> &dm_hierarchy, Engine *engine,
                   const MGMCParameters &params = MGMCParameters::Default())
      : dm_hierarchy{dm_hierarchy}, engine{engine}, n_levels{dm_hierarchy->num_levels()},
        n_smooth{params.n_smooth}, smoothing_type{params.smoothing_type},
        coarse_sampler_type{params.coarse_sampler_type},
        cycles{static_cast<unsigned int>(params.cycle_type)} {
    PetscFunctionBeginUser;

    PARMGMC_INFO << "Start setting up Multigrid sampler using DM hierarchy ("
                 << dm_hierarchy->num_levels() << " levels).\n";
    Timer timer;

    ops.resize(n_levels);
    ops[n_levels - 1] = fine_operator;

    for (int level = n_levels - 1; level > 0; --level) {
      // Create fine matrix using Galerkin projection
      Mat coarse_mat;
      PetscCallVoid(MatPtAP(ops[level]->get_mat(), dm_hierarchy->get_interpolation(level - 1),
                            MAT_INITIAL_MATRIX, PETSC_DEFAULT, &coarse_mat));
      ops[level - 1] = std::make_shared<LinearOperator>(coarse_mat);
      ops[level - 1]->color_matrix(dm_hierarchy->get_dm(level - 1));
    }

    PetscCallVoid(init_vecs_and_smoothers(engine));

    auto elapsed = timer.elapsed();
    PARMGMC_INFO << "Done setting up Multigrid sampler (took " << elapsed << " seconds).\n";

    PetscFunctionReturnVoid();
  }

  /* Constructor that must be called by classes that are derived from this
   * sampler to define a custom sampler. */
  MultigridSampler(const MGMCParameters &params, std::size_t n_levels, Engine *engine)
      : engine{engine}, n_levels{n_levels}, n_smooth{params.n_smooth},
        smoothing_type{params.smoothing_type}, coarse_sampler_type{params.coarse_sampler_type},
        cycles{static_cast<unsigned int>(params.cycle_type)} {}

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    if (!init_done)
      PetscCall(init_vecs_and_smoothers(engine));

    PetscCall(VecCopy(rhs, bs[n_levels - 1]));
    PetscCall(VecCopy(sample, xs[n_levels - 1]));

    for (std::size_t n = 0; n < n_samples; ++n)
      PetscCall(sample_impl(n_levels - 1));

    PetscCall(VecCopy(xs[n_levels - 1], sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::shared_ptr<LinearOperator> get_operator(std::size_t level) const { return ops[level]; }

  virtual ~MultigridSampler() {
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

    init_done = true;

    bs.resize(n_levels);
    xs.resize(n_levels);
    rs.resize(n_levels);

    for (std::size_t level = 0; level < n_levels; ++level) {
      PetscCall(MatCreateVecs(ops[level]->get_mat(), &bs[level], nullptr));
      PetscCall(VecDuplicate(bs[level], &xs[level]));
      PetscCall(VecDuplicate(bs[level], &rs[level]));

      PetscCall(VecZeroEntries(bs[level]));
      PetscCall(VecZeroEntries(xs[level]));
      PetscCall(VecZeroEntries(rs[level]));
    }
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
    bool coarse_cholesky = coarse_sampler_type == MGMCCoarseSamplerType::Cholesky;
    auto coarsest_smoother_index = coarse_cholesky ? 1 : 0;

    for (std::size_t level = coarsest_smoother_index; level < n_levels; ++level)
      smoothers.push_back(std::make_shared<Smoother>(ops[level], engine));

    if (coarse_cholesky)
      coarse_sampler = std::make_shared<CholeskySampler<Engine>>(ops[0], engine);
#else
    for (std::size_t level = 0; level < n_levels; ++level)
      smoothers.push_back(std::make_shared<Smoother>(ops[level], engine));
#endif

    if (smoothing_type == MGMCSmoothingType::Symmetric)
      for (auto &smoother : smoothers)
        smoother->setSweepType(GibbsSweepType::Symmetric);

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode sample_impl(std::size_t level) {
    PetscFunctionBeginUser;

    if (level > 0) {
      Smoother *curr_smoother;
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
      if (coarse_sampler_type == MGMCCoarseSamplerType::Cholesky) {
        curr_smoother = smoothers[level - 1].get();
      } else {
        curr_smoother = smoothers[level].get();
      }
#else
      curr_smoother = smoothers[level].get();
#endif

      // Pre smooth
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        curr_smoother->setSweepType(GibbsSweepType::Forward);
      PetscCall(curr_smoother->setFixedRhs(bs[level]));
      PetscCall(curr_smoother->sample(xs[level], nullptr, n_smooth));

      // Restrict residual
      PetscCall(MatResidual(ops[level]->get_mat(), bs[level], xs[level], rs[level]));

      PetscCall(restrict(level, rs[level], bs[level - 1]));

      // Recursive call to multigrid sampler
      PetscCall(VecZeroEntries(xs[level - 1]));

      for (std::size_t c = 0; c < cycles; ++c)
        PetscCall(sample_impl(level - 1));

      // Prolongate add result
      PetscCall(prolongate_add(level, xs[level - 1], xs[level]));

      // Post smooth
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        curr_smoother->setSweepType(GibbsSweepType::Backward);
      PetscCall(curr_smoother->setFixedRhs(bs[level]));
      PetscCall(curr_smoother->sample(xs[level], nullptr, n_smooth));
    } else {
      // Coarse level
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
      if (coarse_sampler_type == MGMCCoarseSamplerType::Cholesky) {
        PetscCall(coarse_sampler->sample(xs[0], bs[0]));
      } else {
        if (smoothing_type != MGMCSmoothingType::Symmetric)
          smoothers[0]->setSweepType(GibbsSweepType::Symmetric);
        PetscCall(smoothers[0]->sample(xs[0], bs[0], 2 * n_smooth));
      }
#else
      if (smoothing_type != MGMCSmoothingType::Symmetric)
        smoothers[0]->setSweepType(GibbsSweepType::Symmetric);
      PetscCall(smoothers[0]->sample(xs[0], bs[0], 2 * n_smooth));
#endif
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  bool init_done = false;

  std::shared_ptr<DMHierarchy> dm_hierarchy;
  std::vector<std::shared_ptr<Smoother>> smoothers;

#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
  std::shared_ptr<CholeskySampler<Engine>> coarse_sampler;
#endif

  Engine *engine;

protected:
  virtual PetscErrorCode restrict(std::size_t level, Vec residual, Vec rhs) {
    return MatRestrict(dm_hierarchy->get_interpolation(level - 1), residual, rhs);
  }

  virtual PetscErrorCode prolongate_add(std::size_t level, Vec coarse, Vec fine) {
    return MatInterpolateAdd(dm_hierarchy->get_interpolation(level - 1), coarse, fine, fine);
  }

  std::vector<std::shared_ptr<LinearOperator>> ops;

  std::vector<Vec> xs; // sample vector for each level
  std::vector<Vec> rs; // residual vector for each level
  std::vector<Vec> bs; // rhs vector for each level

  std::size_t n_levels;
  std::size_t n_smooth;

  MGMCSmoothingType smoothing_type;
  MGMCCoarseSamplerType coarse_sampler_type;
  unsigned int cycles;
};
} // namespace parmgmc
