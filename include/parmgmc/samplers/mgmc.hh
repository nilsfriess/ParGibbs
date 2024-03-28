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
  MGMCParameters() {
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
    coarseSamplerType = MGMCCoarseSamplerType::Cholesky;
#else
    coarseSamplerType = MGMCCoarseSamplerType::Standard;
#endif
  }

  static MGMCParameters defaultParams() { return MGMCParameters{}; }

  std::size_t nSmooth{2};
  MGMCSmoothingType smoothingType{MGMCSmoothingType::ForwardBackward};
  MGMCCycleType cycleType{MGMCCycleType::V};
  MGMCCoarseSamplerType coarseSamplerType;
};

template <class Engine, class Smoother = MulticolorGibbsSampler<Engine>> class MultigridSampler {
public:
  /* Construct a Multigrid sampler using a given linear operator and a hierarchy
   * of DMs. The operator must be an operator on the finest DM in the
   * hierarchy, the remaining operators are created by Galerkin projection
   * A_coarse = P^T A_fine P. */
  MultigridSampler(const std::shared_ptr<LinearOperator> &fineOperator,
                   const std::shared_ptr<DMHierarchy> &dmHierarchy, Engine *engine,
                   const MGMCParameters &params = MGMCParameters::defaultParams())
      : dmHierarchy{dmHierarchy}, engine{engine}, nLevels{dmHierarchy->numLevels()},
        nSmooth{params.nSmooth}, smoothingType{params.smoothingType},
        coarseSamplerType{params.coarseSamplerType},
        cycles{static_cast<unsigned int>(params.cycleType)} {
    PetscFunctionBeginUser;

    PARMGMC_INFO << "Start setting up Multigrid sampler using DM hierarchy ("
                 << dmHierarchy->numLevels() << " levels).\n";
    Timer timer;

    ops.resize(nLevels);
    ops[nLevels - 1] = fineOperator;

    for (int level = nLevels - 1; level > 0; --level) {
      // Create fine matrix using Galerkin projection
      Mat coarseMat;
      PetscCallVoid(MatPtAP(ops[level]->getMat(), dmHierarchy->getInterpolation(level - 1),
                            MAT_INITIAL_MATRIX, PETSC_DEFAULT, &coarseMat));
      ops[level - 1] = std::make_shared<LinearOperator>(coarseMat);
      ops[level - 1]->colorMatrix(dmHierarchy->getDm(level - 1));
    }

    PetscCallVoid(initVecsAndSmoothers(engine));

    auto elapsed = timer.elapsed();
    PARMGMC_INFO << "Done setting up Multigrid sampler (took " << elapsed << " seconds).\n";

    PetscFunctionReturnVoid();
  }

  /* Constructor that must be called by classes that are derived from this
   * sampler to define a custom sampler. */
  MultigridSampler(const MGMCParameters &params, std::size_t nLevels, Engine *engine)
      : engine{engine}, nLevels{nLevels}, nSmooth{params.nSmooth},
        smoothingType{params.smoothingType}, coarseSamplerType{params.coarseSamplerType},
        cycles{static_cast<unsigned int>(params.cycleType)} {}

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t nSamples = 1) {
    PetscFunctionBeginUser;

    if (!initDone)
      PetscCall(initVecsAndSmoothers(engine));

    PetscCall(VecCopy(rhs, bs[nLevels - 1]));
    PetscCall(VecCopy(sample, xs[nLevels - 1]));

    for (std::size_t n = 0; n < nSamples; ++n)
      PetscCall(sampleImpl(nLevels - 1));

    PetscCall(VecCopy(xs[nLevels - 1], sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  [[nodiscard]] std::shared_ptr<LinearOperator> getOperator(std::size_t level) const {
    return ops[level];
  }

  virtual ~MultigridSampler() {
    PetscFunctionBeginUser;

    for (auto &v : bs)
      PetscCallVoid(VecDestroy(&v));

    for (auto &v : rs)
      PetscCallVoid(VecDestroy(&v));

    for (auto &v : xs)
      PetscCallVoid(VecDestroy(&v));

    PetscFunctionReturnVoid();
  }

  MultigridSampler(MultigridSampler &) = delete;
  MultigridSampler(MultigridSampler &&) = default;
  MultigridSampler operator=(MultigridSampler &) = delete;
  MultigridSampler operator=(MultigridSampler &&) = delete;

private:
  PetscErrorCode initVecsAndSmoothers(Engine *engine) {
    PetscFunctionBeginUser;

    initDone = true;

    bs.resize(nLevels);
    xs.resize(nLevels);
    rs.resize(nLevels);

    for (std::size_t level = 0; level < nLevels; ++level) {
      PetscCall(MatCreateVecs(ops[level]->getMat(), &bs[level], nullptr));
      PetscCall(VecDuplicate(bs[level], &xs[level]));
      PetscCall(VecDuplicate(bs[level], &rs[level]));

      PetscCall(VecZeroEntries(bs[level]));
      PetscCall(VecZeroEntries(xs[level]));
      PetscCall(VecZeroEntries(rs[level]));
    }
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
    bool coarseCholesky = coarseSamplerType == MGMCCoarseSamplerType::Cholesky;
    auto coarsestSmootherIndex = coarseCholesky ? 1 : 0;

    for (std::size_t level = coarsestSmootherIndex; level < nLevels; ++level)
      smoothers.push_back(std::make_shared<Smoother>(ops[level], engine));

    if (coarseCholesky)
      coarseSampler = std::make_shared<CholeskySampler<Engine>>(ops[0], engine);
#else
    for (std::size_t level = 0; level < nLevels; ++level)
      smoothers.push_back(std::make_shared<Smoother>(ops[level], engine));
#endif

    if (smoothingType == MGMCSmoothingType::Symmetric)
      for (auto &smoother : smoothers)
        smoother->setSweepType(GibbsSweepType::Symmetric);

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode sampleImpl(std::size_t level) {
    PetscFunctionBeginUser;

    if (level > 0) {
      Smoother *currSmoother;
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
      if (coarseSamplerType == MGMCCoarseSamplerType::Cholesky) {
        currSmoother = smoothers[level - 1].get();
      } else {
        currSmoother = smoothers[level].get();
      }
#else
      currSmoother = smoothers[level].get();
#endif

      // Pre smooth
      if (smoothingType != MGMCSmoothingType::Symmetric)
        currSmoother->setSweepType(GibbsSweepType::Forward);
      PetscCall(currSmoother->setFixedRhs(bs[level]));
      PetscCall(currSmoother->sample(xs[level], nullptr, nSmooth));

      // Restrict residual
      PetscCall(MatResidual(ops[level]->getMat(), bs[level], xs[level], rs[level]));

      PetscCall(restrict(level, rs[level], bs[level - 1]));

      // Recursive call to multigrid sampler
      PetscCall(VecZeroEntries(xs[level - 1]));

      for (std::size_t c = 0; c < cycles; ++c)
        PetscCall(sampleImpl(level - 1));

      // Prolongate add result
      PetscCall(prolongateAdd(level, xs[level - 1], xs[level]));

      // Post smooth
      if (smoothingType != MGMCSmoothingType::Symmetric)
        currSmoother->setSweepType(GibbsSweepType::Backward);
      PetscCall(currSmoother->setFixedRhs(bs[level]));
      PetscCall(currSmoother->sample(xs[level], nullptr, nSmooth));
    } else {
      // Coarse level
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
      if (coarseSamplerType == MGMCCoarseSamplerType::Cholesky) {
        PetscCall(coarseSampler->sample(xs[0], bs[0]));
      } else {
        if (smoothingType != MGMCSmoothingType::Symmetric)
          smoothers[0]->setSweepType(GibbsSweepType::Symmetric);
        PetscCall(smoothers[0]->sample(xs[0], bs[0], 2 * nSmooth));
      }
#else
      if (smoothingType != MGMCSmoothingType::Symmetric)
        smoothers[0]->setSweepType(GibbsSweepType::Symmetric);
      PetscCall(smoothers[0]->sample(xs[0], bs[0], 2 * nSmooth));
#endif
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  bool initDone = false;

  std::shared_ptr<DMHierarchy> dmHierarchy;
  std::vector<std::shared_ptr<Smoother>> smoothers;

#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
  std::shared_ptr<CholeskySampler<Engine>> coarseSampler;
#endif

  Engine *engine;

protected:
  virtual PetscErrorCode restrict(std::size_t level, Vec residual, Vec rhs) {
    return MatRestrict(dmHierarchy->getInterpolation(level - 1), residual, rhs);
  }

  virtual PetscErrorCode prolongateAdd(std::size_t level, Vec coarse, Vec fine) {
    return MatInterpolateAdd(dmHierarchy->getInterpolation(level - 1), coarse, fine, fine);
  }

  std::vector<std::shared_ptr<LinearOperator>> ops;

  std::vector<Vec> xs; // sample vector for each level
  std::vector<Vec> rs; // residual vector for each level
  std::vector<Vec> bs; // rhs vector for each level

  std::size_t nLevels;
  std::size_t nSmooth;

  MGMCSmoothingType smoothingType;
  MGMCCoarseSamplerType coarseSamplerType;
  unsigned int cycles;
};
} // namespace parmgmc
