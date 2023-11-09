#pragma once

#include "pargibbs/common/helpers.hh"
#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/samplers/gibbs.hh"
#include "pargibbs/samplers/sampler_statistics.hh"

#include <memory>

namespace pargibbs {
template <class Matrix, class Engine>
class MultigridSampler : public SamplerStatistics {
  using Smoother = GibbsSampler<Matrix, Engine>;

public:
  // TODO: Think again about what should be passed to the constructor, maybe
  // some factory functionals instead?
  MultigridSampler(std::shared_ptr<Lattice> finest_lattice,
                   std::shared_ptr<Matrix> prec, Engine *engine,
                   std::size_t levels, std::size_t cycles = 1)
      : SamplerStatistics{finest_lattice.get()}, engine{engine}, levels{levels},
        cycles{cycles} {
    // Create hierachy of lattices
    lattices.resize(levels);
    lattices[0] = finest_lattice;
    for (std::size_t l = 1; l < levels; ++l) {
      lattices[l] = std::make_shared<Lattice>(lattices[l - 1]->coarsen());
    }

    for (std::size_t l = 1; l < levels; ++l) {
      prolongations.push_back(
          make_prolongation(*lattices[l - 1], *lattices[l]));

      restrictions.push_back(make_restriction(*lattices[l - 1], *lattices[l]));
    }

    // Next, create hierachy of operators
    operators.resize(levels);
    operators[0] = prec;
    for (std::size_t l = 1; l < levels; ++l) {
      const auto restr = make_restriction(*lattices[l - 1], *lattices[l]);
      operators[l] = std::make_shared<Matrix>(restr * (*operators[l - 1]) *
                                              prolongations[l - 1]);
    }

    for (std::size_t l = 0; l < levels; ++l) {
      pre_smoothers.emplace_back(lattices[l], operators[l], engine, 1.9852);
      post_smoothers.emplace_back(lattices[l], operators[l], engine, 1.9852);
    }

    for (std::size_t l = 0; l < levels; ++l) {
      current_samples.emplace_back(lattices[l]->get_n_total_vertices());
      pargibbs::for_each_ownindex_and_halo(
          *lattices[l], [&](auto idx) { current_samples[l].insert(idx) = 0; });
    }
  }

  template <class Vector>
  void sample(Vector &sample, const Vector &nu, std::size_t n_samples = 1) {
    current_samples[0] = sample;

    for (std::size_t n = 0; n < n_samples; ++n) {
      sample_impl(0, nu);

      if (est_mean || est_cov)
        update_statistics(current_samples[0]);
    }

    sample = current_samples[0];
  }

private:
  template <class Vector>
  void sample_impl(std::size_t level, const Vector &nu) {
    for (std::size_t k = 0; k < 2; ++k)
      pre_smoothers[level].sample(current_samples[level], nu);

    if (level < levels - 1) {

      // Compute residual
      Vector resid = restrictions[level] *
                     (nu - *operators[level] * current_samples[level]);

      current_samples[level + 1].setZero();

      for (std::size_t cycle = 0; cycle < cycles; ++cycle)
        sample_impl(level + 1, resid);

      // Update fine sample with coarse correction
      current_samples[level] +=
          prolongations[level] * current_samples[level + 1];
    }

    for (std::size_t k = 0; k < 2; ++k)
      post_smoothers[level].sample(current_samples[level], nu);
  }

  Matrix *prec;
  Engine *engine;

  std::vector<std::shared_ptr<Lattice>> lattices;
  std::vector<std::shared_ptr<Matrix>> operators;

  std::vector<Smoother> pre_smoothers;
  std::vector<Smoother> post_smoothers;

  std::vector<Eigen::SparseMatrix<double>> prolongations;
  std::vector<Eigen::SparseMatrix<double>> restrictions;

  std::vector<Eigen::SparseVector<double>> current_samples;

  std::size_t levels;

  std::size_t cycles;
};
} // namespace pargibbs
