#pragma once

#include "pargibbs/common/helpers.hh"
#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/samplers/gibbs.hh"
#include "pargibbs/samplers/sampler_statistics.hh"

#include <cstddef>
#include <memory>

namespace pargibbs {
template <class Operator, class Engine,
          class Smoother = GibbsSampler<Operator, Engine>>
class MultigridSampler : public SamplerStatistics<Operator> {
public:
  struct Parameters {
    std::size_t levels;
    std::size_t cycles;
    std::size_t n_presample;
    std::size_t n_postsample;
    double prepost_sampler_omega = 1.;
  };

  MultigridSampler(std::shared_ptr<Operator> finest_operator, Engine *engine,
                   const Parameters &params)
      : SamplerStatistics<Operator>{finest_operator}, engine{engine},
        levels{params.levels}, cycles{params.cycles},
        n_presample{params.n_presample}, n_postsample{params.n_postsample} {
    assert(levels >= 2 && "Multigrid sampler must at least have two levels");

    operators.push_back(finest_operator);
    pre_smoothers.emplace_back(finest_operator, engine);
    post_smoothers.emplace_back(finest_operator, engine);

    for (std::size_t l = 1; l < levels; ++l) {
      // fine_operator.coarsen() takes Functor that is supposed to return the
      // coarsened matrix given the coarsened_lattice and the fine level matrix.
      auto coarse_operator = operators[l - 1]->coarsen(
          [&](const auto &coarse_lattice, const auto &fine_matrix) {
            prolongations.push_back(make_prolongation(
                operators[l - 1]->get_lattice(), coarse_lattice));

            return prolongations[l - 1].transpose() * fine_matrix *
                   prolongations[l - 1];
          });
      operators.push_back(coarse_operator);

      pre_smoothers.emplace_back(operators[l], engine, params.prepost_sampler_omega);
      post_smoothers.emplace_back(operators[l], engine, params.prepost_sampler_omega);
    }

    for (std::size_t l = 0; l < levels; ++l) {
      current_samples.emplace_back(operators[l]->size());
      for_each_ownindex_and_halo(operators[l]->get_lattice(), [&](auto idx) {
        current_samples[l].coeffRef(idx) = 0;
      });
    }
  }

  void sample(typename Operator::Vector &sample, std::size_t n_samples = 1) {
    current_samples[0] = sample;

    for (std::size_t n = 0; n < n_samples; ++n) {
      sample_impl(0, operators[0]->vector());

      this->update_statistics(current_samples[0]);
    }

    sample = current_samples[0];
  }

private:
  void sample_impl(std::size_t level, const typename Operator::Vector &nu) {
    operators[level]->vector() = nu;
    pre_smoothers[level].sample(current_samples[level], n_presample);

    if (level < levels - 1) {

      // Compute residual
      typename Operator::Vector resid =
          prolongations[level].transpose() *
          (nu - operators[level]->get_matrix() * current_samples[level]);

      current_samples[level + 1].setZero();

      for (std::size_t cycle = 0; cycle < cycles; ++cycle)
        sample_impl(level + 1, resid);

      // Update fine sample with coarse correction
      current_samples[level] +=
          prolongations[level] * current_samples[level + 1];
    }

    post_smoothers[level].sample(current_samples[level], n_postsample);
  }

  Engine *engine;

  std::vector<std::shared_ptr<Operator>> operators;

  std::vector<Smoother> pre_smoothers;
  std::vector<Smoother> post_smoothers;

  std::vector<Eigen::SparseMatrix<double>> prolongations;

  std::vector<typename Operator::Vector> current_samples;

  std::size_t levels;
  std::size_t cycles;
  std::size_t n_presample;
  std::size_t n_postsample;
};
} // namespace pargibbs
