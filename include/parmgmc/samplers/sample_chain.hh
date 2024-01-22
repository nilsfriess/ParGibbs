#pragma once

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <petscsys.h>
#include <petscvec.h>

namespace parmgmc {
template <class Sampler, class QOI> class SampleChain {
  using DataT = typename QOI::DataT;
  static_assert(std::is_floating_point_v<DataT>,
                "DataT must be floating point type.");

public:
  template <typename... Args>
  SampleChain(QOI qoi, std::size_t n_chains, Vec ex_sample,
d              Args &&...sampler_args)
      : qoi{qoi}, curr_samples(n_chains, nullptr), samples(n_chains),
        means(n_chains, 0), square_diffs(n_chains, 0) {
    PetscFunctionBeginUser;

    samplers.reserve(n_chains);
    for (std::size_t n = 0; n < n_chains; ++n)
      samplers.emplace_back(std::forward<Args>(sampler_args)...);

    for (auto &sample : curr_samples)
      PetscCallVoid(VecDuplicate(ex_sample, &sample));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(const Vec rhs, std::size_t n_steps = 1) {
    PetscFunctionBeginUser;

    for (std::size_t c = 0; c < samplers.size(); ++c) {
      for (std::size_t n = 0; n < n_steps; ++n) {
        PetscCall(samplers[c].sample(curr_samples[c], rhs, 1));
        PetscCall(add_sample(curr_samples[c], c));
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode set_sample(Vec sample) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < samplers.size(); ++n)
      PetscCall(set_sample(sample, n));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode set_sample(Vec sample, std::size_t n_chain) {
    PetscFunctionBeginUser;

    PetscCall(VecCopy(sample, curr_samples[n_chain]));
    PetscCall(add_sample(curr_samples[n_chain], n_chain));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  DataT get_mean() const {
    DataT mean = 0;
    for (auto m : means)
      mean += 1. / means.size() * m;
    return mean;
  }

  DataT get_var() const {
    DataT var = 0;
    for (auto diff : square_diffs)
      var += 1. / (square_diffs.size() - 1.) * diff;
    return var;
  }

  DataT get_var(std::size_t n_chain) const {
    return 1. / (samples[n_chain].size() - 1) * square_diffs[n_chain];
  }

  DataT get_mean(std::size_t n_chain) const { return means[n_chain]; }

  void reset() {
    for (auto &chain : samples)
      chain.clear();
    for (auto &mean : means)
      mean = 0;
    for (auto &diff : square_diffs)
      diff = 0;
  }

  DataT gelman_rubin() const {
    if (samplers.size() < 2) {
      throw std::runtime_error(
          "Need at least 2 chains to compute Gelman-Rubin diagonostic.");
    }

    const auto mean_of_means = get_mean();
    const auto mean_of_vars = get_var();

    double var_of_means = 0;
    for (auto m : means)
      var_of_means += 1. / (samplers.size() - 1) * (m - mean_of_means) *
                      (m - mean_of_means);

    const auto n = samples[0].size();
    // return std::sqrt(((n - 1.) / n * mean_of_vars + var_of_means) /
    //                  mean_of_vars);
    return std::sqrt((n * var_of_means / mean_of_vars + n - 1) / n);
  }

  std::size_t integrated_autocorr_time(std::size_t n_chain = 0,
                                       std::size_t window_size = 30) const {
    const auto total_samples = samples[n_chain].size();
    if (window_size > total_samples)
      return total_samples;

    const auto m = get_mean();

    const auto rho = [&](std::size_t s) -> double {
      double sum = 0;
      for (std::size_t j = 1; j < total_samples - s; ++j)
        sum += (samples[n_chain][j] - m) * (samples[n_chain][j + s] - m);
      return 1. / (total_samples - s) * sum;
    };

    double sum = 0;
    const auto rho_zero = rho(0);
    for (std::size_t s = 1; s < window_size; ++s)
      sum += rho(s) / rho_zero;
    const auto tau = static_cast<std::size_t>(std::ceil(1 + 2 * sum));
    return std::max(1UL, tau);
  }

  DataT get_mean_error(std::size_t n_chain = 0) const {
    return std::sqrt(
        ((1.0 * integrated_autocorr_time(n_chain)) / samples[n_chain].size()) *
        get_var(n_chain) / get_mean(n_chain));
  }

  bool converged(DataT tol = 1.01) const {
    const auto gr = gelman_rubin();
    return gr < tol;
  }

  std::size_t get_n_chains() const { return samplers.size(); }

  ~SampleChain() {
    PetscFunctionBeginUser;
    for (auto &v : curr_samples)
      PetscCallVoid(VecDestroy(&v));
    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode add_sample(Vec sample, std::size_t chain) {
    PetscFunctionBeginUser;

    DataT q;
    PetscCall(qoi(sample, &q));
    samples[chain].push_back(q);

    const auto n_samples = samples[chain].size();

    const auto mean_before = means[chain];
    if (n_samples == 0) {
      means[chain] = q;
      square_diffs[chain] = 0;
    } else {
      means[chain] =
          (1. / n_samples) * q + (n_samples - 1.) / n_samples * means[chain];
      square_diffs[chain] += (q - mean_before) * (q - means[chain]);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  QOI qoi;

  std::vector<Vec> curr_samples;

  std::vector<Sampler> samplers;
  std::vector<std::vector<DataT>> samples;

  std::vector<DataT> means;
  std::vector<DataT> square_diffs;
};
} // namespace parmgmc
