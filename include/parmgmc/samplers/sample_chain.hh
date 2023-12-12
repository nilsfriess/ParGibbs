#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <petscsys.h>
#include <petscvec.h>

namespace parmgmc {
template <class Sampler> class SampleChain {
public:
  template <typename... Args>
  SampleChain(std::function<PetscErrorCode(Vec, PetscReal *)> qoi,
              Args &&...sampler_args)
      : sampler(std::forward<Args>(sampler_args)...), qoi{qoi}, n_samples{0},
        save_samples{false}, est_mean_online{true} {}

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_steps = 1) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < n_steps; ++n) {
      n_samples++;

      PetscCall(sampler.sample(sample, rhs));

      if (save_samples || est_mean_online) {
        PetscReal q;
        PetscCall(qoi(sample, &q));

        if (save_samples)
          samples.push_back(q);

        if (est_mean_online)
          update_mean(q);
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscReal get_mean() {
    if (est_mean_online)
      return mean_;
    else
      return compute_mean();
  }

  void update_mean(PetscReal q) {
    if (n_samples == 1)
      mean_ = q;
    else
      mean_ = (1. / n_samples) * q + (n_samples - 1.) / n_samples * mean_;
  }

  void enable_save_samples() { save_samples = true; }
  void disable_save_samples() { save_samples = false; }

  void enable_est_mean_online() { est_mean_online = true; }
  void disable_est_mean_online() { est_mean_online = true; }

  void reset() {
    samples.clear();
    n_samples = 0;
  }

private:
  PetscReal compute_mean() const {
    assert(samples.size() >= 1);

    if (!save_samples)
      throw std::runtime_error("[SampleChain::compute_mean] Cannot compute "
                               "mean when save_samples is not enabled");

    PetscReal m = 0;
    const auto n_samples = samples.size();
    for (auto sample : samples)
      m += 1. / n_samples * sample;

    return m;
  }

  Sampler sampler;
  std::function<PetscErrorCode(Vec, PetscReal *)> qoi;

  PetscReal mean_; // Online estimated mean

  std::vector<PetscReal> samples;
  std::size_t n_samples;

  bool save_samples;
  bool est_mean_online;
};
} // namespace parmgmc
