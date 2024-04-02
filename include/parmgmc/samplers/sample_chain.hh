#pragma once

#include <cassert>
#include <cmath>
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
  static_assert(std::is_floating_point_v<DataT>, "DataT must be floating point type.");

public:
  template <typename... Args>
  SampleChain(QOI qoi, std::size_t nChains, Vec exSample, Args &&...samplerArgs)
      : qoi{qoi}, currSamples(nChains, nullptr), samples(nChains), means(nChains, 0),
        squareDiffs(nChains, 0) {
    PetscFunctionBeginUser;

    samplers.reserve(nChains);
    for (std::size_t n = 0; n < nChains; ++n) {
      samplers.emplace_back(std::forward<Args>(samplerArgs)...);

      PetscCallVoid(VecDuplicate(exSample, &currSamples[n]));
    }

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(const Vec rhs, std::size_t nSteps = 1) {
    PetscFunctionBeginUser;

    for (std::size_t c = 0; c < samplers.size(); ++c) {
      for (std::size_t n = 0; n < nSteps; ++n) {
        PetscCall(samplers[c].sample(currSamples[c], rhs, 1));
        PetscCall(addSample(currSamples[c], c));
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode setSample(Vec sample) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < samplers.size(); ++n)
      PetscCall(setSample(sample, n));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode setSample(Vec sample, std::size_t nChain) {
    PetscFunctionBeginUser;

    PetscCall(VecCopy(sample, currSamples[nChain]));
    PetscCall(addSample(currSamples[nChain], nChain));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  [[nodiscard]] DataT getMean() const {
    DataT mean = 0;
    for (auto m : means)
      mean += 1. / means.size() * m;
    return mean;
  }

  [[nodiscard]] DataT getVar() const {
    DataT var = 0;
    for (auto diff : squareDiffs)
      var += 1. / (squareDiffs.size() - 1.) * diff;
    return var;
  }

  [[nodiscard]] DataT getVar(std::size_t nChain) const {
    return 1. / (samples[nChain].size() - 1) * squareDiffs[nChain];
  }

  [[nodiscard]] DataT getMean(std::size_t nChain) const { return means[nChain]; }

  void reset() {
    for (auto &chain : samples)
      chain.clear();
    for (auto &mean : means)
      mean = 0;
    for (auto &diff : squareDiffs)
      diff = 0;
  }

  [[nodiscard]] DataT gelmanRubin() const {
    if (samplers.size() < 2) {
      throw std::runtime_error("Need at least 2 chains to compute Gelman-Rubin diagonostic.");
    }

    const auto meanOfMeans = getMean();
    const auto meanOfVars = getVar();

    double varOfMeans = 0;
    for (auto m : means)
      varOfMeans += 1. / (samplers.size() - 1) * (m - meanOfMeans) * (m - meanOfMeans);

    const auto n = samples[0].size();
    // return std::sqrt(((n - 1.) / n * mean_of_vars + var_of_means) /
    //                  mean_of_vars);
    return std::sqrt((n * varOfMeans / meanOfVars + n - 1) / n);
  }

  [[nodiscard]] std::size_t integratedAutocorrTime(std::size_t windowSize = 30) const {
    double d = 0;
    for (std::size_t i = 0; i < getNChains(); ++i)
      d += 1. / getNChains() * integratedAutocorrTime(i, windowSize);

    return std::round(d);
  }

  [[nodiscard]] std::size_t integratedAutocorrTime(std::size_t nChain,
                                                   std::size_t windowSize) const {
    const auto totalSamples = samples[nChain].size();
    if (windowSize > totalSamples)
      return totalSamples;

    const auto m = getMean(nChain);

    const auto rho = [&](std::size_t s) -> double {
      double sum = 0;
      for (std::size_t j = 0; j < totalSamples - s; ++j)
        sum += 1. / (totalSamples - s) * (samples[nChain][j] - m) * (samples[nChain][j + s] - m);
      return sum;
    };

    double sum = 0;
    const auto rhoZero = rho(0);
    for (std::size_t s = 1; s < windowSize; ++s)
      sum += rho(s) / rhoZero;
    const auto tau = static_cast<std::size_t>(std::ceil(1 + 2 * sum));
    return std::max(1UL, tau);
  }

  [[nodiscard]] DataT getMeanError(std::size_t nChain = 0) const {
    return std::sqrt(((1.0 * integratedAutocorrTime(nChain)) / samples[nChain].size()) *
                     getVar(nChain) / getMean(nChain));
  }

  [[nodiscard]] bool converged(DataT tol = 1.01) const {
    const auto gr = gelmanRubin();
    return gr < tol;
  }

  [[nodiscard]] std::size_t getNChains() const { return samplers.size(); }

  [[nodiscard]] const Sampler &getSampler(std::size_t nChain = 0) const { return samplers[nChain]; }

  Sampler &getSampler(std::size_t nChain = 0) { return samplers[nChain]; }

  ~SampleChain() {
    PetscFunctionBeginUser;
    for (auto &v : currSamples)
      PetscCallVoid(VecDestroy(&v));
    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode addSample(Vec sample, std::size_t chain) {
    PetscFunctionBeginUser;

    DataT q;
    PetscCall(qoi(sample, &q));
    samples[chain].push_back(q);

    const auto nSamples = samples[chain].size();

    const auto meanBefore = means[chain];
    if (nSamples == 0) {
      means[chain] = q;
      squareDiffs[chain] = 0;
    } else {
      means[chain] = (1. / nSamples) * q + (nSamples - 1.) / nSamples * means[chain];
      squareDiffs[chain] += (q - meanBefore) * (q - means[chain]);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  QOI qoi;

  std::vector<Vec> currSamples;

  std::vector<Sampler> samplers;
  std::vector<std::vector<DataT>> samples;

  std::vector<DataT> means;
  std::vector<DataT> squareDiffs;
};
} // namespace parmgmc
