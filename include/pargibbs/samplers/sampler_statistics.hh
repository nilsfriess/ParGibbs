#pragma once

#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/mpi_helper.hh"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace pargibbs {
class SamplerStatistics {
public:
  void enable_estimate_mean() {
    est_mean = true;
    mean.resize(lattice->get_n_total_vertices());
    mean.setZero();
  }

  void enable_estimate_covariance() {
    if (mpi_helper::get_size() > 1)
      throw std::runtime_error("Estimating covariance matrix is currently only "
                               "supported for sequential applications");

    enable_estimate_mean(); // Estimating cov requires estimating mean

    est_cov = true;
    cov.resize(lattice->get_n_total_vertices(),
               lattice->get_n_total_vertices());
    cov.setZero();
  }

  Eigen::SparseVector<double> get_mean() const {
    Eigen::SparseVector<double> local_mean(lattice->get_n_total_vertices());
    for (auto v : lattice->own_vertices)
      // Remove halo vertices
      local_mean.insert(v) = mean.coeff(v);

    return local_mean;
  }

  const Eigen::MatrixXd &get_covariance() const { return cov; }

  void reset_statistics() {
    n_sample = 0;
    if (est_mean)
      mean.setZero();
    if (est_cov)
      cov.setZero();
  }

protected:
  SamplerStatistics(const Lattice *lattice) : lattice{lattice} {}

  bool est_mean; // true if mean should be estimated during sampling
  bool est_cov; // true if covariance matrix should be estimated during sampling

private:
  void update_statistics(const auto &sample) {
    // Update mean
    if (est_mean) {
      if (n_sample == 1) {
        mean = sample;
      } else {
        mean += 1 / (1. + n_sample) * (sample - mean);
      }
    }

    // Update covariance matrix
    if (est_cov) {
      if (n_sample >= 2) {
        cov *= n_sample / (1. + n_sample);
        cov += n_sample / ((1. + n_sample) * (1. + n_sample)) *
               (sample - mean) * (sample - mean).transpose();
      }
    }

    n_sample++;
  }

  Eigen::SparseVector<double> mean;
  Eigen::MatrixXd cov;

  std::size_t n_sample = 0;

  const Lattice *lattice;
};
} // namespace pargibbs
