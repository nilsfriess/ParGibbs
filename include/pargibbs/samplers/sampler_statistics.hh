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
    reset_statistics();
  }

  void enable_estimate_covariance() {
    if (mpi_helper::get_size() > 1)
      throw std::runtime_error("Estimating covariance matrix is currently only "
                               "supported for sequential applications");

    enable_estimate_mean(); // Estimating cov requires estimating mean

    est_cov = true;
    reset_statistics();
  }

  const Eigen::VectorXd get_mean() const {
    if (not est_mean)
      throw std::runtime_error("Tried to get_mean but est_mean is false. Call "
                               "enable_estimate_mean() before sampling.");

    Eigen::VectorXd ret_mean(mean.size());
    ret_mean.setZero();
    // Remove halo vertices
    for (auto v : lattice->own_vertices)
      ret_mean[v] = mean.coeff(v);

    return ret_mean;
  }

  const Eigen::MatrixXd &get_covariance() const { return cov; }

  void reset_statistics() {
    n_sample = 1;
    if (est_mean) {
      mean.resize(lattice->get_n_total_vertices());
      mean.setZero();
    }

    if (est_cov) {
      cov.resize(lattice->get_n_total_vertices(),
                 lattice->get_n_total_vertices());
      cov.setZero();
    }
  }

protected:
  SamplerStatistics(const Lattice *lattice)
      : est_mean{false}, est_cov{false}, lattice{lattice} {}

  bool est_mean; // true if mean should be estimated during sampling
  bool est_cov; // true if covariance matrix should be estimated during sampling

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

private:
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;

  std::size_t n_sample = 0;

  const Lattice *lattice;
};
} // namespace pargibbs
