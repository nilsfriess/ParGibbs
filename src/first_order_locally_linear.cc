#include "pargibbs/gibbs.hh"

#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <random>
#include <vector>

// Example 6.1 from [Fox, Parker: Accelerated Gibbs sampling of normal
// distributions using matrix splittings and polynomials,
// https://arxiv.org/abs/1505.03512]

using namespace pargibbs;

constexpr int lat_size = 4;
constexpr int size = lat_size * lat_size;

using Vector = Eigen::Vector<double, size>;
using Triplet = Eigen::Triplet<double>;
using SparseMatrix = Eigen::SparseMatrix<double>;
using Matrix = Eigen::Matrix<double, size, size>;

template <class Container> Vector compute_mean(const Container &values) {
  return (1. / values.size()) *
         std::accumulate(values.begin(), values.end(), Vector(0));
}

template <class Container> Matrix compute_cov(const Container &values) {
  const auto mean = compute_mean(values);

  // clang-format off
  auto cov = std::transform_reduce(values.begin(), values.end(),
                                   Matrix{Matrix::Zero()},
                                   std::plus<Matrix>(),
                                   [&](const auto &x) {
                                     return (x - mean) * (x - mean).transpose();
                                   });
  cov *= (1. / (values.size() - 1));
  // clang-format on
  return cov;
}

void update_mean_and_cov(const Vector &new_sample, int n_samples_before,
                         Vector &mean, Matrix &cov) {
  mean *= n_samples_before;
  mean += new_sample;
  mean /= (n_samples_before + 1);

  cov *= n_samples_before - 1;
  cov += (new_sample - mean) * (new_sample - mean).transpose();
  cov /= n_samples_before;
}

int main() {
  constexpr double noise_var = 1e-4;

  const auto get_neighbors = [&](int i) -> std::vector<int> {
    std::vector<int> neighbors;

    // If not at left border of lattice
    if (i % lat_size != 0)
      neighbors.push_back(i - 1);

    // If not at right border of lattice
    if (i % lat_size != lat_size - 1)
      neighbors.push_back(i + 1);

    // If not at lower border of lattice
    if (not(i >= 0 and i <= lat_size - 1))
      neighbors.push_back(i - lat_size);

    // If not at upper border of lattice
    if (not(i >= (lat_size - 1) * lat_size and i <= lat_size * lat_size - 1))
      neighbors.push_back(i + lat_size);

    return neighbors;
  };

  constexpr int nnz = 460;
  std::vector<Triplet> triplets(nnz);
  for (int row = 0; row < size; ++row) {
    const auto neighbors = get_neighbors(row);
    const auto n_neighbors = neighbors.size();

    triplets.emplace_back(row, row, noise_var + n_neighbors);

    for (const auto col : neighbors)
      triplets.emplace_back(row, col, -1);
  }

  SparseMatrix prec(size, size);
  prec.insertFromTriplets(triplets.begin(), triplets.end());

  std::cout << "nnz = " << prec.nonZeros() << "\n";
  std::cout << "||A|| = " << Eigen::MatrixXd(prec).operatorNorm() << "\n";

  std::random_device rd;
  std::mt19937_64 engine(rd());

  // Compute cov to estimate error
  Eigen::SimplicialLLT<SparseMatrix> solver;
  solver.compute(prec);

  SparseMatrix eye(size, size);
  eye.setIdentity();
  Matrix cov = solver.solve(eye);

  auto cov_norm = cov.operatorNorm();
  std::cout << "||A^-1|| = " << cov_norm << "\n";
  auto normal_cov = cov / cov_norm;

  int n_chains = 10000;

  using Sampler = GibbsSampler<SparseMatrix, std::mt19937_64>;
  std::vector<Sampler> samplers(n_chains, Sampler(prec, engine));

  std::vector<Vector> current_samples(n_chains, Vector::Zero());

  int n_iterations = 200;
  std::vector<double> errors(n_iterations, 0);

  for (int iteration = 0; iteration < n_iterations; ++iteration) {
    for (int chain = 0; chain < n_chains; ++chain) {
      current_samples[chain] = samplers[chain].sample(current_samples[chain]);
    }

    auto sample_cov = compute_cov(current_samples);
    errors[iteration] = (sample_cov / cov_norm - normal_cov).norm();

    std::cout << errors[iteration] << "\n";
  }

  // std::vector<double> errors;
  // samples.reserve(n_samples);

  // samples.push_back(Vector::Zero());
  // samples.push_back(sampler.sample(samples[0]));

  // auto sample_mean = compute_mean(samples);
  // auto sample_cov = compute_cov(samples);

  // for (int i = 2; i < n_samples; ++i) {
  //   samples.push_back(sampler.sample(samples[i - 1]));

  //   update_mean_and_cov(samples[i], i, sample_mean, sample_cov);

  //   errors.push_back((sample_cov / cov_norm - normal_cov).norm());
  // }

  //    std::cout << error << "\n";
}
