#include "gmrf_operator.hh"

#include "pargibbs/mpi_helper.hh"
#include "pargibbs/samplers/gibbs.hh"
#include "pargibbs/samplers/multigrid.hh"

#include <cmath>
#include <iostream>

#include <pcg_random.hpp>
#include <random>

namespace pg = pargibbs;

struct GR_Result {
  bool converged = false;
  double R = 0;
  std::size_t iterations = 0;
};

template <class SamplerFactory>
GR_Result gelman_rubin_test(SamplerFactory &&factory, std::size_t n_burnin,
                            std::size_t n_chains, std::size_t vector_size,
                            double R_tol, std::size_t max_its = 100'000) {
  using Vector = Eigen::VectorXd;

  // Create samplers using provided factory function
  std::vector<decltype(factory())> samplers;
  samplers.reserve(n_chains);
  for (std::size_t chain = 0; chain < n_chains; ++chain)
    samplers.emplace_back(factory());

  // Helper variables to store current sample for each chain
  std::vector<Vector> samples(n_chains);
  std::for_each(samples.begin(), samples.end(), [&](auto &sample) {
    sample.resize(vector_size);
    sample.setZero();
  });

  // Perform burnin
  for (std::size_t chain = 0; chain < n_chains; ++chain)
    samplers[chain].sample(samples[chain], n_burnin);

  // Create array of sample norms for each chain
  std::vector<std::vector<double>> sample_norms(n_chains);
  std::size_t it;
  for (it = 0; it < max_its; ++it) {
    // Start sampling
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
      samplers[chain].sample(samples[chain], 1);
      sample_norms[chain].push_back(samples[chain].norm());
    }

    /// Compute Gelman-Rubin diagnostic
    auto n_samples = it + 1;

    // Compute intra chain means
    std::vector<double> intra_means(n_chains);
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
      double mean = 0;
      for (std::size_t n = 0; n < n_samples - 1; ++n) {
        mean += 1. / n_samples * sample_norms[chain][n];
      }
      intra_means[chain] = mean;
    }

    // Compute intra chain variances
    std::vector<double> intra_vars(n_chains);
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
      double var = 0;
      for (std::size_t n = 0; n < n_samples - 1; ++n) {
        var += 1. / (n_samples - 1) *
               (sample_norms[chain][n] - intra_means[chain]) *
               (sample_norms[chain][n] - intra_means[chain]);
      }
      intra_vars[chain] = var;
    }

    // Calculate inter chain mean
    double inter_mean =
        1. / n_chains *
        std::accumulate(intra_means.begin(), intra_means.end(), 0);

    // Compute scattering of means around joint mean
    double mean_scatter = 0;
    for (std::size_t chain = 0; chain < n_chains; ++chain) {
      mean_scatter += n_samples / (n_chains - 1.) *
                      (inter_mean - intra_means[chain]) *
                      (inter_mean - intra_means[chain]);
    }

    // Compute averaged variances
    double avg_variances =
        1. / n_chains *
        std::accumulate(intra_vars.begin(), intra_vars.end(), 0);

    // Compute Gelman Rubin variance
    auto V = (n_samples - 1.) / n_samples * avg_variances +
             (n_chains + 1.) / (n_chains * n_samples) * mean_scatter;

    double R = std::sqrt(V / avg_variances);

    if (std::abs(R - 1) < R_tol) {
      GR_Result res;
      res.converged = true;
      res.iterations = it;
      res.R = R;
      return res;
    }
  }

  GR_Result res;
  res.converged = false;
  return res;
}

int main(int argc, char *argv[]) {
  pg::mpi_helper mh(&argc, &argv);

  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Operator = pg::LatticeOperator<Matrix, Vector>;

  auto op = std::make_shared<Operator>(2, 9, gmrf_matrix_builder);

  pcg32 engine{std::random_device{}()};

  const std::size_t n_chains = 100;
  const std::size_t n_burnin = 100;

  {
    using Sampler = pg::MultigridSampler<Operator, pcg32>;

    auto mg_sampler_factory = [&]() {
      return Sampler(
          op,
          &engine,
          Sampler::Parameters{
              .levels = 3, .cycles = 2, .n_presample = 2, .n_postsample = 2});
    };

    std::cout << "Performing Gelman-Rubin test for Multigrid sampler..."
              << std::flush;
    auto res = gelman_rubin_test(
        mg_sampler_factory, n_burnin, n_chains, op->size(), 0.01);
    std::cout << " Done.\n";

    if (res.converged)
      std::cout << "Converged with R = " << res.R << " in " << res.iterations
                << " iterations.\n";
    else
      std::cout << "Did no converge.\n";
  }

  {
    using Sampler = pg::GibbsSampler<Operator, pcg32>;

    auto mg_sampler_factory = [&]() { return Sampler(op, &engine, 1.9852); };

    std::cout << "Performing Gelman-Rubin test for Gibbs sampler..."
              << std::flush;
    auto res = gelman_rubin_test(
        mg_sampler_factory, n_burnin, n_chains, op->size(), 0.01);
    std::cout << " Done.\n";

    if (res.converged)
      std::cout << "Converged with R = " << res.R << " in " << res.iterations
                << " iterations.\n";
    else
      std::cout << "Did no converge.\n";
  }
}
