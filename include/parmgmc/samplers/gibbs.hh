#pragma once

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/lattice_operator.hh"
#include "parmgmc/common/log.hh"
#include "parmgmc/common/traits.hh"
#include "parmgmc/lattice/lattice.hh"
#include "parmgmc/mpi_helper.hh"
#include "parmgmc/samplers/sampler_statistics.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace parmgmc {

template <class Operator, class Engine>
class GibbsSampler : public SamplerStatistics<Operator> {
public:
  GibbsSampler(std::shared_ptr<Operator> lattice_operator, Engine *engine,
               double omega = 1.)
      : SamplerStatistics<Operator>{lattice_operator}, op{lattice_operator},
        engine{engine}, omega{omega} {
    if (not op->get_matrix().IsRowMajor)
      throw std::runtime_error(
          "Precision matrix must be stored in row major format.");

    inv_diag.resize(op->get_matrix().rows());
    rsqrt_omega_diag.resize(op->get_matrix().rows());

    const auto factor = std::sqrt(omega * (2 - omega));

    for (auto v : op->get_lattice().vertices()) {
      inv_diag[v] = 1. / op->get_matrix().coeff(v, v);
      rsqrt_omega_diag[v] = factor / std::sqrt(op->get_matrix().coeff(v, v));
    }

    rand.resize(op->get_matrix().rows());

    setup_mpi_maps();

#if PARMGMC_DEBUG_LEVEL == PARMGMC_DEBUG_LEVEL_VERBOSE
    if (mpi_helper::is_debug_rank()) {
      if (mpi_send.size() > 0) {
        PARMGMC_DEBUG << "Rank " << mpi_helper::get_rank() << " has to send:\n";
        for (auto &&[rank, vs] : mpi_send) {
          PARMGMC_DEBUG << "To " << rank << ": ";
          for (auto &&idx : vs)
            PARMGMC_DEBUG_NP << idx << " ";
          PARMGMC_DEBUG_NP << "\n";
        }
      }
      if (mpi_recv.size() > 0) {
        PARMGMC_DEBUG << "Rank " << mpi_helper::get_rank() << " receives:\n";
        for (auto &&[rank, vs] : mpi_recv) {
          PARMGMC_DEBUG << "From " << rank << ": ";
          for (auto &&idx : vs)
            PARMGMC_DEBUG_NP << idx << " ";
          PARMGMC_DEBUG_NP << "\n";
        }
      }
    }
#endif
  }

  void sample(typename Operator::Vector &sample, std::size_t n_samples = 1) {
    auto is_red_vertex = [&](auto v) { return v % 2 == 0; };
    auto is_black_vertex = [&](auto v) { return !is_red_vertex(v); };

    for (std::size_t n = 0; n < n_samples; ++n) {
      if constexpr (detail::is_eigen_sparse_vector_v<
                        typename Operator::Vector>) {
        for (int i = 0; i < rand.nonZeros(); ++i)
          rand.valuePtr()[i] = dist(*engine);
      } else {
        for (auto v : op->get_lattice().vertices())
          rand[v] = dist(*engine);
      }

      rand += op->vector();

      // Update sample at "red" vertices
      sample_at_points(sample, is_red_vertex);
      send_recv(sample, is_red_vertex);

      // Update sample at "black" vertices
      sample_at_points(sample, is_black_vertex);
      send_recv(sample, is_black_vertex);

      this->update_statistics(sample);
    }
  }

private:
  template <class Predicate>
  void sample_at_points(typename Operator::Vector &curr_sample,
                        Predicate &&IncludeIndex) {
    assert(curr_sample.size() == op->get_matrix().rows());

    using It = typename Operator::Matrix::InnerIterator;

    for (auto row : op->get_lattice().vertices()) {
      if (not IncludeIndex(row))
        continue;

      double sum = 0.;
      // Loop over non-zero entries in current row
      for (It it(op->get_matrix(), row); it; ++it) {
        assert(row == it.row());
        sum += (it.col() != row) * it.value() * curr_sample.coeff(it.col());
      }

      curr_sample.coeffRef(row) = (1 - omega) * curr_sample.coeff(row) +
                                  rand.coeff(row) * rsqrt_omega_diag[row] -
                                  omega * inv_diag[row] * sum;
    }
  }

  template <class Predicate>
  void send_recv(typename Operator::Vector &curr_sample,
                 const Predicate &IncludeIndex) {
    if (mpi_helper::get_size() == 1)
      return;

    static std::vector<double> mpi_buf(
        op->get_lattice().get_n_border_vertices());

    for (auto &&[target, vs] : mpi_send) {
      for (std::size_t i = 0; i < vs.size(); ++i)
        mpi_buf.at(i) = curr_sample.coeff(vs.at(i));

      MPI_Send(
          mpi_buf.data(), vs.size(), MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
    }

    for (auto &&[source, vs] : mpi_recv) {
      MPI_Recv(mpi_buf.data(),
               vs.size(),
               MPI_DOUBLE,
               source,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      for (std::size_t i = 0; i < vs.size(); ++i)
        if (IncludeIndex(vs.at(i)))
          curr_sample.coeffRef(vs.at(i)) = mpi_buf.at(i);
    }
  }

  void setup_mpi_maps() {
    using IndexT = typename Lattice::IndexType;

    const auto [adj_idx, adj_vert] = op->get_lattice().get_adjacency_lists();
    for (auto v : op->get_lattice().vertices(VertexType::Border)) {
      for (IndexT n = adj_idx.at(v); n < adj_idx.at(v + 1); ++n) {
        auto nb_idx = adj_vert.at(n);

        // If we have a neighbour that is owned by another MPI process, then
        // - we need to send the value at `v` to this process at some point, and
        // - we will receive values at `nb_idx` from this process at some
        //   point.
        if (op->get_lattice().mpiowner[nb_idx] !=
            (IndexT)mpi_helper::get_rank()) {
          halo_indices.insert(nb_idx);

          mpi_send[op->get_lattice().mpiowner.at(nb_idx)].push_back(v);
          mpi_recv[op->get_lattice().mpiowner.at(nb_idx)].push_back(nb_idx);
        }
      }
    }

    // After we are done setting up the maps, we sort the list of indices
    // because it is not guaranteed that the list of indices that rank x has to
    // send to rank y is ordered the same as the list of indices that rank y
    // expects to receive from rank x. We also remove duplicates here, since we
    // don't need to send the same value twice.
    for (auto &[rank, indices] : mpi_send) {
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }

    for (auto &[rank, indices] : mpi_recv) {
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }
  }

  std::shared_ptr<Operator> op;
  Engine *engine;

  typename Operator::Vector rand;
  Eigen::VectorXd inv_diag;
  Eigen::VectorXd rsqrt_omega_diag;

  std::normal_distribution<double> dist;

  double omega; // SOR parameter

  // mpi rank -> vertex indices we need to send
  std::unordered_map<int, std::vector<int>> mpi_send;
  // mpi rank -> vertex indices we will receive
  std::unordered_map<int, std::vector<int>> mpi_recv;
  // Indices of halo vertices
  std::set<typename Lattice::IndexType> halo_indices;
};
} // namespace parmgmc
