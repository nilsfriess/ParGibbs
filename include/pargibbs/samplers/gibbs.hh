#pragma once

#include <set>
#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/mpi_helper.hh"
#include "pargibbs/samplers/sampler_statistics.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef PG_DEBUG_MODE
#include "pargibbs/common/log.hh"
#endif

namespace pargibbs {

template <class Matrix, class Engine>
class GibbsSampler : public SamplerStatistics {
public:
  GibbsSampler(Lattice *lattice, Matrix *prec, Engine *engine,
               double omega = 1.)
      : SamplerStatistics{lattice}, lattice{lattice}, prec{prec},
        engine{engine}, omega{omega} {
    if (not prec->IsRowMajor)
      throw std::runtime_error(
          "Precision matrix must be stored in row major format.");

    inv_diag.resize(prec->rows());
    rsqrt_omega_diag.resize(prec->rows());

    const auto factor = std::sqrt(omega * (2 - omega));
    for (auto v : lattice->own_vertices) {
      inv_diag.coeffRef(v) = 1. / prec->coeff(v, v);
      rsqrt_omega_diag.coeffRef(v) = factor / std::sqrt(prec->coeff(v, v));
    }

    setup_mpi_maps();

#ifdef PG_DEBUG_MODE
    if (mpi_helper::is_debug_rank()) {
      if (mpi_send.size() > 0) {
        PARGIBBS_DEBUG << "Rank " << mpi_helper::get_rank()
                       << " has to send:\n";
        for (auto &&[rank, vs] : mpi_send) {
          PARGIBBS_DEBUG << "To " << rank << ": ";
          for (auto &&idx : vs)
            PARGIBBS_DEBUG_NP << idx << " ";
          PARGIBBS_DEBUG_NP << "\n";
        }
      }
      if (mpi_recv.size() > 0) {
        PARGIBBS_DEBUG << "Rank " << mpi_helper::get_rank() << " receives:\n";
        for (auto &&[rank, vs] : mpi_recv) {
          PARGIBBS_DEBUG << "From " << rank << ": ";
          for (auto &&idx : vs)
            PARGIBBS_DEBUG_NP << idx << " ";
          PARGIBBS_DEBUG_NP << "\n";
        }
      }
    }
#endif
  }

  // TODO: Remove template, this only works with Eigen::SparseVector
  template <class Vector>
  void sample(Vector &sample, const Vector &prec_x_mean,
              std::size_t n_samples = 1) {
    assert(sample.size() == prec->rows() && sample.size() == prec->rows());

    Eigen::SparseVector<double> rand;
    rand.resize(sample.size());
    rand.reserve(sample.nonZeros());
    for (int i = 0; i < sample.nonZeros(); ++i)
      rand.insertBack(sample.innerIndexPtr()[i]);

    auto is_red_vertex = [](auto v) { return v % 2 == 0; };
    auto is_black_vertex = [](auto v) { return v % 2 != 0; };

    for (std::size_t n = 0; n < n_samples; ++n) {
      for (int i = 0; i < rand.nonZeros(); ++i)
        rand.valuePtr()[i] = dist(*engine);
      rand += prec_x_mean;

      // Update sample at "red" vertices
      sample_at_points(sample, rand, is_red_vertex);
      send_recv(sample, is_red_vertex);

      // Update sample at "black" vertices
      sample_at_points(sample, rand, is_black_vertex);
      send_recv(sample, is_black_vertex);

      if (est_mean || est_cov)
        update_statistics(sample);
    }
  }

private:
  template <class Vector, class Predicate>
  void sample_at_points(Vector &curr_sample,
                        const Eigen::SparseVector<double> &rand,
                        const Predicate &IncludeIndex) {
    using It = typename Matrix::InnerIterator;

    for (int i = 0; i < curr_sample.nonZeros(); ++i) {
      const auto row = curr_sample.innerIndexPtr()[i];
      if (halo_indices.contains(row) or (not IncludeIndex(row)))
        continue;

      double sum = 0.;
      for (It it(*prec, row); it; ++it) {
        assert(row == it.row());
        if (it.col() != it.row())
          sum += it.value() * curr_sample.coeff(it.col());
      }

      curr_sample.valuePtr()[i] =
          (1 - omega) * curr_sample.valuePtr()[i] +
          rand.valuePtr()[i] * rsqrt_omega_diag.coeff(row) -
          omega * inv_diag.coeff(row) * sum;
    }
  }

  template <class Vector, class Predicate>
  void send_recv(Vector &curr_sample, const Predicate &IncludeIndex) {
    if (mpi_helper::get_size() == 1)
      return;

    static std::vector<double> mpi_buf(lattice->border_vertices.size());

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

    for (auto v : lattice->border_vertices) {
      for (IndexT n = lattice->adj_idx.at(v); n < lattice->adj_idx.at(v + 1);
           ++n) {
        auto nb_idx = lattice->adj_vert.at(n);

        // If we have a neighbour that is owned by another MPI process, then
        // - we need to send the value at `v` to this process at some point, and
        // - we will receive values at `nb_idx` from this process at some
        //   point.
        if (lattice->mpiowner[nb_idx] != (IndexT)mpi_helper::get_rank()) {
          halo_indices.insert(nb_idx);

          mpi_send[lattice->mpiowner.at(nb_idx)].push_back(v);
          mpi_recv[lattice->mpiowner.at(nb_idx)].push_back(nb_idx);
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

  const Lattice *lattice;
  const Matrix *prec;
  Engine *engine;

  Eigen::SparseVector<double> inv_diag;
  Eigen::SparseVector<double> rsqrt_omega_diag;

  std::normal_distribution<double> dist;

  double omega; // SOR parameter

  // mpi rank -> vertex indices we need to send
  std::unordered_map<int, std::vector<int>> mpi_send;
  // mpi rank -> vertex indices we will receive
  std::unordered_map<int, std::vector<int>> mpi_recv;
  // Indices of halo vertices
  std::set<typename Lattice::IndexType> halo_indices;
};
} // namespace pargibbs
