#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef PG_DEBUG_MODE
#include "pargibbs/common/log.hh"
#endif

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

#include "../mpi_helper.hh"
#include "pargibbs/lattice/lattice.hh"

namespace pargibbs {

template <class Matrix, class Engine> class GibbsSampler {
public:
  GibbsSampler(Lattice *lattice, Matrix *prec, Engine *engine,
               bool est_mean = false, double omega = 1.)
      : lattice(lattice), prec(prec), engine(engine), omega(omega),
        est_mean(est_mean) {

    inv_diag.resize(prec->rows());
    rsqrt_diag.resize(prec->rows());

    for (auto v : lattice->own_vertices) {
      inv_diag.coeffRef(v) = 1. / prec->coeff(v, v);
      rsqrt_diag.coeffRef(v) = 1. / std::sqrt(prec->coeff(v, v));
    }

    if (est_mean)
      mean.resize(prec->rows());

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

  Eigen::SparseVector<double> sample(const Eigen::SparseVector<double> &initial,
                                     std::size_t n_samples) {
    Eigen::VectorXd rand;
    rand.resize(initial.size());

    Eigen::SparseVector<double> next(initial);

    auto is_red_vertex = [](auto v) { return v % 2 == 0; };
    auto is_black_vertex = [](auto v) { return v % 2 != 0; };

    for (std::size_t n = 0; n < n_samples; ++n) {
      std::generate(rand.begin(), rand.end(), [&]() { return dist(*engine); });

      // Update sample at "red" vertices
      sample_at_points(next, rand, is_red_vertex);
      send_recv(next, is_red_vertex);

      // Update sample at "black" vertices
      sample_at_points(next, rand, is_black_vertex);
      send_recv(next, is_black_vertex);

      if (est_mean)
        update_mean(next);
    }

    return next;
  }

  Eigen::SparseVector<double> get_mean() const {
    Eigen::SparseVector<double> local_mean(lattice->get_n_total_vertices());
    for (auto v : lattice->own_vertices)
      local_mean.insert(v) = mean.coeff(v);

    return local_mean;
  }

  void reset_mean() {
    n_sample = 0;
    mean.setZero();
  }

private:
  template <class Predicate>
  void sample_at_points(Eigen::SparseVector<double> &curr_sample,
                        const Eigen::VectorXd &rand,
                        const Predicate &IncludeIndex) {
    using It = typename Matrix::InnerIterator;

    for (auto v : lattice->own_vertices) {
      if (not IncludeIndex(v))
        continue;

      double sum = 0.;
      for (It it(*prec, v); it; ++it) {
        if (it.col() != it.row()) {
          // If the matrix is in row-major order, `it` iterates over the columns
          // of the row `v` and vice-versa for column-major order
          if constexpr (prec->IsRowMajor)
            sum += it.value() * curr_sample.coeff(it.col());
          else
            sum += it.value() * curr_sample.coeff(it.row());
        }
      }

      curr_sample.coeffRef(v) =
          (1 - omega) * curr_sample.coeff(v) +
          rand[v] * std::sqrt(omega * (2 - omega)) * rsqrt_diag.coeff(v) -
          omega * inv_diag.coeff(v) * sum;
    }
  }

  template <class Predicate>
  void send_recv(Eigen::SparseVector<double> &curr_sample,
                 const Predicate &IncludeIndex) {
    static std::vector<double> mpi_buf(lattice->border_vertices.size());
    std::vector<MPI_Request> reqs;
    reqs.reserve(4);

    for (auto &&[target, vs] : mpi_send) {
      for (std::size_t i = 0; i < vs.size(); ++i)
        mpi_buf[i] = curr_sample.coeff(vs[i]);

      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Isend(mpi_buf.data(), vs.size(), MPI_DOUBLE, target, 0,
                MPI_COMM_WORLD, &reqs.back());
    }
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    for (auto &&[source, vs] : mpi_recv) {
      MPI_Recv(mpi_buf.data(), vs.size(), MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      for (std::size_t i = 0; i < vs.size(); ++i)
        if (IncludeIndex(vs[i]))
          curr_sample.coeffRef(vs[i]) = mpi_buf[i];
    }
  }

  void update_mean(const auto &sample) {
    if (n_sample == 1) {
      mean = sample;
    } else {
      mean += 1 / (1. + n_sample) * (sample - mean);
    }
    n_sample++;
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
  Eigen::SparseVector<double> rsqrt_diag;

  std::normal_distribution<double> dist;

  double omega; // SOR parameter

  Eigen::SparseVector<double> mean;
  std::size_t n_sample = 1;
  bool est_mean; // true if mean should be estimated during sampling

  // mpi rank -> vertex indices we need to send
  std::unordered_map<int, std::vector<int>> mpi_send;
  // mpi rank -> vertex indices we will receive
  std::unordered_map<int, std::vector<int>> mpi_recv;
};
} // namespace pargibbs
