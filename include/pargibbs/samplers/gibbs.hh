#pragma once

#include "../mpi_helper.hh"
#include "pargibbs/common/log.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <type_traits>
#include <unordered_map>

#include <Eigen/Eigen>

namespace pargibbs {
template <class LinearOperator, class Engine = std::mt19937_64>
class GibbsSampler {
public:
  GibbsSampler(const LinearOperator &linear_operator, Engine &engine,
               bool est_mean = false, double omega = 1.)
      : linear_operator(linear_operator), prec(linear_operator.get_matrix()),
        engine(engine), omega(omega), est_mean(est_mean) {

    const auto &lattice = linear_operator.get_lattice();
    inv_diag.resize(prec.rows());
    rsqrt_diag.resize(prec.rows());

    for (auto v : lattice.own_vertices) {
      inv_diag.coeffRef(v) = 1. / prec.coeff(v, v);
      rsqrt_diag.coeffRef(v) = 1. / std::sqrt(prec.coeff(v, v));
    }

    if (est_mean)
      mean.resize(prec.rows());

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

  template <class Vector>
  Vector sample(const Vector &initial, std::size_t n_samples) {
    const auto &lattice = linear_operator.get_lattice();

    Eigen::VectorXd rand;
    rand.resize(initial.size());

    Vector next(initial);

    // std::vector<double> mpi_buf; //(lattice.border_vertices.size(), 0.);
    // mpi_buf.resize(1000);

    using It = typename LinearOperator::MatrixType::InnerIterator;
    for (std::size_t n = 0; n < n_samples; ++n) {
      std::generate(rand.begin(), rand.end(), [&]() { return dist(engine); });

      for (auto v : lattice.own_vertices) {

        // If v is odd, then this is a "black" vertex
        if (v % 2 != 0)
          continue;

        double sum = 0.;
        for (It it(prec, v); it; ++it) {
          assert(it.row() == v);
          if (it.col() != it.row())
            sum += it.value() * next.coeff(it.col());
        }

        next.coeffRef(v) =
            (1 - omega) * next.coeff(v) +
            rand[v] * std::sqrt(omega * (2 - omega)) * rsqrt_diag.coeff(v) -
            omega * inv_diag.coeff(v) * sum;
      }

      // Send all values
      // TODO: Maybe we should just send the red values here?
      for (auto &&[target, vs] : mpi_send) {
        std::vector<double> mpi_buf(vs.size());
        for (std::size_t i = 0; i < vs.size(); ++i)
          mpi_buf[i] = next.coeff(vs[i]);

        MPI_Send(mpi_buf.data(), vs.size(), MPI_DOUBLE, target, 0,
                 MPI_COMM_WORLD);
      }

      for (auto &&[source, vs] : mpi_recv) {
        std::vector<double> mpi_buf(vs.size());
        MPI_Recv(mpi_buf.data(), vs.size(), MPI_DOUBLE, source, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (std::size_t i = 0; i < vs.size(); ++i)
          if (vs[i] % 2 == 0)
            next.coeffRef(vs[i]) = mpi_buf[i];
      }

      for (auto v : lattice.own_vertices) {

        // If v is even, then this is a "red" vertex
        if (v % 2 == 0)
          continue;

        double sum = 0.;
        for (It it(prec, v); it; ++it) {
          if (it.col() != it.row())
            sum += it.value() * next.coeff(it.col());
        }

        next.coeffRef(v) =
            (1 - omega) * next.coeff(v) +
            rand[v] * std::sqrt(omega * (2 - omega)) * rsqrt_diag.coeff(v) -
            omega * inv_diag.coeff(v) * sum;
      }

      // Send all values
      // TODO: Maybe we should just send the black values here?
      for (auto &&[target, vs] : mpi_send) {
        std::vector<double> mpi_buf(vs.size());
        for (std::size_t i = 0; i < vs.size(); ++i)
          mpi_buf[i] = next.coeff(vs[i]);

        MPI_Send(mpi_buf.data(), vs.size(), MPI_DOUBLE, target, 0,
                 MPI_COMM_WORLD);
      }

      for (auto &&[source, vs] : mpi_recv) {
        std::vector<double> mpi_buf(vs.size());
        MPI_Recv(mpi_buf.data(), vs.size(), MPI_DOUBLE, source, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (std::size_t i = 0; i < vs.size(); ++i)
          if (vs[i] % 2 != 0)
            next.coeffRef(vs[i]) = mpi_buf[i];
      }

      if (est_mean)
        update_mean(next);
    }

    return next;
  }

  Eigen::SparseVector<double> get_mean() const {
    const auto &lattice = linear_operator.get_lattice();
    Eigen::SparseVector<double> local_mean(lattice.get_n_total_vertices());
    for (auto v : lattice.own_vertices)
      local_mean.insert(v) = mean.coeff(v);

    return local_mean;
  }

  void reset_mean() {
    n_sample = 0;
    mean.setZero();
  }

private:
  void update_mean(const auto &sample) {
    if (n_sample == 1) {
      mean = sample;
    } else {
      mean += 1 / (1. + n_sample) * (sample - mean);
    }
    n_sample++;
  }

  void setup_mpi_maps() {
    const auto &lattice = linear_operator.get_lattice();
    using IndexT = typename std::decay_t<decltype(lattice)>::IndexT;

    for (auto v : lattice.border_vertices) {
      for (IndexT n = lattice.adj_idx.at(v); n < lattice.adj_idx.at(v + 1);
           ++n) {
        auto nb_idx = lattice.adj_vert.at(n);
        // If we have a neighbour that is owned by another MPI process, then
        // - we need to send the value at `v` to this process at some point, and
        // - we will receive values at `nb_idx` from this process at some
        //   point.
        if (lattice.mpiowner[nb_idx] != (IndexT)mpi_helper::get_rank()) {
          mpi_send[lattice.mpiowner.at(nb_idx)].push_back(v);
          mpi_recv[lattice.mpiowner.at(nb_idx)].push_back(nb_idx);
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

  const LinearOperator &linear_operator;
  const typename LinearOperator::MatrixType &prec;
  Eigen::SparseVector<double> inv_diag;
  Eigen::SparseVector<double> rsqrt_diag;

  Engine &engine;

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
