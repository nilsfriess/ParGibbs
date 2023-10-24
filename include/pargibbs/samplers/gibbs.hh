#pragma once

#include "../mpi_helper.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
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

    if (est_mean) {
      mean.resize(prec.rows());
    }
  }

  /* If MPI is used the sampler works as follows:

     1. Each MPI process computes its associated red and black lattice points.
     These points correspond to rows in the matrix that can be processed in
     parallel. The process then uses those matrix rows and the entries in the
     column; it does not care if the other rows in the matrix are there or not,
     it just accesses its associated rows.

     2. If the process has processed all red lattice points, it communicates the
     computed values to the neighbors who need them and receives the values it
     needs. As soon as it has received the values, it starts updating the black
     points and then again sends/receives the updates to compute the red points
     in the next iteration and so forth.
   */
  template <class Vector>
  Vector sample(const Vector &initial, std::size_t n_samples) {
    const auto &lattice = linear_operator.get_lattice();

    Vector rand;
    rand.resize(initial.size());

    Vector next(initial);

    using It = typename LinearOperator::MatrixType::InnerIterator;
    for (std::size_t n = 0; n < n_samples; ++n) {
      std::generate(rand.begin(), rand.end(), [&]() { return dist(engine); });

      for (auto v : lattice.own_vertices) {
        double sum = 0.;
        for (It it(prec, v); it; ++it) {
          assert((int)row == it.row());

          if (it.col() != it.row())
            sum += it.value() * next[it.col()];
        }

        next[v] =
            (1 - omega) * next[v] +
            rand[v] * std::sqrt(omega * (2 - omega)) * rsqrt_diag.coeff(v) -
            omega * inv_diag.coeff(v) * sum;
      }

      if (est_mean)
        update_mean(next);
    }

    return next;

    // const auto red_points = lattice.get_my_points().first;
    // const auto black_points = lattice.get_my_points().second;

    // std::cout << red_points.size() << " ";
    // std::cout << black_points.size() << "\n";

    // // Next, setup a list of indices that we need to send to MPI neighbours
    // and
    // // also a list of indices that we expect from our neighbours. This way,
    // we
    // // only need to communicate the values that were computed, not the
    // indices.
    // // In total, we manage four maps that all map MPI ranks to indices:
    // //   - `own_red_points`: Indices of our red points that another MPI
    // process
    // //     needs.
    // //   - `own_black_points`: Indices of our black points that another MPI
    // //     process needs.
    // //   - `ext_red_points`: Indices of red points that we expect to receive
    // //     from another MPI rank.
    // //   - `ext_black_points`: Indices of black points that we expect to
    // receive
    // //     from another MPI rank.
    // std::unordered_map<int, std::vector<std::size_t>> own_red_points,
    //     own_black_points, ext_red_points, ext_black_points;

    // const auto [size, rank] = mpi_helper::get_size_rank();

    // for (const auto &point : red_points)
    //   for (const auto &neighbour : lattice.get_neighbours(point))
    //     if (neighbour.mpi_owner != rank) {
    //       own_red_points[neighbour.mpi_owner].push_back(point.actual_index);
    //       ext_black_points[neighbour.mpi_owner].push_back(
    //           neighbour.actual_index);
    //     }

    // for (const auto &point : black_points)
    //   for (const auto &neighbour : lattice.get_neighbours(point))
    //     if (neighbour.mpi_owner != rank) {
    //       own_black_points[neighbour.mpi_owner].push_back(point.actual_index);
    //       ext_red_points[neighbour.mpi_owner].push_back(neighbour.actual_index);
    //     }

    // Vector rand;
    // rand.resize(initial.size());

    // Vector next(initial);

    // for (std::size_t sample = 0; sample < n_samples; ++sample) {
    //   // FIXME: We generate more random samples than needed
    //   std::generate(rand.begin(), rand.end(), [&]() { return dist(engine);
    //   });

    //   sample_at_points(next, red_points, rand);
    //   send_recv_points(next, own_red_points, ext_red_points);

    //   sample_at_points(next, black_points, rand);
    //   send_recv_points(next, own_black_points, ext_black_points);

    //   if (est_mean)
    //     update_mean(next);
    // }

    // return next;
  }

  Eigen::VectorXd get_mean() const { return mean; }

  void reset_mean() {
    n_sample = 0;
    mean.setZero();
  }

private:
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

  void sample_at_points(auto &sample, const auto &points, const auto &rand) {
    using It = typename LinearOperator::MatrixType::InnerIterator;

    std::for_each(points.begin(), points.end(), [&](const auto &point) {
      const auto row = point.actual_index;

      double sum = 0.;
      for (It it(prec, row); it; ++it) {
        assert((int)row == it.row());

        if (it.col() != it.row())
          sum += it.value() * sample[it.col()];
      }

      sample[row] =
          (1 - omega) * sample[row] +
          rand[row] * std::sqrt(omega * (2 - omega)) * rsqrt_diag.coeff(row) -
          omega * inv_diag.coeff(row) * sum;
    });
  }

  void send_recv_points(auto &sample, const auto &target_points,
                        const auto &source_points) {
    std::vector<MPI_Request> reqs;

    for (const auto &[target, indices] : target_points) {
      std::vector<double> values(indices.size());
      for (std::size_t i = 0; i < indices.size(); ++i)
        values[i] = sample[indices[i]];

      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Isend(values.data(), indices.size(), MPI_DOUBLE, target, 0,
                MPI_COMM_WORLD, &reqs.back());
    }

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    for (const auto &[source, indices] : source_points) {
      std::vector<double> values(indices.size());

      MPI_Recv(values.data(), indices.size(), MPI_DOUBLE, source, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (std::size_t i = 0; i < indices.size(); ++i)
        sample[indices[i]] = values[i];
    }
  }

  void update_mean(const auto &sample) {
    const auto &lattice = linear_operator.get_lattice();

    Eigen::SparseVector<double> sparse_sample(lattice.get_n_total_vertices());

    for (auto v : lattice.own_vertices) {
      sparse_sample.coeffRef(v) = sample[v];

      if (n_sample == 1) {
        mean = sparse_sample;
      } else {
        mean += 1 / (1. + n_sample) * (sparse_sample - mean);
      }

      n_sample++;
    }
  }
};
} // namespace pargibbs
