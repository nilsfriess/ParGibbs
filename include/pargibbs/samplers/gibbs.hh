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

#include <Eigen/Sparse>

namespace pargibbs {
template <class LinearOperator, class Engine = std::mt19937_64>
class GibbsSampler {
public:
  GibbsSampler(const LinearOperator &linear_operator, Engine &engine,
               double omega = 1.)
      : linear_operator(linear_operator), prec(linear_operator.get_matrix()),
        engine(engine), omega(omega) {
    // TODO: We extract the whole diagonal, even though only a small part is
    // acutally required by the current MPI rank
    auto m_inv_diag = prec.diagonal().array().cwiseInverse();
    inv_diag.resize(m_inv_diag.rows(), m_inv_diag.cols());
    inv_diag = std::move(m_inv_diag);

    auto m_rsqrt_diag = inv_diag.sqrt();
    rsqrt_diag.resize(m_rsqrt_diag.rows(), m_rsqrt_diag.cols());
    rsqrt_diag = std::move(m_rsqrt_diag);
    // rand = diag.array().rsqrt().matrix().asDiagonal() * rand;
    // rand *= std::sqrt(omega * (2 - omega));
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
  Vector sample(const Vector &initial, [[maybe_unused]] std::size_t n_samples) {
    const auto &lattice = linear_operator.get_lattice();

    const auto red_points = lattice.get_my_points().first;
    const auto black_points = lattice.get_my_points().second;

    // Next, setup a list of indices that we need to send to MPI neighbours and
    // also a list of indices that we expect from our neighbours. This way, we
    // only need to communicate the values that were computed, not the indices.
    // In total, we manage four maps that all map MPI ranks to indices:
    //   - `own_red_points`: Indices of our red points that another MPI process
    //     needs.
    //   - `own_black_points`: Indices of our black points that another MPI
    //     process needs.
    //   - `ext_red_points`: Indices of red points that we expect to receive
    //     from another MPI rank.
    //   - `ext_black_points`: Indices of black points that we expect to receive
    //     from another MPI rank.
    std::unordered_map<int, std::vector<std::size_t>> own_red_points,
        own_black_points, ext_red_points, ext_black_points;

    const auto [size, rank] = mpi_helper::get_size_rank();

    for (const auto &point : red_points)
      for (const auto &neighbour : lattice.get_neighbours(point))
        if (neighbour.mpi_owner != rank) {
          own_red_points[neighbour.mpi_owner].push_back(point.actual_index);
          ext_black_points[neighbour.mpi_owner].push_back(
              neighbour.actual_index);
        }

    for (const auto &point : black_points)
      for (const auto &neighbour : lattice.get_neighbours(point))
        if (neighbour.mpi_owner != rank) {
          own_black_points[neighbour.mpi_owner].push_back(point.actual_index);
          ext_red_points[neighbour.mpi_owner].push_back(neighbour.actual_index);
        }

    Vector rand;
    rand.resize(red_points.size() + black_points.size());

    Vector next(initial);
    Vector mean(next);
    mean *= 1. / n_samples;

    for (std::size_t sample = 0; sample < n_samples; ++sample) {
      std::generate(rand.begin(), rand.end(), [&]() { return dist(engine); });

      sample_at_points(next, red_points, rand);
      send_recv_points(next, own_red_points, ext_red_points);

      sample_at_points(next, black_points, rand);
      send_recv_points(next, own_black_points, ext_black_points);

      mean += 1. / n_samples * next;
    }

    return mean;
  }

private:
  const LinearOperator &linear_operator;
  const typename LinearOperator::MatrixType &prec;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>
      inv_diag; // = prec.diagonal().array().cwiseInverse();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>
      rsqrt_diag; // = inv_diag.sqrt();

  Engine &engine;

  std::normal_distribution<double> dist;

  double omega; // SOR parameter

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

      // const auto rand_num = rand[rand_cnt++];
      // next[row] =
      //     (1 - omega) * next[row] + rand_num - omega * inv_diag[row] *
      // sum;
      sample[row] = rand[row] * rsqrt_diag(row, 0) - inv_diag(row, 0) * sum;
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
    reqs.clear();

    for (const auto &[source, indices] : source_points) {
      std::vector<double> values(indices.size());

      MPI_Recv(values.data(), indices.size(), MPI_DOUBLE, source, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (std::size_t i = 0; i < indices.size(); ++i)
        sample[indices[i]] = values[i];
    }
  }
};
}; // namespace pargibbs
