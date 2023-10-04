#pragma once

#include "pargibbs/common/log.hh"
#include "pargibbs/lattice/coordinate.hh"
#include "pargibbs/lattice/helpers.hh"
#include "pargibbs/mpi_helper.hh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pargibbs {

template <std::size_t dim> struct LatticePoint {
  Coordinate<dim> coordinate;
  int mpi_owner;
  std::size_t linear_index;
  std::size_t actual_index;
};

enum class LatticeOrdering { Rowwise, RedBlack };

// For a red-black ordered lattice, there are different ways (in dim >= 2) to
// decompose the lattice and distribute the points to the MPI processes.
// Currently, only `BlockRow` layout is supported which means that each MPI
// process is assigned a block of `rows` of the 2D lattice (for 3D it is not
// implemented yet).
enum class ParallelLayout { BlockRow, /* Checkerboard */ };

// Represents a square discrete lattice of the cube [0,1]^dim.
// The lattice points are stored in a std::vector.

// TODO: Extend to more colours.
template <std::size_t dim, LatticeOrdering ordering = LatticeOrdering::Rowwise,
          ParallelLayout layout = ParallelLayout::BlockRow>
class Lattice {
  static_assert(dim >= 1 && dim <= 3, "Dimension must be between 1 and 3");

public:
  static constexpr std::size_t Dim = dim;
  static constexpr LatticeOrdering Ordering = ordering;
  static constexpr ParallelLayout Layout = layout;

  Lattice(std::size_t points_per_dim)
      : n_points_per_dim(points_per_dim), meshwidth(1. / (points_per_dim - 1)),
        points(std::pow(points_per_dim, dim)) {
    n_red_points = (get_total_points() + 1) / 2;

    if (ordering == LatticeOrdering::RedBlack and (points_per_dim % 2 == 0)) {
      n_points_per_dim++;
      points.resize(points.size() + n_points_per_dim);

      PARGIBBS_DEBUG
          << "The number of points per lattice dimension must be "
             "odd when using red/black ordering. Incrementing and using "
          << n_points_per_dim << " instead.\n";
    }

    if (mpi_helper::get_rank() == 0) {
      PARGIBBS_DEBUG << "Initialising " << dim << "D lattice with "
                     << n_points_per_dim
                     << " points per dim (total: " << get_total_points()
                     << " points).\n";
    }

    // Computes the MPI rank that is responsible for the lattice point with
    // index idx
    const auto rank_for_index = [&](auto idx) {
      const auto [size, rank] = mpi_helper::get_size_rank();
      const auto n_rows = get_points_per_dim() / size;

      if (n_rows == 0)
        throw std::runtime_error(
            "Not enough lattice points to distribute among processes");

      for (int r = 0; r < size; ++r) {
        std::size_t first = 0;
        std::size_t last = 0;
        if constexpr (dim == 1) {
          first = r * n_rows;
          last = first + n_rows;
        } else if (dim == 2) {
          first = r * n_rows * get_points_per_dim();
          last = first + n_rows * get_points_per_dim() - 1;
        } else {
          static_assert(dim != 3, "Not implemented");
        }
        assert(last < get_total_points());

        if (r == size - 1)
          last = get_total_points() - 1;

        if (idx >= first && idx <= last)
          return r;
      }

      assert(false && "Unreachable");
      return -1;
    };

    // TODO: Only construct the lattice points that the current MPI rank
    // actually needs
    for (std::size_t idx = 0; idx < get_total_points(); ++idx) {
      LatticePoint<dim> point;

      auto coord = linear_to_coord<dim>(idx, n_points_per_dim);
      scale_coord(coord, meshwidth);

      if constexpr (ordering == LatticeOrdering::RedBlack) {
        if (idx % 2 == 0) {
          coord.type = CoordinateType::Red;
          point.actual_index = idx / 2;
        } else {
          coord.type = CoordinateType::Black;
          point.actual_index = n_red_points + (idx - 1) / 2;
        }
      } else {
        point.actual_index = idx;
        coord.type = CoordinateType::None;
      }

      point.mpi_owner = rank_for_index(idx);

      point.linear_index = idx;
      point.coordinate = coord;
      points.at(idx) = point;
    }
  }

  std::size_t get_total_points() const { return points.size(); }
  std::size_t get_points_per_dim() const { return n_points_per_dim; }

  // Get indices of the lattice points that the current MPI rank is responsible
  // for. The pair that is returned contains a vector of the red points and a
  // vector of the black points. In case of sequential execution, the second
  // vector is empty.
  using LatticePoints = std::vector<LatticePoint<dim>>;
  std::pair<LatticePoints, LatticePoints> get_my_points() const {
    const auto [size, rank] = mpi_helper::get_size_rank();

    LatticePoints red;
    LatticePoints black;

    for (const auto &point : points)
      if (point.mpi_owner == rank) {
        if (point.coordinate.type == CoordinateType::Black)
          black.push_back(point);
        else
          red.push_back(point);
      }

    return {red, black};
  }

  const Coordinate<dim> &get_coord(std::size_t index) const {
    return points.at(index);
  }

  auto begin() { return points.begin(); }
  auto end() { return points.end(); }

  auto begin() const { return points.begin(); }
  auto end() const { return points.end(); }

  std::vector<LatticePoint<dim>>
  get_neighbours(const LatticePoint<dim> &point) const {
    std::vector<LatticePoint<dim>> neighbours;
    neighbours.reserve(4);

    // If not at left border (= if we have a left neighbor)
    if (point.linear_index % n_points_per_dim != 0)
      neighbours.push_back(points.at(point.linear_index - 1));

    // If not at right border (= if we have a right neighbor)
    if (point.linear_index % n_points_per_dim != n_points_per_dim - 1)
      neighbours.push_back(points.at(point.linear_index + 1));

    if constexpr (dim > 1) {
      // If not at bottom border (= if we have a bottom neighbor)
      if (point.linear_index / n_points_per_dim != 0)
        neighbours.push_back(points.at(point.linear_index - n_points_per_dim));

      if (point.linear_index / n_points_per_dim != n_points_per_dim - 1)
        neighbours.push_back(points.at(point.linear_index + n_points_per_dim));
    }

    if constexpr (dim > 2) {
      static_assert(dim != 3, "Not implemented yet");
    }

    return neighbours;
  }

private:
  std::size_t n_points_per_dim;
  std::size_t n_red_points;

  double meshwidth;

  std::vector<LatticePoint<dim>> points;
};

} // namespace pargibbs
