#pragma once

#include "coordinate.hh"
#include "helpers.hh"
#include "pargibbs/common/log.hh"
#include "pargibbs/mpi_helper.hh"
#include "partition.hh"
#include "types.hh"

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
// TODO: Extend to more colours.
template <std::size_t dim, LatticeOrdering ordering = LatticeOrdering::Rowwise,
          ParallelLayout layout = ParallelLayout::None>
class Lattice {
  static_assert(dim >= 1 && dim <= 3, "Dimension must be between 1 and 3");

public:
  static constexpr std::size_t Dim = dim;
  static constexpr LatticeOrdering Ordering = ordering;
  static constexpr ParallelLayout Layout = layout;

  Lattice(std::size_t vertices_per_dim)
      : n_vertices_per_dim(vertices_per_dim),
        meshwidth(1. / (n_vertices_per_dim - 1)) {
    if (ordering == LatticeOrdering::RedBlack and
        (n_vertices_per_dim % 2 == 0)) {
      n_vertices_per_dim++;

      PARGIBBS_DEBUG
          << "The number of points per lattice dimension must be "
             "odd when using red/black ordering. Incrementing and using "
          << n_vertices_per_dim << " instead.\n";
    }

    if (mpi_helper::is_debug_rank()) {
      PARGIBBS_DEBUG << "Initialising " << dim << "D lattice with "
                     << n_vertices_per_dim
                     << " vertices per dim (total: " << get_n_total_vertices()
                     << " vertices).\n";
    }

    // TODO: This assumes 2D
    constexpr int n_neighbours = 4;
    vertices.resize((n_neighbours + 1) * std::pow(n_vertices_per_dim, dim));

    /// Loop over all vertices and store the vertex index and the indices of the
    /// vertex's neighbours (or -1 in case there is no neighbour in the
    /// corresponding direction)
    // TODO: This currently only works for 2D
    int tot_vert = static_cast<int>(get_n_total_vertices());
    for (int i = 0; i < tot_vert; ++i) {
      int start = (n_neighbours + 1) * i;
      vertices.at(start) = i;

      // Check if north neighbour exists
      if (i + n_vertices_per_dim < get_n_total_vertices())
        vertices.at(start + 1) = i + n_vertices_per_dim;
      else
        vertices.at(start + 1) = -1;

      // Check if east neighbour exists
      if (i % n_vertices_per_dim != n_vertices_per_dim - 1)
        vertices.at(start + 2) = i + 1;
      else
        vertices.at(start + 2) = -1;

      // Check if south neighbour exists
      if (i - static_cast<int>(n_vertices_per_dim) >= 0)
        vertices.at(start + 3) = i - static_cast<int>(n_vertices_per_dim);
      else
        vertices.at(start + 3) = -1;

      // Check if west neighbour exists
      if (i % n_vertices_per_dim != 0)
        vertices.at(start + 4) = i - 1;
      else
        vertices.at(start + 4) = -1;
    }

    /// Next we partition the domain and store for each vertex the rank of the
    /// MPI process that is responsible for that vertex.
    mpiowner.resize(get_n_total_vertices());
    std::fill(mpiowner.begin(), mpiowner.end(), -1);
    if constexpr (layout == ParallelLayout::None) {
      std::fill(mpiowner.begin(), mpiowner.end(), 0);
    } else {
      auto size = mpi_helper::get_size();
      std::array<std::size_t, dim> dimensions;
      for (auto &entry : dimensions)
        entry = n_vertices_per_dim;
      auto partitions = detail::make_partition<layout>(dimensions, size);
      assert(partitions.size() == static_cast<std::size_t>(size));

      for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
        const auto &partition = partitions[p_idx];

        std::size_t tot_points = 1;
        for (auto e : partition.size)
          tot_points *= e;

        for (std::size_t idx = 0; idx < tot_points; ++idx) {
          // Convert linear index within partition to global coordinate
          auto local_coord = linear_to_xyz(idx, partition.size);
          for (std::size_t d = 0; d < dim; ++d)
            local_coord[d] += partition.start[d];

          // And convert this to a global linear index
          auto global_idx = xyz_to_linear(local_coord, n_vertices_per_dim);

          mpiowner.at(global_idx) = p_idx;
        }
      }

      /// As a helper array, we also separately store the vertices that the
      /// current MPI rank is responsible for as well as the subset of those
      /// that have neighbouring vertices that are owned by a different MPI rank
      /// (this can be used when checking which vertices have to be communicated
      /// to other MPI ranks).
      // TODO: Maybe we should store separately in which direction the
      // neighbours lies
      for (std::size_t v = 0; v < get_n_total_vertices(); ++v) {
        if (mpiowner[v] == mpi_helper::get_rank()) {
          own_vertices.push_back(v);

          for (std::size_t n = 1; n < 5; ++n) {
            if ((vertices[5 * v + n] != -1) &&
                (mpiowner[vertices[5 * v + n]] != mpi_helper::get_rank())) {
              // At least one of the neighbouring vertices of the current
              // vertex is not owned by the current MPI rank, i.e., it is at
              // the border of a partition
              border_vertices.push_back(v);
              break;
            }
          }
        }
      }
    }
  }

  std::size_t get_vertices_per_dim() const { return n_vertices_per_dim; }
  std::size_t get_n_total_vertices() const {
    return std::pow(n_vertices_per_dim, dim);
  }

  // Stores indices of vertices and direct neighbours in the following format:
  // index ||     i     |    i+1    |    i+2   |    i+3    |   i+4    |
  // value || vertex id | north nb. | east nb. | south nb. | west nb. |
  // The total size of this vector is thus 5 * n_vertices.
  std::vector<int> vertices;

  // Indices of vertices that the current MPI rank owns
  std::vector<int> own_vertices;
  // Indices of vertices that the current MPI rank owns and that have to be
  // communicated to some other MPI rank
  std::vector<int> border_vertices;

  // Maps indices to mpi ranks
  std::vector<int> mpiowner;

  std::size_t n_vertices_per_dim;

  double meshwidth;
};

} // namespace pargibbs
