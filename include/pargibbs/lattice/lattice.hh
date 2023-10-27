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
template <std::size_t dim, typename IndexType,
          LatticeOrdering ordering = LatticeOrdering::Rowwise,
          ParallelLayout layout = ParallelLayout::None>
class Lattice {
  static_assert(dim >= 1 && dim <= 3, "Dimension must be between 1 and 3");

public:
  static constexpr std::size_t Dim = dim;
  static constexpr LatticeOrdering Ordering = ordering;
  static constexpr ParallelLayout Layout = layout;
  using IndexT = IndexType;

  Lattice(IndexType vertices_per_dim)
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

    /// Construct array of vertices in CSR style.
    IndexType curr_idx = 0;
    for (IndexType i = 0; i < get_n_total_vertices(); ++i) {
      adj_idx.push_back(curr_idx);

      // Check if west neighbour exists
      if (i % n_vertices_per_dim != 0) {
        adj_vert.push_back(i - 1);
        curr_idx++;
      }

      // Check if east neighbour exists
      if (i % n_vertices_per_dim != n_vertices_per_dim - 1) {
        adj_vert.push_back(i + 1);
        curr_idx++;
      }

      if constexpr (dim > 1) {
        // Check if north neighbour exists
        if (i + n_vertices_per_dim < get_n_total_vertices()) {
          adj_vert.push_back(i + n_vertices_per_dim);
          curr_idx++;
        }

        // Check if south neighbour exists
        if (i >= n_vertices_per_dim) {
          adj_vert.push_back(i - n_vertices_per_dim);
          curr_idx++;
        }
      }
    }
    adj_idx.push_back(adj_vert.size());

    /// Next we partition the domain and store for each vertex the rank of the
    /// MPI process that is responsible for that vertex.
    if constexpr (layout == ParallelLayout::None) {
      std::fill(mpiowner.begin(), mpiowner.end(), 0);
    } else {
      auto size = mpi_helper::get_size();
      mpiowner = detail::make_partition(*this, size);
      assert((IndexType)mpiowner.size() == get_n_total_vertices());

      if (mpi_helper::is_debug_rank() && get_vertices_per_dim() < 10) {
        PARGIBBS_DEBUG << "Partitioned domain:\n";
        for (std::size_t i = 0; i < mpiowner.size(); ++i) {
          PARGIBBS_DEBUG_NP << mpiowner[i] << " ";
          if (i % get_vertices_per_dim() == get_vertices_per_dim() - 1) {
            PARGIBBS_DEBUG_NP << "\n";
          }
        }
      }

      /// As a helper array, we also separately store the vertices that the
      /// current MPI rank is responsible for as well as the subset of those
      /// that have neighbouring vertices that are owned by a different MPI
      /// rank (this can be used when checking which vertices have to be
      /// communicated to other MPI ranks).
      for (std::size_t i = 0; i < (std::size_t)get_n_total_vertices(); ++i) {
        if (mpiowner.at(i) == mpi_helper::get_rank()) {
          own_vertices.push_back((IndexType)i);

          for (IndexType j = adj_idx.at(i); j < adj_idx.at(i + 1); ++j) {
            auto nb = adj_vert.at(j); // Get index of neighbouring vertex
            if (mpiowner.at(nb) != mpi_helper::get_rank()) {
              // At least one of the neighbouring vertices of the current
              // vertex is not owned by the current MPI rank, i.e., it is at
              // the border of a partition
              border_vertices.push_back((IndexType)i);
              break;
            }
          }
        }
      }
    }
  }

  IndexType get_vertices_per_dim() const { return n_vertices_per_dim; }
  IndexType get_n_total_vertices() const {
    return std::pow(n_vertices_per_dim, dim);
  }

  // clang-format off
  /* Stores indices of vertices in a CSR style format.
     Consider the following 2d grid:

     6 -- 7 -- 8
     |    |    |
     3 -- 4 -- 5
     |    |    |
     0 -- 1 -- 2

     Then the two arrays `adj_idx` and `adj_vert` look as follows:
     idx_adj  = [ 0,    2,       5,    7,      10,         14,      17,   19,      22,   24] 
     vert_adj = [ 1, 3, 0, 2, 4, 1, 5, 4, 6, 0, 3, 5, 7, 1, 4, 8, 2, 7, 3, 6, 8, 4, 7, 5 ]
  */
  // clang-format on
  std::vector<IndexType> adj_idx;
  std::vector<IndexType> adj_vert;

  // Indices of vertices that the current MPI rank owns
  std::vector<IndexType> own_vertices;
  // Indices of vertices that the current MPI rank owns and that have to be
  // communicated to some other MPI rank
  std::vector<IndexType> border_vertices;

  // Maps indices to mpi ranks
  std::vector<IndexType> mpiowner;

  IndexType n_vertices_per_dim;

  double meshwidth;
};

} // namespace pargibbs
