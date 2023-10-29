#include "pargibbs/lattice/lattice.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ostream>
#include <stdexcept>

#include "pargibbs/common/log.hh"
#include "pargibbs/lattice/partition.hh"
#include "pargibbs/lattice/types.hh"
#include "pargibbs/mpi_helper.hh"

namespace pargibbs {

Lattice::Lattice(std::size_t dim, IndexType vertices_per_dim,
                 ParallelLayout layout)
    : dim(dim), n_vertices_per_dim(vertices_per_dim),
      meshwidth(1. / (n_vertices_per_dim - 1)), layout{layout} {
  if (dim < 1 or dim > 3)
    throw std::runtime_error("Dimension must be between 1 and 3");

  if (n_vertices_per_dim % 2 == 0) {
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

    if (dim > 1) {
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
  if (layout == ParallelLayout::None) {
    std::fill(mpiowner.begin(), mpiowner.end(), 0);
  } else {
    auto size = mpi_helper::get_size();
    mpiowner = detail::make_partition(*this, size);
    assert((IndexType)mpiowner.size() == get_n_total_vertices());

    if (mpi_helper::is_debug_rank() && get_vertices_per_dim() < 10) {
      PARGIBBS_DEBUG << "Partitioned domain:\n";
      for (IndexType i = 0; i < (IndexType)mpiowner.size(); ++i) {
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
      if (mpiowner.at(i) == (IndexType)mpi_helper::get_rank()) {
        own_vertices.push_back((IndexType)i);

        for (IndexType j = adj_idx.at(i); j < adj_idx.at(i + 1); ++j) {
          auto nb = adj_vert.at(j); // Get index of neighbouring vertex
          if (mpiowner.at(nb) != (IndexType)mpi_helper::get_rank()) {
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
} // namespace pargibbs
