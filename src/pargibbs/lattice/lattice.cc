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
                 ParallelLayout layout, LatticeOrdering ordering)
    : dim(dim), n_vertices_per_dim(vertices_per_dim),
      meshwidth(1. / (n_vertices_per_dim - 1)), layout{layout},
      ordering{ordering} {
  if (dim < 1 or dim > 3)
    throw std::runtime_error("Dimension must be between 1 and 3");

  // Check if vertices_per_dim is of the form 2^n + 1. If not, warn and
  // increase vertices_per_dim to the smallest number satisfying this (that is
  // not smaller than vertices_per_dim).
  auto n_elements = vertices_per_dim - 1;
  if ((n_elements & (n_elements - 1)) != 0) {
    IndexType power = 1;
    while (power < n_vertices_per_dim)
      power *= 2;
    n_vertices_per_dim = power + 1;

    PARGIBBS_DEBUG << "The number of points per lattice dimension must be "
                      "of the form 2^n + 1. Incrementing and using "
                   << n_vertices_per_dim << " instead.\n";
  }

  if (mpi_helper::is_debug_rank()) {
    PARGIBBS_DEBUG << "Initialising " << dim << "D lattice with "
                   << n_vertices_per_dim
                   << " vertices per dim (total: " << get_n_total_vertices()
                   << " vertices)." << std::endl;
  }

  // For lexicographic ordering, this function is just the identity. For red
  // black ordering, we map red indices to [0, 1, 2, ..., N/2] and black indices
  // to [N/2 + 1, N/2 + 2, ..., N].
  const auto convert_index = [&](auto i) {
    if (ordering == LatticeOrdering::Lexicographic)
      return i;

    if (i % 2 == 0)
      return i / 2;
    else
      return (get_n_total_vertices() / 2) + (i + 1) / 2;
  };

  const auto handle_vertex = [&](IndexType &curr_idx, IndexType i) {
    adj_idx.push_back(curr_idx);

    // Check if west neighbour exists
    if (i % n_vertices_per_dim != 0) {
      adj_vert.push_back(convert_index(i - 1));
      curr_idx++;
    }

    // Check if east neighbour exists
    if (i % n_vertices_per_dim != n_vertices_per_dim - 1) {
      adj_vert.push_back(convert_index(i + 1));
      curr_idx++;
    }

    if (dim > 1) {
      // Check if north neighbour exists
      if (i + n_vertices_per_dim < get_n_total_vertices()) {
        adj_vert.push_back(convert_index(i + n_vertices_per_dim));
        curr_idx++;
      }

      // Check if south neighbour exists
      if (i >= n_vertices_per_dim) {
        adj_vert.push_back(convert_index(i - n_vertices_per_dim));
        curr_idx++;
      }
    }
  };

  if (ordering == LatticeOrdering::Lexicographic) {
    IndexType curr_idx = 0;

    // In case of lexicographic ordering, we loop over all indices directly
    for (IndexType i = 0; i < get_n_total_vertices(); ++i)
      handle_vertex(curr_idx, i);
    adj_idx.push_back(adj_vert.size());
  } else {
    IndexType curr_idx = 0;

    // In case of red/black ordering, we first traverse "red" (=even) vertices
    for (IndexType i = 0; i < get_n_total_vertices(); i += 2)
      handle_vertex(curr_idx, i);

    // Next, we traverse "black" (=odd) indices
    for (IndexType i = 1; i < get_n_total_vertices(); i += 2)
      handle_vertex(curr_idx, i);

    adj_idx.push_back(adj_vert.size());
  }

  /// Next we partition the domain and store for each vertex the rank of the
  /// MPI process that is responsible for that vertex.
  auto size = mpi_helper::get_size();
  mpiowner = detail::make_partition(*this, size);
  assert((IndexType)mpiowner.size() == get_n_total_vertices());

  if (mpi_helper::get_size() > 1 && mpi_helper::is_debug_rank() &&
      get_vertices_per_dim() < 10) {
    PARGIBBS_DEBUG << "Partitioned domain:\n";
    for (IndexType i = 0; i < (IndexType)mpiowner.size(); ++i) {
      PARGIBBS_DEBUG_NP << mpiowner[convert_index(i)] << " ";
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
  const auto handle_mpi_vertex = [&](auto i) {
    auto conv_idx = convert_index(i);

    if (mpiowner.at(conv_idx) == (IndexType)mpi_helper::get_rank()) {
      own_vertices.push_back(conv_idx);

      for (IndexType j = adj_idx.at(conv_idx); j < adj_idx.at(conv_idx + 1);
           ++j) {
        auto nb = adj_vert.at(j); // Get index of neighbouring vertex
        if (mpiowner.at(nb) != (IndexType)mpi_helper::get_rank()) {
          // At least one of the neighbouring vertices of the current
          // vertex is not owned by the current MPI rank, i.e., it is at
          // the border of a partition
          border_vertices.push_back(conv_idx);
          break;
        }
      }
    }
  };

  if (ordering == LatticeOrdering::Lexicographic) {
    for (IndexType i = 0; i < get_n_total_vertices(); ++i)
      handle_mpi_vertex(i);
  } else {
    for (IndexType i = 0; i < get_n_total_vertices(); i += 2)
      handle_mpi_vertex(i);
    for (IndexType i = 1; i < get_n_total_vertices(); i += 2)
      handle_mpi_vertex(i);
  }
}

Lattice Lattice::coarsen() const {
  if (n_vertices_per_dim == 1)
    throw std::runtime_error("Lattice::coarsen(): Lattice only consists of one "
                             "vertex, cannot coarsen further.");

  return Lattice(2, (n_vertices_per_dim + 1) / 2, layout);
}
} // namespace pargibbs
