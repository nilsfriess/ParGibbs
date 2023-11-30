#include "parmgmc/lattice/lattice.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ostream>
#include <stdexcept>

#include "parmgmc/common/log.hh"
#include "parmgmc/lattice/helpers.hh"
#include "parmgmc/lattice/partition.hh"
#include "parmgmc/lattice/types.hh"
#include "parmgmc/mpi_helper.hh"

namespace parmgmc {

Lattice::Lattice(std::size_t dim, IndexType vertices_per_dim,
                 ParallelLayout layout)
    : dim(dim), n_vertices_per_dim(vertices_per_dim),
      meshwidth(1. / (n_vertices_per_dim - 1)), layout{layout} {
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

    PARMGMC_DEBUG << "The number of points per lattice dimension must be "
                     "of the form 2^n + 1. Incrementing and using "
                  << n_vertices_per_dim << " instead.\n";
  }

  if (mpi_helper::is_debug_rank()) {
    PARMGMC_DEBUG << "Initialising " << dim << "D lattice with "
                  << n_vertices_per_dim
                  << " vertices per dim (total: " << get_n_total_vertices()
                  << " vertices)." << std::endl;
  }

  // Setup graph desribing the grid
  setup_graph();

  // Partition graph
  auto size = mpi_helper::get_size();
  mpiowner = detail::make_partition(*this, size);
  assert((IndexType)mpiowner.size() == get_n_total_vertices());
  print_partition();
  update_vertices();
}

void Lattice::setup_graph() {
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
}

void Lattice::print_partition() const {
  if (mpi_helper::get_size() > 1 && mpi_helper::is_debug_rank() &&
      get_vertices_per_dim() < 10) {
    PARMGMC_DEBUG << "Partitioned domain:\n";
    for (IndexType i = 0; i < (IndexType)mpiowner.size(); ++i) {
      PARMGMC_DEBUG_NP << mpiowner[i] << " ";
      if (i % get_vertices_per_dim() == get_vertices_per_dim() - 1) {
        PARMGMC_DEBUG_NP << "\n";
      }
    }
  }
}

void Lattice::update_vertices() {
  n_internal_vertices = 0;
  n_border_vertices = 0;
  for (IndexType i = 0; i < get_n_total_vertices(); ++i) {
    if (mpiowner.at(i) == (IndexType)mpi_helper::get_rank()) {
      n_internal_vertices++;
      own_vertices.push_back(i);

      bool is_border = false;
      for (IndexType j = adj_idx.at(i); j < adj_idx.at(i + 1); ++j) {
        auto nb = adj_vert.at(j); // Get index of neighbouring vertex
        if (mpiowner.at(nb) != (IndexType)mpi_helper::get_rank()) {
          is_border = true;
          n_border_vertices++;
          break;
        }
      }

      if (is_border)
        vertex_types.push_back(VertexType::Border);
      else
        vertex_types.push_back(VertexType::Internal);
    } else {
      // Vertex is not assigned directly to us, but might be a ghost vertex
      for (IndexType j = adj_idx.at(i); j < adj_idx.at(i + 1); ++j) {
        auto nb = adj_vert.at(j); // Get index of neighbouring vertex
        if (mpiowner.at(nb) == (IndexType)mpi_helper::get_rank()) {
          // If we own the neighbouring vertex, then this is a ghost vertex
          own_vertices.push_back(i);
          vertex_types.push_back(VertexType::Ghost);
          break;
        }
      }
    }
  }
}

Lattice Lattice::coarsen() const {
  if (n_vertices_per_dim == 1)
    throw std::runtime_error("Lattice::coarsen(): Lattice only consists of one "
                             "vertex, cannot coarsen further.");

  const std::vector<std::size_t> vert_per_dim(dim, n_vertices_per_dim);

  Lattice coarse_lattice;
  coarse_lattice.dim = dim;
  coarse_lattice.n_vertices_per_dim = (n_vertices_per_dim + 1) / 2;

  // Since we have a vector that maps lattice vertices to MPI ranks,
  // we can simply coarsen this map and recompute inner vertices, border
  // vertices etc. using the Lattice::update_vertices() method.

  const auto convert_to_coarse = [&](auto fine_idx) {
    auto fine_xyz = linear_to_xyz(fine_idx, vert_per_dim);
    for (auto &coord : fine_xyz)
      coord /= 2; // Compute coarsened coordinate
    return xyz_to_linear(fine_xyz, coarse_lattice.n_vertices_per_dim);
  };

  coarse_lattice.mpiowner.resize(coarse_lattice.get_n_total_vertices());
  for (IndexType i = 0; i < get_n_total_vertices(); ++i) {
    auto fine_xyz = linear_to_xyz(i, vert_per_dim);

    bool keep_on_coarse = true;
    for (auto coord : fine_xyz)
      // Coarsen by only keeping every second point in each direction
      if (coord % 2 != 0) {
        keep_on_coarse = false;
        break;
      }

    if (!keep_on_coarse)
      continue;

    coarse_lattice.mpiowner.at(convert_to_coarse(i)) = mpiowner.at(i);
  }

  coarse_lattice.setup_graph();
  coarse_lattice.update_vertices();

  return coarse_lattice;
}
} // namespace parmgmc
