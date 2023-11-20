#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "types.hh"

namespace parmgmc {
struct Lattice {
  using IndexType = int;

  Lattice(std::size_t dim, IndexType vertices_per_dim,
          ParallelLayout layout = ParallelLayout::None,
          LatticeOrdering ordering = LatticeOrdering::Lexicographic);

  IndexType get_vertices_per_dim() const { return n_vertices_per_dim; }
  IndexType get_n_total_vertices() const {
    return std::pow(n_vertices_per_dim, dim);
  }

  Lattice coarsen() const;

  std::size_t dim;

  // clang-format off
  /* Stores indices of vertices in a CSR style format.
     Consider the following 2d grid:

     If the lattice is ordered lexicografically

     6 -- 7 -- 8
     |    |    |
     3 -- 4 -- 5
     |    |    |
     0 -- 1 -- 2

     then the two arrays `adj_idx` and `adj_vert` look as follows:

     idx_adj  = [ 0,    2,       5,    7,      10,         14,      17,   19,      22,   24] 
     vert_adj = [ 1, 3, 0, 2, 4, 1, 5, 4, 6, 0, 3, 5, 7, 1, 4, 8, 2, 7, 3, 6, 8, 4, 7, 5 ]



     If the lattice is ordered in red-black ordering

     3 -- 8 -- 4
     |    |    |
     6 -- 2 -- 7
     |    |    |
     0 -- 5 -- 1

     then the two arrays `adj_idx` and `adj_vert` look as follows:

     idx_adj  = [ 0,    2,    4,          8,   10,   12,      15,      18,      21,      24 ]
     vert_adj = [ 5, 6, 5, 7, 6, 7, 8, 5, 8, 6, 8, 7, 0, 1, 2, 2, 3, 0, 2, 4, 1, 3, 4, 2 ]

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

  ParallelLayout layout;
  LatticeOrdering ordering;
};

} // namespace parmgmc
