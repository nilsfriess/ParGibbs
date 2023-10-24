#pragma once

#include "coordinate.hh"

#include <cstddef>

namespace pargibbs {
enum class Colour { Red, Black, None };

template <std::size_t dim> struct LatticePoint {
  int mpi_owner;
  std::size_t linear_index;
  std::size_t actual_index;
};

enum class LatticeOrdering { Rowwise, RedBlack };

// For a red-black ordered lattice, there are different ways (in dim >= 2) to
// decompose the lattice and distribute the points to the MPI processes.
// - `BlockRow` layout means that each MPI process is assigned a block of `rows`
//   of the 2D lattice (for 3D it is not implemented yet).
// - `WORB` uses weighted orthogonal recursive bisection to distribute the
//    points such that communcation is (approximately) minimised.
  enum class ParallelLayout { None, BlockRow, WORB };
}; // namespace pargibbs
