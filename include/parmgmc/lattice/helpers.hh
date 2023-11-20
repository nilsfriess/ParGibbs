#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace parmgmc {
inline std::size_t xyz_to_linear(std::vector<std::size_t> coord,
                                 std::vector<std::size_t> points_per_dim) {
  const auto dim = coord.size();
  if (dim == 1)
    return coord[0];
  else if (dim == 2)
    return coord[1] * points_per_dim[0] + coord[0];
  else if (dim == 2)
    return coord[2] * points_per_dim[0] * points_per_dim[1] +
           coord[1] * points_per_dim[0] + coord[0];
  else
    __builtin_unreachable();
}

// Overload for xyz_to_linear for square grids
inline std::size_t xyz_to_linear(std::vector<std::size_t> coord,
                                 std::size_t n_points_per_dim) {
  std::vector<std::size_t> points_per_dim(coord.size());
  for (auto &entry : points_per_dim)
    entry = n_points_per_dim;

  return xyz_to_linear(coord, points_per_dim);
}

inline std::vector<std::size_t>
linear_to_xyz(std::size_t idx, std::vector<std::size_t> n_points_per_dim) {
  const auto dim = n_points_per_dim.size();
  std::vector<std::size_t> coord(dim);
  if (dim == 1) {
    coord[0] = idx;
  } else if (dim == 2) {
    coord[1] = idx / n_points_per_dim[0];
    coord[0] = idx % n_points_per_dim[0];
  } else if (dim == 3) {
    throw std::runtime_error("dim = not implemented yet");
  } else {
    __builtin_unreachable();
  }

  return coord;
}
} // namespace parmgmc
