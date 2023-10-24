#pragma once

#include <cstddef>

#include "coordinate.hh"

namespace pargibbs {
template <std::size_t dim>
inline std::size_t xyz_to_linear(std::array<std::size_t, dim> coord,
                                 std::array<std::size_t, dim> points_per_dim) {
  if constexpr (dim == 1)
    return coord[0];
  else if constexpr (dim == 2)
    return coord[1] * points_per_dim[0] + coord[0];
  else // dim == 3
    return coord[2] * points_per_dim[0] * points_per_dim[1] +
           coord[1] * points_per_dim[0] + coord[0];
}

// Overload for xyz_to_linear for square grids
template <std::size_t dim>
inline std::size_t xyz_to_linear(std::array<std::size_t, dim> coord,
                                 std::size_t n_points_per_dim) {
  std::array<std::size_t, dim> points_per_dim;
  for (auto &entry : points_per_dim)
    entry = n_points_per_dim;

  return xyz_to_linear(coord, points_per_dim);
}

template <std::size_t dim>
inline std::array<std::size_t, dim>
linear_to_xyz(std::size_t idx, std::array<std::size_t, dim> n_points_per_dim) {
  std::array<std::size_t, dim> coord;
  if constexpr (dim == 1) {
    coord[0] = idx;
  } else if (dim == 2) {
    coord[1] = idx / n_points_per_dim[0];
    coord[0] = idx % n_points_per_dim[0];
  } else if (dim == 3) {
    static_assert(dim != 3, "linear_to_xyz not implemented yet for dim == 3");
  }

  return coord;
}

// template <std::size_t dim>
// inline Coordinate<dim> linear_to_xyz(std::size_t idx,
//                                        std::size_t n_points_per_dim) {
//   std::array<std::size_t, dim> per_dim;
//   for (auto &entry : per_dim)
//     entry = n_points_per_dim;
//   return linear_to_coord(idx, per_dim);
// }

template <std::size_t dim>
void scale_coord(Coordinate<dim> &coord, double scaling) {
  for (std::size_t i = 0; i < dim; ++i)
    coord[i] *= scaling;
}

template <std::size_t dim> void switch_color(Coordinate<dim> &coordinate) {
  if (coordinate.type == CoordinateType::None)
    return;

  if (coordinate.type == CoordinateType::Red)
    coordinate.type = CoordinateType::Black;
  else if (coordinate.type == CoordinateType::Black)
    coordinate.type = CoordinateType::Red;
  else
    __builtin_unreachable();
}
} // namespace pargibbs
