#pragma once

#include <cstddef>

#include "coordinate.hh"

namespace pargibbs {
// template <std::size_t dim>
// inline std::size_t coord_to_linear(Coordinate<dim> coord,
//                                    std::size_t n_points_per_dim) {
//   if constexpr (dim == 1)
//     return coord.x;
//   else if constexpr (dim == 2)
//     return coord.x * n_points_per_dim + coord.y;
//   else // dim == 3
//     return coord.x * n_points_per_dim * n_points_per_dim +
//            coord.y * n_points_per_dim + coord.z;
// }

template <std::size_t dim>
inline Coordinate<dim> linear_to_coord(std::size_t idx,
                                       std::size_t n_points_per_dim) {
  Coordinate<dim> coord;
  if constexpr (dim == 1) {
    coord.x = idx;
  } else if (dim == 2) {
    coord.y = idx / n_points_per_dim;
    coord.x = idx % n_points_per_dim;
  } else if (dim == 3) {
    static_assert(dim != 3, "linear_to_coord not implemented yet for dim == 3");
  }

  return coord;
}

template <std::size_t dim>
void scale_coord(Coordinate<dim> &coord, double scaling) {
  for (std::size_t i = 0; i < dim; ++i)
    coord[i] *= scaling;
}

  template <std::size_t dim>
  void switch_color(Coordinate<dim> &coordinate) {
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
