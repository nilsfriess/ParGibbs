#pragma once

#include <cstddef>
#include <ostream>
#include <cassert>

namespace pargibbs {

enum class CoordinateType { None, Red, Black };

struct CoordinateBase {
  CoordinateType type = CoordinateType::None;

  CoordinateBase() = default;
  CoordinateBase(CoordinateType type) : type(type){};
};

template <std::size_t dim> struct Coordinate : CoordinateBase {};

template <> struct Coordinate<1> : CoordinateBase {
  double x;

  // Returns x, even if `dimension > 0`
  double &operator[](std::size_t /*dimension*/) { return x; }
  double operator[](std::size_t /*dimension*/) const { return x; }
};

template <> struct Coordinate<2> : CoordinateBase {
  double x;
  double y;

  // Returns x, if `dimension == 0` and y otherwise
  double &operator[](std::size_t dimension) {
    if (dimension == 0)
      return x;
    else
      return y;
  }

  double operator[](std::size_t dimension) const {
    if (dimension == 0)
      return x;
    else
      return y;
  }
};

template <> struct Coordinate<3> : CoordinateBase {
  double x;
  double y;
  double z;

  // Returns x, if `dimension == 0` y if `dimension == 1`, and z otherwise
  double &operator[](std::size_t dimension) {
    if (dimension == 0)
      return x;
    else if (dimension == 1)
      return y;
    else
      return z;
  }

  double operator[](std::size_t dimension) const {
    if (dimension == 0)
      return x;
    else if (dimension == 1)
      return y;
    else
      return z;
  }
};
}; // namespace pargibbs

inline std::ostream &operator<<(std::ostream &out,
                                pargibbs::CoordinateType type) {
  using enum pargibbs::CoordinateType;
  switch (type) {
  case None:
    out << "none";
    break;
  case Red:
    out << "red";
    break;
  case Black:
    out << "black";
    break;
  default:
    assert(false);
  }
  return out;
}

template <std::size_t dim>
inline std::ostream &operator<<(std::ostream &out,
                                const pargibbs::Coordinate<dim> &coord) {
  out << coord.type << "( ";
  for (std::size_t i = 0; i < dim; ++i)
    out << coord[i] << " ";
  out << ")";
  return out;
}
