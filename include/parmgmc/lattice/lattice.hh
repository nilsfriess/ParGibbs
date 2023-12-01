#pragma once

#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

#include "parmgmc/mpi_helper.hh"
#include "types.hh"

namespace parmgmc {
/* Type of a specific vertex. Internal vertices are those owned by the current
 * MPI process. Border vertices are internal vertices that have at least one
 * neighbouring index that is owned by another MPI index. Ghost vertices are
 * owned by another MPI process but have at least one of our vertices as
 * neighbours. Any is the union of Internal and Ghost vertices. */
enum class VertexType { Internal, Border, Ghost, Any };

class Lattice {
public:
  using IndexType = int;

  /* Iterator class that allows to easily loop over certain types of lattice
     vertices. Should be used with the Lattice::vertices(VertexType) method.
   */
  class Iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = IndexType;
    using pointer = value_type *;
    using reference = const value_type &;

    Iterator(const Lattice *lattice, VertexType type, std::size_t id)
        : lattice{lattice}, type{type}, index{id} {
      while (index < lattice->own_vertices.size() and
             (not is_right_type_index(index)))
        index++;
    }

    reference operator*() const { return lattice->own_vertices[index]; }

    Iterator &operator++() {
      index++;
      while (index < lattice->own_vertices.size() and
             (not is_right_type_index(index)))
        index++;

      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.index == b.index;
    }

    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return !(a == b);
    }

  private:
    bool is_right_type_index(std::size_t i) const {
      if (type == VertexType::Any)
        return true;

      if (type == VertexType::Internal) {
        // Skip Ghost vertices
        if (lattice->vertex_types[i] == VertexType::Ghost)
          return false;
      } else if (type == VertexType::Border) {
        // Skip everything but border indices
        if (lattice->vertex_types[i] != VertexType::Border)
          return false;
      } else if (type == VertexType::Ghost) {
        // Skip both kinds of internal vertices
        if (lattice->vertex_types[i] == VertexType::Internal or
            lattice->vertex_types[i] == VertexType::Border)
          return false;
      }

      return true;
    }

    const Lattice *lattice;
    VertexType type;

    std::size_t index;
  };

  class VerticesProxy {
  public:
    VerticesProxy(const Lattice *lattice, VertexType type)
        : lattice{lattice}, type{type} {}

    Iterator begin() { return Iterator{lattice, type, 0}; }
    Iterator end() {
      return Iterator{lattice, type, lattice->own_vertices.size()};
    }

  private:
    const Lattice *lattice;
    VertexType type;
  };

  Lattice(std::size_t dim, IndexType vertices_per_dim,
          ParallelLayout layout = ParallelLayout::None);

  IndexType get_vertices_per_dim() const { return n_vertices_per_dim; }
  IndexType get_n_total_vertices() const {
    return std::pow(n_vertices_per_dim, dim);
  }
  std::size_t get_n_own_vertices() const { return n_internal_vertices; }
  std::size_t get_n_border_vertices() const { return n_border_vertices; }

  Lattice coarsen() const;

  VerticesProxy vertices(VertexType type = VertexType::Internal) const {
    return VerticesProxy(this, type);
  }

  ParallelLayout get_layout() const { return layout; };

  using vec_ref = const std::vector<IndexType> &;
  std::pair<vec_ref, vec_ref> get_adjacency_lists() const {
    return std::pair<vec_ref, vec_ref>(adj_idx, adj_vert);
  }

  std::size_t dim;

  // Maps indices to mpi ranks
  std::vector<IndexType> mpiowner;

private:
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

  */
  // clang-format on
  std::vector<IndexType> adj_idx;
  std::vector<IndexType> adj_vert;

  // Indices of vertices that the current MPI rank owns
  std::vector<IndexType> own_vertices;
  // Indices of vertices that the current MPI rank owns and that have to be
  // communicated to some other MPI rank
  std::vector<IndexType> border_vertices;

  std::vector<VertexType> vertex_types;

  IndexType n_vertices_per_dim;

  double meshwidth;

  ParallelLayout layout;

  // Number of internal (including border) vertices.
  std::size_t n_internal_vertices;
  // Number of border vertices.
  std::size_t n_border_vertices;

  Lattice() = default;

  void setup_graph();
  void print_partition() const;
  void update_vertices();
};

} // namespace parmgmc
