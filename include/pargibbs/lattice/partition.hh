#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <ostream>
#include <stack>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if USE_METIS
#include <metis.h>
#endif

#include "pargibbs/common/log.hh"
#include "pargibbs/lattice/helpers.hh"
#include "types.hh"

namespace pargibbs::detail {
template <std::size_t dim> using nd_id = std::array<std::size_t, dim>;

template <std::size_t dim> struct partition {
  nd_id<dim> start;
  nd_id<dim> size;
  std::size_t weight; // only used within `worb`, can be ignored outside
};

template <class Lattice, std::size_t dim>
inline std::vector<typename Lattice::IndexT>
partition_to_mpimap(const std::vector<partition<dim>> &partitions,
                    const Lattice &lattice) {
  std::vector<typename Lattice::IndexT> mpiowner(
      lattice.get_n_total_vertices());

  for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
    const auto &partition = partitions[p_idx];

    std::size_t tot_points = 1;
    for (auto e : partition.size)
      tot_points *= e;

    for (std::size_t idx = 0; idx < tot_points; ++idx) {
      // Convert linear index within partition to global coordinate
      auto local_coord = linear_to_xyz(idx, partition.size);
      for (std::size_t d = 0; d < dim; ++d)
        local_coord[d] += partition.start[d];

      // And convert this to a global linear index
      auto global_idx =
          xyz_to_linear(local_coord, lattice.get_vertices_per_dim());

      mpiowner.at(global_idx) = p_idx;
    }
  }

  return mpiowner;
}

// Performs weighted orthogonal recursive bisection to partition a rectangular
// grid in dimension `dim` with `dimensions[d]` nodes along dimension `d` into
// `n_partitions` partitions. The resulting partition approximately minimises
// the total length of the boundaries between the partitions (which
// appproximately minimises MPI communication).
//
// Returns the list of partitions where each partition holds its `start`
// coordinate and its `size`.
// TODO: Currently only supports 2D grids.
template <class Lattice>
inline std::vector<typename Lattice::IndexT> worb(const Lattice &lattice,
                                                  std::size_t n_partitions) {
  constexpr auto dim = Lattice::Dim;

  static_assert(dim == 2, "Only dim == 2 supported currently");

  std::array<std::size_t, dim> dimensions;
  for (auto &entry : dimensions)
    entry = lattice.get_vertices_per_dim();

  std::vector<partition<dim>> final_partitions;
  final_partitions.reserve(n_partitions);
  std::stack<partition<dim>> unfinished_partitions;

  partition<dim> initial_partition;
  initial_partition.start = nd_id<dim>{0};
  initial_partition.size = dimensions;
  initial_partition.weight = n_partitions;

  if (n_partitions == 1) {
    final_partitions.push_back(std::move(initial_partition));
    return partition_to_mpimap(final_partitions, lattice);
  }

  unfinished_partitions.push(std::move(initial_partition));

  // Recursively traverse list of partitions that are not small enough yet
  while (not unfinished_partitions.empty()) {
    auto cur_partition = unfinished_partitions.top();
    unfinished_partitions.pop();

    std::size_t total_points = 1;
    for (auto d : cur_partition.size)
      total_points *= d;

    // TODO: This assumes 2D
    auto cut_dim = std::distance(
        cur_partition.size.begin(),
        std::max_element(cur_partition.size.begin(), cur_partition.size.end()));
    auto other_dim = 1 - cut_dim; // cut_dim = 0 => other_dim = 1 and vice versa

    auto weight_left =
        static_cast<std::size_t>(std::floor(cur_partition.weight / 2.));
    auto n_left = total_points * weight_left / cur_partition.weight;

    auto weight_right =
        static_cast<std::size_t>(std::ceil(cur_partition.weight / 2.));
    // auto n_right = total_points * weight_right / cur_partition.weight;

    partition<dim> left;
    left.size[cut_dim] = n_left / cur_partition.size[other_dim];
    left.size[other_dim] = cur_partition.size[other_dim];
    left.start = cur_partition.start;
    left.weight = weight_left;

    partition<dim> right;
    right.size[cut_dim] = cur_partition.size[cut_dim] - left.size[cut_dim];
    right.size[other_dim] = cur_partition.size[other_dim];
    right.start = cur_partition.start;
    right.start[cut_dim] += left.size[cut_dim];
    right.weight = weight_right;

    if (left.weight == 1)
      final_partitions.push_back(std::move(left));
    else
      unfinished_partitions.push(std::move(left));

    if (right.weight == 1)
      final_partitions.push_back(std::move(right));
    else
      unfinished_partitions.push(std::move(right));
  }

  return partition_to_mpimap(final_partitions, lattice);
}

// Partitions a rectangular grid in dimension `dim` with `dimensions[d]` nodes
// along dimension `d` into `n_partitions` partitions by slicing the domain into
// rows of approximately equal size along dimension `dim` (if the domain cannot
// be distributed equally, one partition is assigned a larger subdomain; no load
// balancing is performed).
//
// Returns the list of partitions where each partition holds its `start`
// coordinate and its `size`.
template <class Lattice>
inline std::vector<typename Lattice::IndexT>
block_row(const Lattice &lattice, std::size_t n_partitions) {
  constexpr auto dim = Lattice::Dim;
  std::array<std::size_t, dim> dimensions;
  for (auto &entry : dimensions)
    entry = lattice.get_vertices_per_dim();

  std::vector<partition<dim>> partitions;
  partitions.reserve(n_partitions);

  const std::size_t len = dimensions[dim - 1] / n_partitions;
  if (len == 0) {
    PARGIBBS_DEBUG << "Error: Cannot partition along dimension of length "
                   << dimensions[dim - 1] << " into " << n_partitions
                   << " partitions.\n";
    throw std::runtime_error(
        "Error during block-row partitioning. Too many partitions requested");
  }

  for (std::size_t i = 0; i < n_partitions; ++i) {
    partition<dim> part;

    for (std::size_t d = 0; d < dim - 1; ++d)
      part.start[d] = 0;
    part.start[dim - 1] = i * len;

    for (std::size_t d = 0; d < dim - 1; ++d)
      part.size[d] = dimensions[d];
    part.size[dim - 1] = len;

    // Last partition might be bigger
    if (i == n_partitions - 1)
      part.size[dim - 1] = dimensions[dim - 1] - part.start[dim - 1];

    partitions.push_back(std::move(part));
  }

  return partition_to_mpimap(partitions, lattice);
}

#if USE_METIS
template <class Lattice>
inline std::vector<typename Lattice::IndexT> metis(const Lattice &lattice,
                                                   std::size_t n_partitions) {
  static_assert(std::is_same_v<typename Lattice::IndexT, idx_t>,
                "METIS requires to use `idx_t` type as IndexType in Lattice.");
  std::vector<idx_t> mpiowner(lattice.get_n_total_vertices());

  idx_t nvtxs = lattice.get_n_total_vertices();
  idx_t ncon = 1;
  idx_t nparts = n_partitions;

  idx_t objval = 0;

  // Const-casting is fine here, METIS does not change these pointers
  auto *xadj = const_cast<idx_t *>(lattice.adj_idx.data());
  auto *adjncy = const_cast<idx_t *>(lattice.adj_vert.data());

  METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL,
                           &nparts, NULL, NULL, NULL, &objval, mpiowner.data());

  return mpiowner;
}
#endif

template <class Lattice>
inline std::vector<typename Lattice::IndexT>
make_partition(const Lattice &lattice, std::size_t n_partitions) {
  switch (lattice.layout) {
  case ParallelLayout::None: {
    std::vector<typename Lattice::IndexT> mpiowner(
        lattice.get_n_total_vertices());
    std::fill(mpiowner.begin(), mpiowner.end(), 0);
    return mpiowner;
  }
  case ParallelLayout::BlockRow:
    return block_row(lattice, n_partitions);

  case ParallelLayout::WORB:
    return worb(lattice, n_partitions);

#ifdef USE_METIS
  case ParallelLayout::METIS:
    return metis(lattice, n_partitions);
#endif

  default:
    __builtin_unreachable();
  }
}

}; // namespace pargibbs::detail
