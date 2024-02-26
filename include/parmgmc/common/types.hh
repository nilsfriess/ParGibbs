#pragma once

#include <ostream>
#include <petsc.h>
#include <petscsftypes.h>
#include <petscsystypes.h>

#include <vector>

namespace parmgmc {
struct RemoteNode {
  RemoteNode(PetscInt index, PetscMPIInt owner) : index{index}, owner{owner} {}
  RemoteNode() = default;

  PetscInt index;
  PetscMPIInt owner;

  bool operator==(const RemoteNode &other) const {
    return index == other.index && owner == other.owner;
  }
};

/** A vertex of the lattice/ FE graph which has a neighboring vertex that is
 * onwed by another MPI process. */
struct BoundaryNode {
  BoundaryNode(PetscInt index, PetscInt neighbor_global_index, PetscMPIInt neighbor_rank)
      : index{index}, neighbor{neighbor_global_index, neighbor_rank} {}
  BoundaryNode() = default;

  /// Local index on owning process
  PetscInt index;

  /// Neighboring node
  RemoteNode neighbor;

  bool operator==(const BoundaryNode &other) const {
    return index == other.index && neighbor == other.neighbor;
  }
};

struct MidNode {
  PetscInt index;

  /// Neighboring nodes on other ranks
  std::vector<RemoteNode> neighbors;

  std::vector<PetscInt> lower_dependents;
  std::vector<PetscInt> higher_dependents;
};

struct BotMidTopPartition {
  VecScatter high_to_low;
  VecScatter low_to_high;

  std::vector<BoundaryNode> top;
  std::vector<BoundaryNode> bot;

  std::vector<PetscInt> interior1;
  std::vector<PetscInt> interior2;

  std::vector<MidNode> mid;

  void clear() {
    top.clear();
    bot.clear();
    interior1.clear();
    interior2.clear();
    mid.clear();
  }
};

inline std::ostream &operator<<(std::ostream &out, const BoundaryNode &node) {
  out << node.index << " -> [" << node.neighbor.owner << ": " << node.neighbor.index << "]";
  return out;
}

} // namespace parmgmc