#pragma once

#include <algorithm>
#include <ostream>
#include <petsc.h>
#include <petscsftypes.h>
#include <petscsystypes.h>

#include <set>
#include <vector>

namespace parmgmc {
struct DependentNode {
  PetscInt index;
  PetscScalar value;

  bool operator<(const DependentNode &other) const { return index < other.index; }
};

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

  std::set<PetscInt> lower_dependents;
  std::set<PetscInt> higher_dependents;

  std::set<DependentNode> received_dependents;

  bool done;

  void insert_dependent(PetscInt index, PetscScalar value) {}

  bool is_ready(bool forward_pass = true) const {
    const auto &compare_to = forward_pass ? higher_dependents : lower_dependents;

    if (received_dependents.size() != compare_to.size())
      return false;

    auto it1 = received_dependents.begin();
    auto it2 = compare_to.begin();
    for (; it1 != received_dependents.end() && it2 != compare_to.end(); ++it1, ++it2)
      if (it1->index != *it2)
        return false;

    return true;
  }
};

struct BotMidTopPartition {
  VecScatter topscatter; // Get the values needed to process the top nodes
  VecScatter botscatter; // Get the values needed to process the bot nodes

  std::vector<PetscInt> topscatter_indices;
  std::vector<PetscInt> botscatter_indices;

  Vec top_sctvec;
  Vec bot_sctvec;

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

  void reset_mid_nodes() {
    for (auto &node : mid) {
      node.done = false;
      node.received_dependents_idx.clear();
      node.received_dependents_val.clear();
    }
  }

  bool mid_nodes_done() const {
    return std::all_of(mid.begin(), mid.end(), [](const MidNode &node) { return node.done; });
  }
};

inline std::ostream &operator<<(std::ostream &out, const BoundaryNode &node) {
  out << node.index << " -> [" << node.neighbor.owner << ": " << node.neighbor.index << "]";
  return out;
}

} // namespace parmgmc