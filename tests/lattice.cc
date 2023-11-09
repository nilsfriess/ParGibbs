#include <gtest/gtest.h>

#include "pargibbs/lattice/lattice.hh"

TEST(LatticeCoarsening, Lattice2d5x5) {
  namespace pg = pargibbs;

  pg::Lattice lattice(2, 5);

  auto coarse_lattice = lattice.coarsen();

  EXPECT_EQ(coarse_lattice.get_vertices_per_dim(), 3);
  EXPECT_EQ(coarse_lattice.get_n_total_vertices(), 9);
}

TEST(LatticeConstruction, Lexicographic) {
  pargibbs::Lattice lattice(2,
                            3,
                            pargibbs::ParallelLayout::None,
                            pargibbs::LatticeOrdering::Lexicographic);

  std::vector<int> adj_idx_expected = {0, 2, 5, 7, 10, 14, 17, 19, 22, 24};
  std::vector<int> adj_vert_expected = {1, 3, 0, 2, 4, 1, 5, 4, 6, 0, 3, 5,
                                        7, 1, 4, 8, 2, 7, 3, 6, 8, 4, 7, 5};

  EXPECT_EQ(lattice.adj_idx, adj_idx_expected);
  EXPECT_EQ(lattice.adj_vert, adj_vert_expected);
}

TEST(LatticeConstruction, RedBlack) {
  pargibbs::Lattice lattice(2,
                            3,
                            pargibbs::ParallelLayout::None,
                            pargibbs::LatticeOrdering::RedBlack);

  std::vector<int> adj_idx_expected = {0, 2, 4, 8, 10, 12, 15, 18, 21, 24};
  std::vector<int> adj_vert_expected = {5, 6, 5, 7, 6, 7, 8, 5, 8, 6, 8, 7,
                                        0, 1, 2, 2, 3, 0, 2, 4, 1, 3, 4, 2};

  EXPECT_EQ(lattice.adj_idx, adj_idx_expected);
  EXPECT_EQ(lattice.adj_vert, adj_vert_expected);
}
