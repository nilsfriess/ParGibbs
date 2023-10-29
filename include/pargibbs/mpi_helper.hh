#pragma once

#include <utility>

namespace pargibbs {
struct mpi_helper {
  mpi_helper(int *argc = nullptr, char ***argv = nullptr);
  ~mpi_helper();

  static std::pair<int, int> get_size_rank();
  static int get_rank();
  static int get_size();
  static int debug_rank();
  static bool is_debug_rank();

private:
  static void assert_initalised();
};
}; // namespace pargibbs
