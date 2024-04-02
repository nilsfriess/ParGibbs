#pragma once

#include <mpi.h>

namespace parmgmc {
class Timer {
public:
  Timer() { reset(); }

  void reset() { starttime = MPI_Wtime(); }
  [[nodiscard]] double elapsed() const { return MPI_Wtime() - starttime; }

private:
  double starttime;
};
} // namespace parmgmc
