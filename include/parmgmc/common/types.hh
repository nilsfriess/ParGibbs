#pragma once

#include <petsc.h>
#include <petscsftypes.h>
#include <petscsystypes.h>

#include <vector>

namespace parmgmc {
struct BotMidTopPartition {
  VecScatter high_to_low;
  VecScatter low_to_high;

  std::vector<PetscInt> top;
  std::vector<PetscInt> bot;

  std::vector<PetscInt> interior1;
  std::vector<PetscInt> interior2;
};
} // namespace parmgmc