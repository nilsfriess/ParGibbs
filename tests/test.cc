#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/dm_hierarchy.hh"

#include <catch2/catch_session.hpp>
#include <petscviewer.h>

int main(int argc, char *argv[]) {
  parmgmc::PetscHelper::init(argc, argv);

  const auto res = Catch::Session().run(argc, argv);

  return res;
}
