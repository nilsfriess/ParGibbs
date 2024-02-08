#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/dm_hierarchy.hh"

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) {
  parmgmc::PetscHelper::init(argc, argv);

  return Catch::Session().run(argc, argv);
}
