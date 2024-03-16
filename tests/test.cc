#include "parmgmc/common/petsc_helper.hh"

#include <catch2/catch_session.hpp>
#include <petscoptions.h>

int main(int argc, char *argv[]) {
  parmgmc::PetscHelper::init(argc, argv);

  // If we pass -log_view, Catch2 will throw an error since it doesn't recognize
  // it as a valid command line option, so we just remove it after PETSc has
  // parsed it.
  if (argc > 1 && !std::strcmp(argv[1], "-ksp_monitor"))
    argv[1][0] = 0;

  return Catch::Session().run(argc, argv);
}
