#include <pybind11/pybind11.h>

// #include "petsc_caster.hh"
#include "parmgmc/parmgmc.h"

namespace py = pybind11;

PYBIND11_MODULE(pymgmc, m)
{
  ParMGMCInitialize();
};
