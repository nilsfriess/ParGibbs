/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include <pybind11/pybind11.h>

// #include "petsc_caster.hh"
#include "parmgmc/parmgmc.h"

namespace py = pybind11;

PYBIND11_MODULE(pymgmc, m)
{
  ParMGMCInitialize();
};
