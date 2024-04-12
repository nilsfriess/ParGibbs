#include <petscsystypes.h>
#include <pybind11/pybind11.h>

#include <random>

#include "parmgmc/common/helpers.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/hogwild.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "petsc_caster.hh"

#include <petscerror.h>

namespace py = pybind11;
using namespace parmgmc;

PYBIND11_MODULE(pymgmc, m) {
  using Engine = std::mt19937;
  static Engine engine{std::random_device{}()};

  m.def("seedEngine", [&](int seed) { engine.seed(seed); });

  m.def("fillVecRand", [&](Vec v) {
    PetscFunctionBeginUser;
    PetscCallVoid(fillVecRand(v, engine));
    PetscFunctionReturnVoid();
  });

  py::class_<LinearOperator>(m, "LinearOperator")
      .def(py::init([](Mat m) {
        return LinearOperator{m, false};
      }))
      .def("getMat", &LinearOperator::getMat, py::return_value_policy::reference)
      .def("colorMatrix", py::overload_cast<>(&LinearOperator::colorMatrix),
           "Generate a colouring for the matrix stored in the operator")
      .def("colorMatrix", py::overload_cast<DM>(&LinearOperator::colorMatrix),
           "Generate a red/black colouring for the matrix stored in the operator");

  py::enum_<GibbsSweepType>(m, "GibbsSweepType")
      .value("Forward", GibbsSweepType::Forward)
      .value("Backward", GibbsSweepType::Backward)
      .value("Symmetric", GibbsSweepType::Symmetric);

  using GibbsSampler = MulticolorGibbsSampler<Engine>;
  py::class_<GibbsSampler>(m, "GibbsSampler")
      .def(py::init([&](LinearOperator &op, double omega, GibbsSweepType sweepType) {
             return GibbsSampler{op, engine, omega, sweepType};
           }),
           py::arg("linearOperator"), py::kw_only(), py::arg("omega") = 1.,
           py::arg("sweepType") = GibbsSweepType::Forward)
      .def("sample", [](GibbsSampler &self, Vec rhs, Vec sample) {
        PetscFunctionBeginUser;

        PetscCallVoid(self.sample(sample, rhs));

        PetscFunctionReturnVoid();
      });

  py::class_<DMHierarchy>(m, "DMHierarchy")
      .def(py::init([](DM coarseSpace, PetscInt nLevels) {
        return DMHierarchy{coarseSpace, nLevels, false};
      }))
      .def("getDM", &DMHierarchy::getDm)
      .def("getCoarse", &DMHierarchy::getCoarse)
      .def("getFine", &DMHierarchy::getFine, py::return_value_policy::automatic_reference)
      .def("numLevels", &DMHierarchy::numLevels);

  py::enum_<MGMCSmoothingType>(m, "SmoothingType")
      .value("ForwardBackward", MGMCSmoothingType::ForwardBackward)
      .value("Symmetric", MGMCSmoothingType::Symmetric);

  py::enum_<MGMCCycleType>(m, "CycleType")
      .value("V", MGMCCycleType::V)
      .value("W", MGMCCycleType::W);

  py::enum_<MGMCCoarseSamplerType>(m, "CoarseSamplerType")
      .value("SymmetricGibbs", MGMCCoarseSamplerType::Standard)
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
      .value("Cholesky", MGMCCoarseSamplerType::Cholesky)
#endif
      ;

  using MGMCSampler = MultigridSampler<Engine>;
  py::class_<MGMCSampler>(m, "MGMCSampler")
      .def(py::init([&](LinearOperator &fineOperator, const DMHierarchy &dmHierarchy,
                        MGMCSmoothingType smoothingType, MGMCCycleType cycleType,
                        std::size_t smoothingSteps, MGMCCoarseSamplerType coarseSamplerType) {
             MGMCParameters params;
             params.smoothingType = smoothingType;
             params.cycleType = cycleType;
             params.nSmooth = smoothingSteps;
             params.coarseSamplerType = coarseSamplerType;

             return MGMCSampler{fineOperator, dmHierarchy, engine, params};
           }),
           py::arg("fineOperator"), py::arg("dmHierarchy"), py::kw_only(),
           py::arg("smoothing") = MGMCSmoothingType::ForwardBackward,
           py::arg("cycle") = MGMCCycleType::V, py::arg("smoothingSteps") = 2,
           py::arg("coarseSampler") =
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
               MGMCCoarseSamplerType::Cholesky
#else
	       MGMCCoarseSamplerType::Standard
#endif
           )
      .def("sample", [](MGMCSampler &self, Vec rhs, Vec sample) {
        PetscFunctionBeginUser;

        PetscCallVoid(self.sample(sample, rhs));

        PetscFunctionReturnVoid();
      });

  using HogwildSampler = HogwildGibbsSampler<Engine>;
  py::class_<HogwildSampler>(m, "HogwildSampler")
      .def(py::init([&](const LinearOperator &op) {
        return HogwildSampler{op, engine};
      }))
      .def("sample", [](HogwildSampler &self, Vec rhs, Vec sample) {
        PetscFunctionBeginUser;

        PetscCallVoid(self.sample(sample, rhs));

        PetscFunctionReturnVoid();
      });
};
