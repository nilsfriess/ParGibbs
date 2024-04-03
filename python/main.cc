
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
#include <stdexcept>

namespace py = pybind11;
using namespace parmgmc;

PYBIND11_MODULE(pymgmc, m) {
  using Engine = std::mt19937;
  auto *engine = new Engine(std::random_device{}());

  m.def("seedEngine", [=](int seed) { engine->seed(seed); });

  m.def("fillVecRand", [=](Vec v) {
    PetscFunctionBeginUser;
    PetscCallVoid(fillVecRand(v, *engine));
    PetscFunctionReturnVoid();
  });

  py::class_<LinearOperator, std::shared_ptr<LinearOperator>>(m, "LinearOperator")
      .def(py::init([](Mat m) { return std::make_shared<LinearOperator>(m, false); }))
      .def("colorMatrix", py::overload_cast<>(&LinearOperator::colorMatrix),
           "Generate a colouring for the matrix stored in the operator")
      .def("colorMatrix", py::overload_cast<DM>(&LinearOperator::colorMatrix),
           "Generate a red/black colouring for the matrix stored in the operator");

  using GibbsSampler = MulticolorGibbsSampler<Engine>;
  py::class_<GibbsSampler, std::shared_ptr<GibbsSampler>>(m, "GibbsSampler")
      .def(py::init([=](const std::shared_ptr<LinearOperator> &op) {
        return std::make_shared<GibbsSampler>(op, engine);
      }))
      .def("sample", [](const std::shared_ptr<GibbsSampler> &self, Vec rhs, Vec sample) {
        PetscFunctionBeginUser;

        PetscCallVoid(self->sample(sample, rhs));

        PetscFunctionReturnVoid();
      });

  py::class_<DMHierarchy, std::shared_ptr<DMHierarchy>>(m, "DMHierarchy")
      .def(py::init([](DM coarseSpace, std::size_t nLevels) {
        return std::make_shared<DMHierarchy>(coarseSpace, nLevels, false);
      }))
      .def("getDM", &DMHierarchy::getDm)
      .def("getCoarse", &DMHierarchy::getCoarse)
      .def("getFine", &DMHierarchy::getFine, py::return_value_policy::reference)
      .def("numLevels", &DMHierarchy::numLevels);

  py::enum_<MGMCSmoothingType>(m, "SmoothingType")
      .value("ForwardBackward", MGMCSmoothingType::ForwardBackward)
      .value("Symmetric", MGMCSmoothingType::Symmetric);

  py::enum_<MGMCCycleType>(m, "CycleType")
      .value("V", MGMCCycleType::V)
      .value("W", MGMCCycleType::W);

  using MGMCSampler = MultigridSampler<Engine>;
  py::class_<MGMCSampler, std::shared_ptr<MGMCSampler>>(m, "MGMCSampler")
      .def(py::init([=](const std::shared_ptr<LinearOperator> &fineOperator,
                        const std::shared_ptr<DMHierarchy> &dmHierarchy,
                        MGMCSmoothingType smoothingType, MGMCCycleType cycleType,
                        std::size_t smoothingSteps) {
             MGMCParameters params;
             params.smoothingType = smoothingType;
             params.cycleType = cycleType;
             params.nSmooth = smoothingSteps;

             return std::make_shared<MGMCSampler>(fineOperator, dmHierarchy, engine, params);
           }),
           py::arg("fineOperator"), py::arg("dmHierarchy"), py::kw_only(),
           py::arg("smoothing") = MGMCSmoothingType::ForwardBackward,
           py::arg("cycle") = MGMCCycleType::V, py::arg("smoothingSteps") = 2)
      .def("sample", [](const std::shared_ptr<MGMCSampler> &self, Vec rhs, Vec sample) {
        PetscFunctionBeginUser;

        PetscCallVoid(self->sample(sample, rhs));

        PetscFunctionReturnVoid();
      });

  using HogwildSampler = HogwildGibbsSampler<Engine>;
  py::class_<HogwildSampler, std::shared_ptr<HogwildSampler>>(m, "HogwildSampler")
      .def(py::init([=](const std::shared_ptr<LinearOperator> &op) {
        return std::make_shared<HogwildSampler>(op, engine);
      }))
      .def("sample", [](const std::shared_ptr<HogwildSampler> &self, Vec rhs, Vec sample) {
        PetscFunctionBeginUser;

        PetscCallVoid(self->sample(sample, rhs));

        PetscFunctionReturnVoid();
      });
};
