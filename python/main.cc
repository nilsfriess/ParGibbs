#include <pybind11/pybind11.h>
#include <random>

#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"
#include "petsc_caster.hh"

namespace py = pybind11;
using namespace parmgmc;

PYBIND11_MODULE(pymgmc, m) {
  using Engine = std::mt19937;
  auto *engine = new Engine(std::random_device{}());

  m.def("seedEngine", [=](int seed) { engine->seed(seed); });

  py::class_<LinearOperator, std::shared_ptr<LinearOperator>>(m, "LinearOperator")
      .def(py::init([](Mat m) {
        return LinearOperator{m, false};
      }))
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
};
