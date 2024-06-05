****************************
Building the Python bindings
****************************

To enable the Python bindings ``petsc4py`` must be installed (e.g., by passing ``--with-petsc4py`` when configuring PETSc), and `pybind11 <https://github.com/pybind/pybind11>`_ must be available. Simply pass ``-DPARMGMC_ENABLE_PYTHON_BINDINGS=On`` during CMake config and then run ``make pymgmc``. This generates a shared library object in ``build/python``.

It is not possible yet to install this library somewhere, it can only be used from the folder ``build/python``.
