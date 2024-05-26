# Parallel Multigrid Monte Carlo (MGMC)
ParMGMC is a C library implementation of the Multigrid Monte Carlo method in PETSc to sample from high-dimensional Gaussian distributions on distributed memory machines.

## Dependencies
ParMGMC has the following dependencies:
- An MPI installation (e.g., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/))
- [PETSc](https://petsc.org/)  (tested with version 3.21, anything >= 3.19 should work)
- pybind11 and petsc4py if the Python bindings should be enabled.

### Building the library
CMake is used to configure and simplify building ParMGMC. To compile the library and examples run
```bash
$ git clone https://github.com/nilsfriess/ParMGMC.git
$ cd ParMGMC
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
```
To specify a custom compiler (e.g., a MPI compiler wrapper) add `-DCMAKE_C_COMPILER=mpicc` and/or `-DCMAKE_C_COMPILER=mpicc` (the library itself is written in C but can also be used from C++ as shown in the examples in the `examples_cpp` folder).

## Python bindings
The library can also be used from Python. If the necessary requirements are satisfied, pass `-DPARMGMC_ENABLE_PYTHON_BINDINGS=True` during the CMake config to enable Python bindings. See the files in the `examples_py` folder.
