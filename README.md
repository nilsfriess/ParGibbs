# Parallel Multigrid Monte Carlo (MGMC)
ParMGMC is a C library implementation of the Multigrid Monte Carlo method in PETSc to sample from high-dimensional Gaussian distributions on distributed memory machines.

## Dependencies
ParMGMC has the following dependencies:
- An MPI installation (e.g., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/))
- [PETSc](https://petsc.org/)  (tested with version 3.21, anything >= 3.19 should work) built with C/Pardiso enabled
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

After Intel MKL has been installed can be configured by running
```bash
./configure --with-mkl_cpardiso   \
            --with-mkl_pardiso    \
            --with-blas-lapack-dir=/opt/intel/oneapi/mkl/latest/lib
```
in the PETSc directory (the path to the Intel MKL might differ depending on the platform). For details, see [PETSc documentation](https://petsc.org/release/install/).

If the Python bindings should be enabled, pybind11 and `petsc4py` are also required. `petsc4py` can be built by passing `--with-petsc4py` during configure.

## Building the library
To build the ParMGMC library CMake is required. Run
```bash
$ git clone https://github.com/nilsfriess/ParMGMC.git
$ cd ParMGMC
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
$ make
```
To specify a custom compiler (e.g., a MPI compiler wrapper) add `-DCMAKE_C_COMPILER=mpicc`.

### Installing the library
To install the library to some directory, add `-DCMAKE_INSTALL_PREFIX=/path/to/install` during CMake configuration. Then run `make install` to copy the compiled library and headers to the specified directory. This also generates a `pkg-config` file that can be used to simplify using this library in other projects. If the environment variable `PKG_CONFIG_PATH` contains both the path to `parmgmc.pc` (located in `/path/to/install/lib/pkgconfig`) and the path to PETSc's `pkg-config` file (located in `/path/to/petsc/lib/pkgconfig`), then a program using ParMGMC can be compiled with
```bash
gcc main.c -o main $(pkg-config --cflags --libs petsc parmgmc)
```

## Python bindings (experimental)
The library has experimental support for usage from Python. Pass `-DPARMGMC_ENABLE_PYTHON_BINDINGS=True` during the CMake config to enable Python bindings.
