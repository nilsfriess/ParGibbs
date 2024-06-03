# Parallel Multigrid Monte Carlo (MGMC)
ParMGMC is a C library implementation of the Multigrid Monte Carlo method in PETSc to sample from high-dimensional Gaussian distributions on distributed memory machines.

## Dependencies
ParMGMC has the following dependencies:
- An MPI installation (e.g., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/))
- [PETSc](https://petsc.org/)  (tested with version 3.21, anything >= 3.19 should work) built with MUMPS and C/Pardiso enabled
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

After Intel MKL has been installed, the remaining dependencies can be installed via PETSc's `./configure` command by running
```bash
./configure --download-mpich      \
            --with-petsc4py       \
            --download-mumps      \
            --download-scalapack  \
            --with-mkl_cpardiso   \
            --with-mkl_pardiso    \
            --with-blas-lapack-dir=/opt/intel/oneapi/mkl/latest/lib
```
after downloading PETSc (the path to the Intel MKL might differ depending on the platform).

If the Python bindings should be enabled, pybind11 and petsc4py are also required.

## Building the library
CMake is used to configure and simplify building ParMGMC. To compile the library and examples run
```bash
$ git clone https://github.com/nilsfriess/ParMGMC.git
$ cd ParMGMC
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
$ make
```
To specify a custom compiler (e.g., a MPI compiler wrapper) add `-DCMAKE_CXX_COMPILER=mpicxx` and/or `-DCMAKE_C_COMPILER=mpicc` (the library itself is written in C but can also be used from C++ as shown in the examples in the `examples_cpp` folder).

### Installing the library
To install the library to some directory, add `-DCMAKE_INSTALL_PREFIX=/path/to/install` during CMake configuration. Then run `make install` to copy the compiled library and headers to the specified directory. This also generates a `pkg-config` file that can be used to simplify using this library in other projects. If the environment variable `PKG_CONFIG_PATH` contains both the path to `parmgmc.pc` (located in `/path/to/install/lib/pkgconfig`) and the path to PETSc's `pkg-config` file (located in `/path/to/petsc/lib/pkgconfig`), then a program using ParMGMC can be compiled with
```bash
gcc main.c -o main $(pkg-config --cflags --libs petsc parmgmc)
```

## Python bindings
The library can also be used from Python. If the necessary requirements are satisfied, pass `-DPARMGMC_ENABLE_PYTHON_BINDINGS=True` during the CMake config to enable Python bindings. See the files in the `examples_py` folder for examples.
