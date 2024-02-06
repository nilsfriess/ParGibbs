# Parallel Multigrid Monte Carlo (MGMC)
ParMGMC is a C++17 library built on top of [PETSc](https://petsc.org/) to sample from high-dimensional Gaussian distributions on distributed memory machines.

## Installation and usage
ParMGMC is a header-only library and as such requires no installation. If you want to use the library, simply copy the folder `parmgmc` in the `include` directory somewhere on your computer. ParMGMC has the following dependencies:
- An MPI installation (e.g., OpenMPI or MPICH)
- PETSc (version >= 3.17)

To compile and run the tests and/or the examples, you need CMake. To compile and run the tests first clone the repository and configure the build with
```bash
$ git clone https://github.com/nilsfriess/ParMGMC.git
$ cd ParMGMC
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
```
Then, run
```
$ make tests
$ ./tests/tests
```
to run the serial tests and
```
$ mpirun -np 4 ./tests/tests [mpi]
```
to run the parallel test suite.
